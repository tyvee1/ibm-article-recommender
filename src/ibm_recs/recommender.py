"""Article recommendations for the IBM Watson Studio community.

One class, ArticleRecommender, wraps the methods built in the project
notebook so they can be used from other code or a web app:

* rank based:      the most-viewed articles (for anyone, and for new users)
* user-user:       articles read by the most similar users
* content based:   articles with the most similar titles
* SVD:             articles with the most similar latent features

It is fitted from the interaction table (user email, article id, title),
and every method returns plain Python lists, so callers never handle
pandas objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

UNKNOWN_USER = "unknown_user"


def file_sha256(path: str) -> str:
    """Hex SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_interactions(path: str, expected_sha256: str | None = None) -> pd.DataFrame:
    """Read the interaction CSV and map emails to whole-number user ids.

    INPUT:
    path - CSV with columns article_id, title, email (as the course ships it)
    expected_sha256 - when given, the file must have exactly this hash, so a
                      changed or substituted file is refused before use

    OUTPUT:
    df - columns article_id (int), title (str), user_id (int). Missing emails
         become one "unknown_user", as in the notebook; ids count from 1 in
         order of first appearance.

    Raises ValueError when the hash does not match or a column is missing.
    """
    if expected_sha256 is not None:
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"{path}: sha256 {actual} is not the expected {expected_sha256}")
    raw = pd.read_csv(path, dtype={"article_id": int, "title": str, "email": str})
    missing = {"article_id", "title", "email"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    emails = raw["email"].fillna(UNKNOWN_USER)
    coded = {email: n for n, email in enumerate(emails.unique(), start=1)}
    return pd.DataFrame(
        {
            "article_id": raw["article_id"].astype(int),
            "title": raw["title"].astype(str),
            "user_id": [coded[e] for e in emails],
        }
    )


@dataclass(frozen=True)
class Recommendation:
    """One recommended article: its id, title and the score it was ranked by."""

    article_id: int
    title: str
    score: float


class ArticleRecommender:
    """Fit once on the interaction table, then ask for recommendations.

    Parameters
    ----------
    latent_features : number of SVD features used for article similarity
        (200, as chosen in the notebook).
    random_state : seed for the two randomized SVD fits, so results repeat.
    """

    def __init__(self, latent_features: int = 200, random_state: int = 42):
        self.latent_features = latent_features
        self.random_state = random_state
        self._fitted = False

    # ------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> "ArticleRecommender":
        """Build every model from the interactions; returns self.

        INPUT:
        df - columns article_id, title, user_id (see load_interactions)
        """
        need = {"article_id", "title", "user_id"}
        if not need <= set(df.columns):
            raise ValueError(f"df needs columns {sorted(need)}")
        if df.empty:
            raise ValueError("df has no interactions")
        self.titles = df.drop_duplicates("article_id").set_index("article_id")["title"].to_dict()
        views = df.groupby("article_id").size()
        # Most views first, ties by article id: the Part II ranking.
        self.view_ranking = sorted(views.index.tolist(), key=lambda a: (-int(views[a]), int(a)))
        self.views = {int(a): int(v) for a, v in views.items()}
        self.view_rank = {a: i for i, a in enumerate(self.view_ranking)}

        counts = df.groupby(["user_id", "article_id"]).size().unstack(fill_value=0)
        self.user_item = (counts > 0).astype(np.int64)
        self.user_ids = [int(u) for u in self.user_item.index]
        self.article_ids = [int(a) for a in self.user_item.columns]
        self._user_row = {u: i for i, u in enumerate(self.user_ids)}
        self._article_col = {a: i for i, a in enumerate(self.article_ids)}
        self._matrix = self.user_item.to_numpy()
        self._user_sizes = self._matrix.sum(axis=1)

        self._fit_content()
        self._fit_svd()
        self._fitted = True
        return self

    def _fit_content(self) -> None:
        """Title vectors: words, word pairs and character pieces, then LSA."""
        titles = [self.titles[a] for a in self.article_ids]
        words = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        features = sparse.hstack([words.fit_transform(titles), chars.fit_transform(titles)]).tocsr()
        components = min(200, features.shape[1] - 1, len(titles) - 1)
        lsa = make_pipeline(TruncatedSVD(n_components=components, random_state=self.random_state), Normalizer(copy=False))
        self._title_vectors = lsa.fit_transform(features)

    def _fit_svd(self) -> None:
        """Latent features of each article from a truncated SVD of user_item.

        As in the notebook, the SVD is fitted with as many components as the
        matrix has columns and the first `latent_features` are kept. Fitting
        only `latent_features` components directly is a different randomized
        approximation and changes which articles come out closest.
        """
        full = min(self._matrix.shape)
        svd = TruncatedSVD(n_components=full, n_iter=5, random_state=self.random_state)
        svd.fit(self._matrix)
        vt = svd.components_[: min(self.latent_features, full)]
        # A column that is only rounding noise has no meaningful direction:
        # zero it so it scores 0 against everything, on every machine.
        lengths = np.linalg.norm(vt, axis=0)
        self._vt = np.where(lengths < 1e-10 * lengths.max(), 0.0, vt)

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() first")

    # ------------------------------------------------------------ lookups
    def has_user(self, user_id: int) -> bool:
        """True when the user has at least one interaction."""
        self._require_fit()
        return int(user_id) in self._user_row

    def has_article(self, article_id: int) -> bool:
        """True when the article has at least one interaction."""
        self._require_fit()
        return int(article_id) in self._article_col

    def title(self, article_id: int) -> str:
        """The article's title; KeyError for an unknown id."""
        self._require_fit()
        return self.titles[int(article_id)]

    def user_articles(self, user_id: int) -> list[int]:
        """Ids of the articles a user has opened, smallest first."""
        self._require_fit()
        row = self._matrix[self._user_row[int(user_id)]]
        return [a for a, seen in zip(self.article_ids, row) if seen]

    def _wrap(self, ids: list[int], scores: list[float]) -> list[Recommendation]:
        return [Recommendation(int(a), self.titles[int(a)], float(s)) for a, s in zip(ids, scores)]

    # ------------------------------------------------------------ rank based
    def top_articles(self, n: int = 10) -> list[Recommendation]:
        """The n most-viewed articles (each view counts, repeats included)."""
        self._require_fit()
        ids = self.view_ranking[: max(0, int(n))]
        return self._wrap(ids, [self.views[a] for a in ids])

    # ------------------------------------------------------------ user-user
    def similar_users(self, user_id: int) -> list[tuple[int, float, int]]:
        """Every other user as (user_id, cosine similarity, articles read).

        Sorted by similarity, then by articles read, both descending, then by
        user id. The similarity is shared / sqrt(|a| * |b|) from whole-number
        counts, so it rounds the same way on every machine.
        """
        self._require_fit()
        i = self._user_row[int(user_id)]
        target = self._matrix[i]
        shared = self._matrix @ target
        similarity = shared / np.sqrt(self._user_sizes * int(target.sum()))
        rows = [
            (u, float(similarity[j]), int(self._user_sizes[j]))
            for j, u in enumerate(self.user_ids)
            if j != i
        ]
        rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
        return rows

    def recommend_for_user(self, user_id: int, n: int = 10) -> list[Recommendation]:
        """n articles for a user: from their most similar users, or popularity.

        For a known user: walk neighbours in similar_users order and take the
        articles the user has not read, most viewed first within each
        neighbour (the notebook's user_user_recs_part2). An unknown or new
        user gets the most-viewed articles.
        """
        self._require_fit()
        n = max(0, int(n))
        if not self.has_user(user_id):
            return self.top_articles(n)
        seen = set(self.user_articles(user_id))
        picks: list[int] = []
        for neighbour, _, _ in self.similar_users(user_id):
            new = [a for a in self.user_articles(neighbour) if a not in seen and a not in picks]
            picks.extend(sorted(new, key=self.view_rank.__getitem__))
            if len(picks) >= n:
                break
        picks = picks[:n]
        return self._wrap(picks, [self.views[a] for a in picks])

    # ------------------------------------------------------------ content
    def similar_by_title(self, article_id: int, n: int = 10) -> list[Recommendation]:
        """n articles whose titles are most similar (cosine, LSA title vectors)."""
        self._require_fit()
        i = self._article_col[int(article_id)]
        sims = self._title_vectors @ self._title_vectors[i]
        order = sorted((j for j in range(len(sims)) if j != i), key=lambda j: (-sims[j], self.article_ids[j]))
        ids = [self.article_ids[j] for j in order[: max(0, int(n))]]
        return self._wrap(ids, [float(sims[self._article_col[a]]) for a in ids])

    # ------------------------------------------------------------ SVD
    def similar_by_readers(self, article_id: int, n: int = 10) -> list[Recommendation]:
        """n articles with the most similar SVD latent features (cosine)."""
        self._require_fit()
        i = self._article_col[int(article_id)]
        sims = cosine_similarity(self._vt.T[[i]], self._vt.T).ravel()
        order = sorted((j for j in range(len(sims)) if j != i), key=lambda j: (-sims[j], self.article_ids[j]))
        ids = [self.article_ids[j] for j in order[: max(0, int(n))]]
        return self._wrap(ids, [float(sims[self._article_col[a]]) for a in ids])
