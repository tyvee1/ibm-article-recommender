"""The class must give the notebook's answers, which the course's tests check."""

import shutil

import pytest

from ibm_recs import ArticleRecommender, load_interactions
from ibm_recs.web import COURSE_DATA_SHA256

COURSE_TOP_10 = {1314, 1429, 1293, 1427, 1162, 1364, 1304, 1170, 1431, 1330}
COURSE_SVD_4 = {1199, 1068, 486, 1202, 176, 1120, 244, 793, 58, 132}


def test_counts_match_the_course(model):
    """5,149 users (with the unknown user) and 714 articles, as sol_1_test expects."""
    assert len(model.user_ids) == 5149
    assert len(model.article_ids) == 714
    assert len(model.user_articles(1)) == 36


def test_top_articles_are_the_course_top_10(model):
    assert {r.article_id for r in model.top_articles(10)} == COURSE_TOP_10
    views = [r.score for r in model.top_articles(20)]
    assert views == sorted(views, reverse=True)


def test_similar_users_match_sol_5(model):
    """The three neighbours sol_5_test checks: 1st of user 1, 6th of 2, 10th of 131."""
    assert model.similar_users(1)[0][0] == 3933
    assert model.similar_users(2)[5][0] == 3586
    assert model.similar_users(131)[9][0] == 383


def test_user_recommendations_match_the_notebook(model):
    """user_user_recs_part2(20, 10) in the notebook gives this list."""
    got = [r.article_id for r in model.recommend_for_user(20, 10)]
    assert got == [1293, 1162, 1185, 1165, 254, 1271, 1328, 1410, 40, 1402]
    assert not set(got) & set(model.user_articles(20))


def test_new_user_gets_the_most_viewed(model):
    assert not model.has_user(0)
    assert {r.article_id for r in model.recommend_for_user(0, 10)} == COURSE_TOP_10


def test_svd_neighbours_of_article_4_match_the_course(model):
    picks = model.similar_by_readers(4, 10)
    assert {r.article_id for r in picks} == COURSE_SVD_4
    assert 4 not in {r.article_id for r in picks}


def test_title_neighbours_exclude_the_article_and_are_ranked(model):
    picks = model.similar_by_title(25, 10)
    assert len(picks) == 10 and 25 not in {r.article_id for r in picks}
    scores = [r.score for r in picks]
    assert scores == sorted(scores, reverse=True)


def test_a_title_with_no_whole_word_still_gets_neighbours(model):
    """'readr 1.0.0' had no features in the course's TF-IDF; it must here."""
    readr = next(a for a in model.article_ids if model.title(a) == "readr 1.0.0")
    assert model.similar_by_title(readr, 5)[0].score > 0


def test_a_changed_data_file_is_refused(data_path, tmp_path):
    copy = tmp_path / "data.csv"
    shutil.copyfile(data_path, copy)
    load_interactions(str(copy), expected_sha256=COURSE_DATA_SHA256)  # silent: genuine file
    with open(copy, "a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="sha256"):
        load_interactions(str(copy), expected_sha256=COURSE_DATA_SHA256)


def test_using_an_unfitted_model_fails_clearly():
    with pytest.raises(RuntimeError):
        ArticleRecommender().top_articles(5)
