# Article recommendations for the IBM Watson Studio community

Author: Ty

Udacity Data Scientist Nanodegree, Course 5 (Unsupervised Machine Learning
and Recommendation Systems): the "Recommendations with IBM" project.

## 1. Motivation

IBM Watson Studio has a large community library of articles, notebooks and
datasets. Users read some of them, and nothing tells them which others they
would find useful. This project builds four ways to recommend articles from
the record of who read what (5149 users, 714 articles), and
checks how well each one works:

- **Rank based:** the most-viewed articles. This is all a brand-new user can get.
- **User-user collaborative filtering:** articles read by the users whose reading is most similar.
- **Content based:** articles with similar titles, using TF-IDF, LSA and k-means clusters.
- **Matrix factorisation:** articles with similar latent features from a singular value decomposition (SVD) of the user-article matrix.

## 2. What is in this repository

| Path | What it is |
| --- | --- |
| `Recommendations_with_IBM.ipynb` | The project notebook, executed, with every answer and test result |
| `Recommendations_with_IBM.html` | The same notebook exported to HTML, for reading without Jupyter |
| `project_tests.py`, `top_5.p`, `top_10.p`, `top_20.p` | The course's tests and expected answers, used by the notebook (unchanged) |
| `data/user-item-interactions.csv` | The course's interaction data (unchanged) |
| `src/ibm_recs/` | The recommenders as an installable Python package with one class, `ArticleRecommender`, and a small web app (`web.py`, `templates/`) |
| `tests/` | Tests for the package and the web app |
| `pyproject.toml` | Package definition (`pip install -e .`) |
| `requirements.txt` | The exact library versions the notebook and package were run with |
| `LICENSE` | MIT licence for the code written for this project |

## 3. Installation

Python 3.12 or newer is needed; the work was run on Python 3.14.6. From the
repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead of the `source` line.

Libraries: pandas 3.1.1, NumPy 2.5.5, SciPy 1.18.1, scikit-learn
1.9.1, matplotlib 3.11.2, Jupyter Notebook 7.6.5, Flask
3.1.3 (web app) and pytest 9.2.0 (tests).

## 4. How to run it, and what it found

**The notebook.** Open it in Jupyter:

```bash
jupyter notebook Recommendations_with_IBM.ipynb
```

or run every cell without opening a browser:

```bash
jupyter nbconvert --to notebook --execute --inplace Recommendations_with_IBM.ipynb
```

It takes about a minute. Every course test in it prints its passing message.

**The tests.**

```bash
pytest
```

They check that the package gives the same answers as the notebook and the
course's tests, and that the web app refuses bad input, escapes what it
shows, and listens only on this machine.

**The web app.**

```bash
python -m ibm_recs.web
```

Then open http://127.0.0.1:8050 to get recommendations for a user id,
similar articles for an article id, or to search titles. Press Ctrl-C to
stop it. It listens on this computer only.

**Results** (from the executed notebook):

- **Popularity is easy to inflate.** Of 45993 interactions, 12311 are users re-opening an article they had already read. Counted by distinct readers rather than views, a different article comes first (article 1330, read by 467 users, against article 1429, the most viewed).
- **The course's "6th most similar user" answer rests on rounding.** 38 users share exactly the same similarity to user 2. The course's answer is decided by the last binary digit of the similarity. The rubric's rule (more interactions first) would rank user 4230 (9 articles) above it, but rounding puts that user at position 43. The notebook computes similarity in a way that rounds the same on every machine.
- **Title clusters.** The inertia curve bends most at 51 clusters. 49 of the 714 titles have no words left after the course's vocabulary filter, so they all fall into one cluster. The stronger content recommender in the notebook (word and character TF-IDF with nearest neighbours) leaves no title empty and can recommend all 714 articles, against 480 for the clusters.
- **SVD on held-out data.** On the course's train/test split, 662 of the 682 test users cannot be scored at all (cold start). For the 20 who can, accuracy is the wrong measure: predicting "no interaction" everywhere scores 0.9810, higher than SVD at any number of features (its best is 0.9784). Ranked, SVD with 200 latent features found 33 new reads in 200 recommendations, against 16 for popularity. The difference in precision at 10 is 0.085, with a 95% bootstrap interval of 0.035 to 0.140, which stays above zero. SVD beats popularity from 70 to 390 latent features. With only 20 users this evidence is thin.

## 5. Licensing, authors and acknowledgements

- Code written for this project (the notebook's solutions, `src/` and `tests/`): MIT licence, see `LICENSE`. Author: Ty.
- The starter notebook's questions, `project_tests.py`, the `.p` answer files and the interaction data come from Udacity's project repository, [udacity/dsnd-recommendation-systems-project](https://github.com/udacity/dsnd-recommendation-systems-project). They are used under Udacity's educational-content terms and are not covered by the MIT licence.
- The interaction data records reading on the IBM Watson Studio community platform, provided by IBM to Udacity for this project.
