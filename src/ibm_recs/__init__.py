"""ibm_recs: article recommendations for the IBM Watson Studio community.

    from ibm_recs import ArticleRecommender, load_interactions
    model = ArticleRecommender().fit(load_interactions("data/user-item-interactions.csv"))
    model.recommend_for_user(1, n=10)
"""

from .recommender import ArticleRecommender, Recommendation, file_sha256, load_interactions

__all__ = ["ArticleRecommender", "Recommendation", "file_sha256", "load_interactions"]
__version__ = "1.0.0"
