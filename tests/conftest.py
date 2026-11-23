"""Shared fixtures: the course data and one fitted recommender per session."""

import os

import pytest

from ibm_recs import ArticleRecommender, load_interactions

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "user-item-interactions.csv")


@pytest.fixture(scope="session")
def data_path():
    """Path to the course's interaction file."""
    return os.path.abspath(DATA)


@pytest.fixture(scope="session")
def model(data_path):
    """The recommender fitted on the full course data, once per test run."""
    return ArticleRecommender().fit(load_interactions(data_path))
