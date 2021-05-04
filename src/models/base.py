"""
Definition of base recommender class
"""


class BaseRecommender:
    """Base recommender interface"""

    def preprocess(self):
        """Implement any needed input data preprocessing"""
        raise NotImplementedError

    def fit(self):
        """Implement model fitter"""
        raise NotImplementedError

    def recommend(self, *args, **kwargs):
        """Implement recommend method
        Should return a DataFrame containing
        * user_id: id of the user for whom we provide recommendations
        * n columns containing item recommendations (or None if missing)
        """
        raise NotImplementedError
