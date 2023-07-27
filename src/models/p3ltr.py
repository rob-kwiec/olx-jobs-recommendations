"""Module implementing a P3LTR model"""

from .base import BaseRecommender
from .p3ltr_predictor import P3LTRPredictor
from .p3ltr_trainer import P3LTRTrainer
from ..data.initializer import DataLoaderSaver


class P3LTR(BaseRecommender, DataLoaderSaver):
    """
    P3 Learning to Rank model
    """

    def __init__(
        self,
        learning_rate=0.1,
        regularization=0.1,
        batch_size=100,
        iterations=100,
        top_k=10,
        user_selection="random",
        loss="log_ratio",
    ):
        super().__init__()

        # data
        self.interactions = None
        self.trainer = P3LTRTrainer(
            learning_rate=learning_rate,
            regularization=regularization,
            batch_size=batch_size,
            iterations=iterations,
            top_k=top_k,
            user_selection=user_selection,
            loss=loss,
        )
        self.predictor = None

    def preprocess(self):
        """
        Prepare interactions dataset for training model
        """
        self.trainer.set_interactions(self.interactions)
        self.trainer.preprocess()

    def fit(self):
        """
        Fit the model
        """
        self.trainer.fit()

        self.predictor = P3LTRPredictor(
            feature_encoders=self.trainer.feature_encoders,
            feature_preprocessor=self.trainer.feature_preprocessor,
        )
        self.predictor.set_interactions(self.interactions)
        self.predictor.preprocess()
        self.predictor.fit()

    def recommend(
        self,
        target_users,
        n_recommendations,
        filter_out_interacted_items=True,
    ):
        """
            Recommends n_recommendations items for target_users
        :return:
            pd.DataFrame (user, item_1, item_2, ..., item_n)
        """
        return self.predictor.recommend(
            target_users=target_users,
            n_recommendations=n_recommendations,
            filter_out_interacted_items=filter_out_interacted_items,
        )
