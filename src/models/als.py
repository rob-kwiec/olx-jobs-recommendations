"""Module implementing a wrapper for the ALS model"""

from functools import partial
from multiprocessing.pool import ThreadPool

import implicit
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender


class ALS(BaseRecommender, DataLoaderSaver):
    """
    Wrapper over ALS model
    """

    def __init__(
        self,
        factors=100,
        regularization=0.01,
        use_gpu=False,
        iterations=15,
        event_weights_multiplier=100,
        show_progress=True,
    ):
        """
        Source of descriptions:
        https://github.com/benfred/implicit/blob/master/implicit/als.py

        Alternating Least Squares
        A Recommendation Model based on the algorithms described in the paper
        'Collaborative Filtering for Implicit Feedback Datasets'
        with performance optimizations described in 'Applications of the
        Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.'

        Parameters
        ----------
        factors : int, optional
            The number of latent factors to compute
        regularization : float, optional
            The regularization factor to use
        use_gpu : bool, optional
            Fit on the GPU if available, default is to run on CPU
        iterations : int, optional
            The number of ALS iterations to use when fitting data
        event_weights_multiplier: int, optional
            The multiplier of weights.
            Used to find a tradeoff between the importance of interacted and not interacted items.
        """

        super().__init__()

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            use_gpu=use_gpu,
            iterations=iterations,
        )

        self.event_weights_multiplier = event_weights_multiplier

        # data
        self.interactions = None
        self.train_ui = None
        self.user_id_code = None
        self.user_code_id = None
        self.item_code_id = None

        self.show_progress = show_progress

    def preprocess(self):
        """
        Prepare interactions dataset for training model
        """

        data = self.interactions.copy()
        data["event_value"] = self.event_weights_multiplier

        self.user_code_id = dict(enumerate(data["user"].unique()))
        self.user_id_code = {v: k for k, v in self.user_code_id.items()}
        data["user_code"] = data["user"].apply(self.user_id_code.get)

        self.item_code_id = dict(enumerate(data["item"].unique()))
        item_id_code = {v: k for k, v in self.item_code_id.items()}
        data["item_code"] = data["item"].apply(item_id_code.get)

        self.train_ui = sparse.csr_matrix(
            (data["event_value"], (data["user_code"], data["item_code"]))
        )

    def fit(self):
        """
        Fit the model
        """
        self.model.fit(self.train_ui.T, show_progress=self.show_progress)

    def recommend(
        self,
        target_users,
        n_recommendations,
        filter_out_interacted_items=True,
        show_progress=True,
    ) -> pd.DataFrame:
        """
            Recommends n_recommendations items for target_users
        :return:
            pd.DataFrame (user, item_1, item_2, ..., item_n)
        """

        with ThreadPool() as thread_pool:
            recommendations = list(
                tqdm(
                    thread_pool.imap(
                        partial(
                            self.recommend_per_user,
                            n_recommendations=n_recommendations,
                            filter_out_interacted_items=filter_out_interacted_items,
                        ),
                        target_users,
                    ),
                    disable=not self.show_progress,
                )
            )

        return pd.DataFrame(recommendations)

    def recommend_per_user(
        self, user, n_recommendations, filter_out_interacted_items=True
    ):
        """
        Recommends n items per user
        :param user: User id
        :param n_recommendations: Number of recommendations
        :param filter_out_interacted_items: boolean value to filter interacted items
        :return: list of format [user_id, item1, item2 ...]
        """
        u_code = self.user_id_code.get(user)
        u_recommended_items = []
        if u_code is not None:

            u_recommended_items = list(
                zip(
                    *self.model.recommend(
                        u_code,
                        self.train_ui,
                        N=n_recommendations,
                        filter_already_liked_items=filter_out_interacted_items,
                    )
                )
            )[0]

            u_recommended_items = [self.item_code_id[i] for i in u_recommended_items]

        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )
