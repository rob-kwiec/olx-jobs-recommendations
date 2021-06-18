"""Module implementing a RP3Beta model"""

from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from tqdm import tqdm

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender


class RP3Beta(BaseRecommender, DataLoaderSaver):
    """
    RP3Beta model proposed in the paper "Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive
    Applications". In our implementation we perform direct computations on sparse matrices instead of random walks
    approximation.
    """

    def __init__(
        self,
        alpha=1,
        beta=0,
        show_progress=True,
    ):

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # data
        self.interactions = None
        self.train_ui = None
        self.user_id_code = None
        self.user_code_id = None
        self.item_code_id = None
        self.show_progress = show_progress
        self.p_ui = None
        self.similarity_matrix = None

    def preprocess(self):
        """
        Prepare interactions dataset for training model
        """

        data = self.interactions.copy()

        self.user_code_id = dict(enumerate(data["user"].unique()))
        self.user_id_code = {v: k for k, v in self.user_code_id.items()}
        data["user_code"] = data["user"].apply(self.user_id_code.get)

        self.item_code_id = dict(enumerate(data["item"].unique()))
        item_id_code = {v: k for k, v in self.item_code_id.items()}
        data["item_code"] = data["item"].apply(item_id_code.get)

        self.train_ui = sparse.csr_matrix(
            (np.ones(len(data)), (data["user_code"], data["item_code"]))
        )

    def fit(self):
        """
        Fit the model
        """
        # Define Pui
        self.p_ui = normalize(self.train_ui, norm="l1", axis=1).power(self.alpha)

        # Define Piu
        p_iu = normalize(
            self.train_ui.transpose(copy=True).tocsr(), norm="l1", axis=1
        ).power(self.alpha)

        self.similarity_matrix = p_iu * self.p_ui
        item_orders = (self.train_ui > 0).sum(axis=0).A.ravel()

        self.similarity_matrix *= sparse.diags(1 / item_orders.clip(min=1) ** self.beta)

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

            exclude_items = []
            if filter_out_interacted_items:
                exclude_items = self.train_ui.indices[
                    self.train_ui.indptr[u_code] : self.train_ui.indptr[u_code + 1]
                ]

            scores = self.p_ui[u_code] * self.similarity_matrix
            u_recommended_items = scores.indices[
                (-scores.data).argsort()[: n_recommendations + len(exclude_items)]
            ]

            u_recommended_items = [
                self.item_code_id[i]
                for i in u_recommended_items
                if i not in exclude_items
            ]

            u_recommended_items = u_recommended_items[:n_recommendations]

        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )
