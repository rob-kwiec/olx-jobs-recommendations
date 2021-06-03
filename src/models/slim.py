"""Module implementing SLIM model"""

from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.utils.testing import ignore_warnings
from tqdm import tqdm

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender


class SLIM(BaseRecommender, DataLoaderSaver):
    """
    SLIM model proposed in "SLIM: Sparse Linear Methods for Top-N Recommender Systems"
    """

    def __init__(
        self,
        alpha=0.0001,
        l1_ratio=0.5,
        iterations=3,
        show_progress=True,
    ):

        super().__init__()

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.iterations = iterations

        # data
        self.interactions = None
        self.train_ui = None
        self.user_id_code = None
        self.user_code_id = None
        self.item_code_id = None
        self.similarity_matrix = None
        self.show_progress = show_progress

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
            (
                np.ones(len(data)),
                (data["user_code"], data["item_code"]),
            )
        )

    def fit_per_item(self, column_id):
        """
        Fits ElasticNet per item
        :param column_id: Id of column to setup as predicted value
        :return: coefficients of the ElasticNet model
        """
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection="random",
            max_iter=self.iterations,
        )
        # set to zeros all entries in the given column of train_ui
        y = self.train_ui[:, column_id].A
        start_indptr = self.train_ui.indptr[column_id]
        end_indptr = self.train_ui.indptr[column_id + 1]
        column_ratings = self.train_ui.data[start_indptr:end_indptr].copy()
        self.train_ui.data[start_indptr:end_indptr] = 0

        # learn item-item similarities
        model.fit(self.train_ui, y)

        # return original ratings to train_ui
        self.train_ui.data[start_indptr:end_indptr] = column_ratings

        return model.sparse_coef_.T

    @ignore_warnings(category=ConvergenceWarning)
    def fit(self):
        """
        Fit the model
        """
        self.train_ui = self.train_ui.tocsc()

        with ThreadPool() as thread_pool:
            coefs = list(
                tqdm(
                    thread_pool.imap(self.fit_per_item, range(self.train_ui.shape[1])),
                    disable=not self.show_progress,
                )
            )

        self.similarity_matrix = sparse.hstack(coefs).tocsr()

        self.train_ui = self.train_ui.tocsr()

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
        if u_code is not None:

            exclude_items = []
            if filter_out_interacted_items:
                exclude_items = self.train_ui.indices[
                    self.train_ui.indptr[u_code] : self.train_ui.indptr[u_code + 1]
                ]

            scores = self.train_ui[u_code] * self.similarity_matrix
            u_recommended_items = scores.indices[
                (-scores.data).argsort()[: n_recommendations + len(exclude_items)]
            ]

            u_recommended_items = [
                self.item_code_id[i]
                for i in u_recommended_items
                if i not in exclude_items
            ][:n_recommendations]
        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )
