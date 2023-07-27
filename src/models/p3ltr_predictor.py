"""Module implementing a a P3LTR Predictor"""

from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm

from .base import BaseRecommender
from ..data.initializer import DataLoaderSaver


class P3LTRPredictor(BaseRecommender, DataLoaderSaver):
    """
    P3LTR model predictor
    """

    def __init__(
        self,
        feature_encoders,
        feature_preprocessor,
    ):
        super().__init__()

        self.feature_encoders = feature_encoders
        self.feature_preprocessor = feature_preprocessor

        # data
        self.interactions = None
        self.user_id_code = None
        self.user_code_id = None
        self.item_code_id = None
        self.p = None
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

        features = self.feature_preprocessor.preprocess(data)
        is_direction_ui = features["is_direction_ui"]

        # define transition matrices p
        self.p = 3 * [None]
        with torch.no_grad():
            for layer, mask in [
                (0, is_direction_ui),
                (1, ~is_direction_ui),
                (2, is_direction_ui),
            ]:
                event_values_ui = np.array(
                    self.feature_encoders[layer](
                        destination_degree=features["destination_degree"][mask],
                        recency=features["recency"][mask],
                        events=features["events"][mask],
                    ).flatten()
                )
                self.p[layer] = sparse.csr_matrix(
                    (
                        event_values_ui,
                        (features["source"][mask], features["destination"][mask]),
                    )
                )

    def fit(self):
        """
        Fit the model
        """
        self.similarity_matrix = self.p[1] * self.p[2]

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
                exclude_items = self.p[0].indices[
                    self.p[0].indptr[u_code] : self.p[0].indptr[u_code + 1]
                ]
            scores = self.p[0][u_code] * self.similarity_matrix
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
