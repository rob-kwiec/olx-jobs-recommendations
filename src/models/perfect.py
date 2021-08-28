"""Module implementing Perfect recommender, which recommends items from test set.
Used only to check best possible values of evaluation metrics on a given dataset."""

import random
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender


class Perfect(BaseRecommender, DataLoaderSaver):
    """
    Perfect recommender, which recommends items from test set
    """

    def __init__(
        self,
        path_test,
        restrict_items_to_train=False,
        restrict_items_to_d3=False,
        show_progress=True,
    ):
        """
        Parameters
        ----------
        restrict_items_to_train : boolean, optional
            Whether to recommend only items which appeared in train set interactions
        restrict_items_to_d3: boolean, optional
            Whether to recommend only items with distance 3 from a given user in user-item bipartite matrix

        """
        super().__init__()

        self.path_test = path_test
        self.restrict_items_to_train = restrict_items_to_train
        self.restrict_items_to_d3 = restrict_items_to_d3

        # data
        self.interactions = None
        self.train_ui = None
        self.user_id_code = None
        self.user_code_id = None
        self.item_code_id = None
        self.item_id_code = None
        self.mapping_user_test_items = None
        self.similarity_matrix = None

        self.show_progress = show_progress

    def preprocess(self):
        """
        Preprocessing
        """
        self.mapping_user_test_items = dict()

        data = DataLoaderSaver()
        data.load_interactions(self.path_test)
        test_df = data.interactions

        if self.restrict_items_to_train:
            # to make all users appear in self.mapping_user_test_items keys
            train_users = self.interactions["user"].drop_duplicates()
            self.mapping_user_test_items = dict(
                zip(train_users, [[]] * len(train_users))
            )

            train_items = self.interactions[["item"]].drop_duplicates()
            test_df = test_df.merge(train_items)

        self.mapping_user_test_items.update(
            test_df.groupby("user")
            .agg({"item": lambda x: list(set(x))})["item"]
            .to_dict()
        )

        if self.restrict_items_to_d3:
            data = self.interactions.copy()

            self.user_code_id = dict(enumerate(data["user"].unique()))
            self.user_id_code = {v: k for k, v in self.user_code_id.items()}
            data["user_code"] = data["user"].apply(self.user_id_code.get)

            self.item_code_id = dict(enumerate(data["item"].unique()))
            self.item_id_code = {v: k for k, v in self.item_code_id.items()}
            data["item_code"] = data["item"].apply(self.item_id_code.get)

            self.train_ui = sparse.csr_matrix(
                (np.ones(len(data)), (data["user_code"], data["item_code"]))
            )

    def fit(self):
        """
        Fit the model
        """
        if self.restrict_items_to_d3:
            self.similarity_matrix = self.train_ui.transpose().tocsr() * self.train_ui

    def recommend(
        self,
        target_users,
        n_recommendations,
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
                        ),
                        target_users,
                    ),
                    disable=not self.show_progress,
                )
            )

        return pd.DataFrame(recommendations)

    def recommend_per_user(
        self,
        user,
        n_recommendations,
    ):
        """
        Recommends n items per user
        :param user: User id
        :param n_recommendations: Number of recommendations
        :return: list of format [user_id, item1, item2 ...]
        """

        if self.restrict_items_to_d3:
            # mapping through item_id_code is expensive so we do it only if necessary
            allowed_items = list(
                map(self.item_id_code.get, self.mapping_user_test_items[user])
            )
            u_code = self.user_id_code.get(user)
            d3_items = (self.train_ui[u_code] * self.similarity_matrix).indices
            allowed_items = list(set(allowed_items) & set(d3_items))
            u_recommended_items = random.sample(
                allowed_items, min(n_recommendations, len(allowed_items))
            )
            u_recommended_items = list(map(self.item_code_id.get, u_recommended_items))

        else:
            allowed_items = self.mapping_user_test_items[user]
            u_recommended_items = random.sample(
                allowed_items, min(n_recommendations, len(allowed_items))
            )

        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )
