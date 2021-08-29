"""Module implementing a wrapper for the ALS model"""

import multiprocessing
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from lightfm import LightFM

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender


class LFM(BaseRecommender, DataLoaderSaver):
    """
    Wrapper over LightFM model
    """

    def __init__(
        self,
        no_components=30,
        k=5,
        n=10,
        learning_schedule="adagrad",
        loss="logistic",
        learning_rate=0.05,
        rho=0.95,
        epsilon=1e-06,
        item_alpha=0.0,
        user_alpha=0.0,
        max_sampled=10,
        random_state=42,
        epochs=20,
        show_progress=True,
    ):
        """
        Source of descriptions:
        https://making.lyst.com/lightfm/docs/_modules/lightfm/lightfm.html#LightFM

        A hybrid latent representation recommender model.

        The model learns embeddings (latent representations in a high-dimensional
        space) for users and items in a way that encodes user preferences over items.
        When multiplied together, these representations produce scores for every item
        for a given user; items scored highly are more likely to be interesting to
        the user.

        The user and item representations are expressed in terms of representations
        of their features: an embedding is estimated for every feature, and these
        features are then summed together to arrive at representations for users and
        items. For example, if the movie 'Wizard of Oz' is described by the following
        features: 'musical fantasy', 'Judy Garland', and 'Wizard of Oz', then its
        embedding will be given by taking the features' embeddings and adding them
        together. The same applies to user features.

        The embeddings are learned through `stochastic gradient
        descent <http://cs231n.github.io/optimization-1/>`_ methods.

        Four loss functions are available:

        - logistic: useful when both positive (1) and negative (-1) interactions
        are present.
        - BPR: Bayesian Personalised Ranking [1]_ pairwise loss. Maximises the
        prediction difference between a positive example and a randomly
        chosen negative example. Useful when only positive interactions
        are present and optimising ROC AUC is desired.
        - WARP: Weighted Approximate-Rank Pairwise [2]_ loss. Maximises
        the rank of positive examples by repeatedly sampling negative
        examples until rank violating one is found. Useful when only
        positive interactions are present and optimising the top of
        the recommendation list (precision@k) is desired.
        - k-OS WARP: k-th order statistic loss [3]_. A modification of WARP that
        uses the k-th positive example for any given user as a basis for pairwise
        updates.

        Two learning rate schedules are available:

        - adagrad: [4]_
        - adadelta: [5]_

        Parameters
        ----------

        no_components: int, optional
            the dimensionality of the feature latent embeddings.
        k: int, optional
            for k-OS training, the k-th positive example will be selected from the
            n positive examples sampled for every user.
        n: int, optional
            for k-OS training, maximum number of positives sampled for each update.
        learning_schedule: string, optional
            one of ('adagrad', 'adadelta').
        loss: string, optional
            one of  ('logistic', 'bpr', 'warp', 'warp-kos'): the loss function.
        learning_rate: float, optional
            initial learning rate for the adagrad learning schedule.
        rho: float, optional
            moving average coefficient for the adadelta learning schedule.
        epsilon: float, optional
            conditioning parameter for the adadelta learning schedule.
        item_alpha: float, optional
            L2 penalty on item features. Tip: setting this number too high can slow
            down training. One good way to check is if the final weights in the
            embeddings turned out to be mostly zero. The same idea applies to
            the user_alpha parameter.
        user_alpha: float, optional
            L2 penalty on user features.
        max_sampled: int, optional
            maximum number of negative samples used during WARP fitting.
            It requires a lot of sampling to find negative triplets for users that
            are already well represented by the model; this can lead to very long
            training times and overfitting. Setting this to a higher number will
            generally lead to longer training times, but may in some cases improve
            accuracy.
        random_state: int seed, RandomState instance, or None
            The seed of the pseudo random number generator to use when shuffling
            the data and initializing the parameters.

        epochs: (int, optional) number of epochs to run
        """

        super().__init__()

        self.model = LightFM(
            no_components=no_components,
            k=k,
            n=n,
            learning_schedule=learning_schedule,
            loss=loss,
            learning_rate=learning_rate,
            rho=rho,
            epsilon=epsilon,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
            max_sampled=max_sampled,
            random_state=random_state,
        )
        self.epochs = epochs

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
        data["event_value"] = 1

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
        self.model.fit(
            self.train_ui,
            epochs=self.epochs,
            num_threads=multiprocessing.cpu_count(),
            verbose=self.show_progress,
        )

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
        items_to_recommend = np.arange(len(self.item_code_id))

        with ThreadPool() as thread_pool:
            recommendations = list(
                tqdm(
                    thread_pool.imap(
                        partial(
                            self.recommend_per_user,
                            n_recommendations=n_recommendations,
                            items_to_recommend=items_to_recommend,
                        ),
                        target_users,
                    ),
                    disable=not self.show_progress,
                )
            )

        return pd.DataFrame(recommendations)

    def recommend_per_user(self, user, n_recommendations, items_to_recommend):
        """
        Recommends n items per user
        :param user: User id
        :param n_recommendations: Number of recommendations
        :return: list of format [user_id, item1, item2 ...]
        """
        u_code = self.user_id_code.get(user)

        if u_code is not None:
            interacted_items = self.train_ui.indices[
                self.train_ui.indptr[u_code] : self.train_ui.indptr[u_code + 1]
            ]

            scores = self.model.predict(int(u_code), items_to_recommend)

            item_recommendations = items_to_recommend[np.argsort(-scores)][
                : n_recommendations + len(interacted_items)
            ]
            item_recommendations = [
                self.item_code_id[item]
                for item in item_recommendations
                if item not in interacted_items
            ]

        return (
            [user]
            + item_recommendations
            + [None] * (n_recommendations - len(item_recommendations))
        )
