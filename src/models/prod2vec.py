"""Module implementing a wrapper for the Prod2Vec model"""

import logging
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from ..data.initializer import DataLoaderSaver
from .base import BaseRecommender

WORKERS = cpu_count()


def _restrict_to_target_users(interactions, target_users):
    """
    Merge interactions with target users on user column
    :param interactions: Interactions dataset
    :param target_users: Target user dataset
    :return: Interactions dataset for target users
    """
    return interactions.merge(
        pd.DataFrame(target_users).rename(columns={0: "user"}), on="user"
    )


def _interactions_to_list_of_lists(interactions):
    """
    Transforms interactions dataframe user, item to user, [item1, item2] dataframe
    :param interactions: Interactions dataframe
    :return: Interactions dataframe format user, [item1, item2]
    """
    interactions = interactions.sort_values(by="timestamp")
    return interactions.groupby("user")["item"].apply(list)


class Prod2Vec(BaseRecommender, DataLoaderSaver):
    """
    Wrapper over Word2Vec model
    """

    def __init__(
        self,
        vector_size=48,
        alpha=0.1,
        window=5,
        min_count=2,
        sample=1e-3,
        workers=WORKERS,
        min_alpha=0.0001,
        sg=1,
        hs=0,
        negative=50,
        ns_exponent=0.75,
        cbow_mean=1,
        epochs=20,
        show_progress=True,
    ):
        """
        Source:
        https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py

        vector_size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs : {0, 1}, optional
            If 1, hierarchical softmax will be used for model training.
            If 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {0, 1}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        epochs : int, optional
            Number of iterations (epochs) over the corpus. (Formerly: `iter`)
        """

        super().__init__()

        # model
        self.model = Word2Vec(
            vector_size=vector_size,
            alpha=alpha,
            window=window,
            min_count=min_count,
            sample=sample,
            workers=workers,
            min_alpha=min_alpha,
            sg=sg,
            hs=hs,
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_mean=cbow_mean,
            epochs=epochs,
        )

        # data

        self.interactions = None
        self.train_sequences = None
        self.user_sequences = None

        self.show_progress = show_progress
        if not show_progress:
            logging.getLogger("gensim").setLevel(logging.WARNING)

    def _restrict_to_vocab(self, interactions):
        known_items = pd.DataFrame(
            self.model.wv.key_to_index.keys(), columns=["item"]
        ).astype(str)
        return interactions.merge(known_items, on="item")

    def _prepare_user_sequences(self, interactions, target_users):
        """
        It returns pd.Series with user as Index and a list of interacted items from model vocabulary as value.
        :param target_users: list of target users to be considered in the output, for None all users with interactions
        will be considered
        """

        restricted_interactions = _restrict_to_target_users(interactions, target_users)
        restricted_interactions = self._restrict_to_vocab(restricted_interactions)
        return _interactions_to_list_of_lists(restricted_interactions)

    def preprocess(self):
        """
        Prepare sequences for training the Word2Vec model
        """
        self.train_sequences = _interactions_to_list_of_lists(self.interactions)

    def fit(self):
        """
        Returns Word2VecKeyedVectors from trained i2vCF model
        """
        # Build vocabulary
        self.model.build_vocab(self.train_sequences)

        # Train
        self.model.train(
            self.train_sequences,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )

        # Precompute L2-normalized vectors
        self.model.wv.init_sims(replace=True)

    def recommend(
        self,
        target_users,
        n_recommendations,
        filter_out_interacted_items=True,
        show_progress=True,
    ):
        """
            Recommends n_recommendations items for target_users
        :return:
            pd.DataFrame (user, item_1, item_2, ..., item_n)
        """

        self.user_sequences = self._prepare_user_sequences(
            self.interactions, target_users
        )

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

        u_recommended_items = []
        if self.user_sequences.get(user) is not None:
            u_items = self.user_sequences.get(user)
            u_recommended_items = list(
                list(
                    zip(
                        *self.model.wv.most_similar(
                            u_items,
                            topn=n_recommendations
                            + len(u_items) * filter_out_interacted_items,
                        )
                    )
                )[0]
            )
            if filter_out_interacted_items:
                u_recommended_items = [
                    i for i in u_recommended_items if i not in u_items
                ][:n_recommendations]
        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )
