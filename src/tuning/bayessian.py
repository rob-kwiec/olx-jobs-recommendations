"""Module implementing Bayessian optimization"""

import json

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
from tqdm import tqdm

from src.common.helpers import get_unix_path
from src.models.model_factory import initialize_model
from .config import MODEL_SEARCH_SPACES, MODEL_TUNING_CONFIGURATION


def evaluate(
    recommendations_df,
    preprocessed_test,
    top_k=10,
):
    """
    Calculates precision score based on provided recommendations dataframe
    and test interactions dataset
    :param recommendations_df: recommendations dataframe
    :param preprocessed_test: tuple with user_id_code, item_id_code, test_ui
    :param top_k: top k interactions to evaluate
    :return: precision score
    """
    recommendations_df = recommendations_df.iloc[:, : top_k + 1].copy()
    user_id_code, item_id_code, test_ui = preprocessed_test

    users = recommendations_df.iloc[:, :1].apply(
        np.vectorize(lambda x: user_id_code.setdefault(str(x), -1))
    )
    items = recommendations_df.iloc[:, 1:].apply(
        np.vectorize(
            lambda x: -1
            if pd.isnull(x) or x == "None"
            else item_id_code.setdefault(str(x), -1)
        )
    )
    recommendations_array = np.array(pd.concat([users, items], axis=1))

    return precision_total(test_ui, recommendations_array, top_k=top_k)


def precision_total(test_ui, recommendations, top_k=10):
    """
    Calculates precision score based on provided preprocessed recommendations numpy array
    and test interactions matrix
    :param test_ui:
    :param recommendations:
    :param top_k:
    :return:
    """
    relevant_users = 0
    precision_sum = 0

    for (nb_user, user) in tqdm(enumerate(recommendations[:, 0])):
        u_rated_items = set(
            test_ui.indices[test_ui.indptr[user] : test_ui.indptr[user + 1]]
        )

        if len(u_rated_items) > 0:  # skip users with no items in test set

            nb_user_successes = sum(
                [
                    1
                    for item in recommendations[nb_user, 1 : top_k + 1]
                    if item in u_rated_items
                ]
            )

            relevant_users += 1
            precision_sum += nb_user_successes / top_k

    return precision_sum / relevant_users


def save_evaluation_results(model_name, score, model_parameters, output_dir):
    """
    Saves model's score to the csv file
    :param model_name: Model name
    :param score: Score
    :param model_parameters: Model parameters
    :param output_dir: Output path
    :return:
    """

    def _np_encoder(object_to_encode):
        if isinstance(object_to_encode, np.generic):
            return object_to_encode.item()
        raise ValueError("Provided object is not np.generic")

    output = pd.DataFrame(
        [[model_name, score, json.dumps(model_parameters, default=_np_encoder)]],
        columns=["model_name", "score", "model_parameters"],
    )
    output.to_csv(
        get_unix_path(output_dir),
        index=False,
    )
    print(output)


def tune(
    model_name,
    interactions,
    target_users,
    preprocessed_test,
    n_recommendations,
    output_dir,
):
    """
    Runs optimization for model
    :param model_name: Model to optimize
    :param interactions: Interactions history
    :param target_users: Target users for making recommendations
    :param preprocessed_test: tuple with user_id_code, item_id_code, test_ui
    :param n_recommendations: Number of recommendations to predict
    :param output_dir: Output path for storing iteration results
    :return:
    """

    # to be replaced by config per model
    space = MODEL_SEARCH_SPACES[model_name]
    tuning_configuration = MODEL_TUNING_CONFIGURATION[model_name]

    @use_named_args(space)
    def _objective(**model_parameters):
        try:
            model = initialize_model(model_name, **model_parameters)
            model.set_interactions(interactions)
            model.preprocess()
            model.fit()
            recommendations = model.recommend(
                target_users=target_users, n_recommendations=n_recommendations
            )
            score = evaluate(
                recommendations, preprocessed_test, top_k=n_recommendations
            )
        except ValueError:
            score = 0

        save_evaluation_results(model_name, score, model_parameters, output_dir)

        return -score

    gp_minimize(_objective, space, random_state=0, **tuning_configuration)
