"""
Module with functions for metrics calculation
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def ranking_metrics(test_matrix, recommendations, k=10):
    """
    Calculates ranking metrics (precision, recall, F1, F0.5, NDCG, mAP, MRR, LAUC, HR)
    based on test interactions matrix and recommendations
    :param test_matrix: Test interactions matrix
    :param recommendations: Recommendations
    :param k: Number of top recommendations to calculate metrics on
    :return: Dataframe with metrics
    """

    items_number = test_matrix.shape[1]
    result = []

    for (user_count, user) in tqdm(enumerate(recommendations[:, 0])):
        u_interacted_items = get_interacted_items(test_matrix, user)
        interacted_items_amount = len(u_interacted_items)

        if interacted_items_amount > 0:  # skip users with no items in test set

            # evaluation
            success_statistics = calculate_successes(
                k, recommendations, u_interacted_items, user_count
            )

            result.append(
                [user]
                + calculate_ranking_metrics(
                    success_statistics,
                    interacted_items_amount,
                    items_number,
                    k,
                )
            )

    return pd.DataFrame(
        result,
        columns=[
            "user",
            "precision",
            "recall",
            "ndcg",
            "map",
            "mrr",
            "lauc",
            "hr",
        ],
    )


def calculate_ranking_metrics(
    success_statistics,
    interacted_items_amount,
    items_number,
    k,
):
    """
    Calculates ranking metrics based on success statistics
    :param success_statistics: Success statistics dictionary
    :param interacted_items_amount:
    :param items_number:
    :param k: Number of top recommendations to calculate metrics on
    :return: Dictionary with metrics
    """
    precision = success_statistics["total_amount"] / k
    recall = success_statistics["total_amount"] / interacted_items_amount
    user_metrics = [
        precision,
        recall,
        calculate_ndcg(interacted_items_amount, k, success_statistics["total"]),
        calculate_map(success_statistics, interacted_items_amount, k),
        calculate_mrr(success_statistics["total"]),
        calculate_lauc(success_statistics, interacted_items_amount, items_number, k),
        int(success_statistics["total_amount"] > 0),
    ]
    return user_metrics


def calculate_mrr(user_successes):
    return (
        1 / (user_successes.nonzero()[0][0] + 1)
        if user_successes.nonzero()[0].size > 0
        else 0
    )


def calculate_f(precision, recall, f):
    return (
        (f ** 2 + 1) * (precision * recall) / (f ** 2 * precision + recall)
        if precision + recall > 0
        else 0
    )


def calculate_lauc(successes, interacted_items_amount, items_number, k):
    return (
        np.dot(successes["cumsum"], 1 - successes["total"])
        + (successes["total_amount"] + interacted_items_amount)
        / 2
        * ((items_number - interacted_items_amount) - (k - successes["total_amount"]))
    ) / ((items_number - interacted_items_amount) * interacted_items_amount)


def calculate_map(successes, interacted_items_amount, k):
    return np.dot(successes["cumsum"] / np.arange(1, k + 1), successes["total"]) / min(
        k, interacted_items_amount
    )


def calculate_ndcg(interacted_items_amount, k, user_successes):
    cumulative_gain = 1.0 / np.log2(np.arange(2, k + 2))
    cg_sum = np.cumsum(cumulative_gain)
    return (
        np.dot(user_successes, cumulative_gain)
        / cg_sum[min(k, interacted_items_amount) - 1]
    )


def calculate_successes(k, recommendations, u_interacted_items, user_count):

    items = recommendations[user_count, 1 : k + 1]
    user_successes = np.isin(items, u_interacted_items)

    return dict(
        total=user_successes.astype(int),
        total_amount=user_successes.sum(),
        cumsum=np.cumsum(user_successes),
    )


def get_reactions(test_matrix, user):
    return test_matrix.data[test_matrix.indptr[user] : test_matrix.indptr[user + 1]]


def get_interacted_items(test_matrix, user):
    return test_matrix.indices[test_matrix.indptr[user] : test_matrix.indptr[user + 1]]
