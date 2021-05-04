"""
Module with global helper functions
"""
import os
import tracemalloc
from datetime import datetime
from time import time

import numpy as np
import pandas as pd


def dict_to_df(dictionary):
    """
    Creates pandas dataframe from dictionary
    :param dictionary: Original dictionary
    :return: Dataframe from original dictionary
    """
    return pd.DataFrame({k: [v] for k, v in dictionary.items()})


def efficiency(path, base_params=None):
    """
    Parametrized decorator for executing function with efficiency logging and
    storing the results under the given path
    """
    base_params = base_params or {}

    def efficiency_decorator(func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time()
            result = func(*args, **kwargs)
            execution_time = time() - start_time
            _, peak = tracemalloc.get_traced_memory()
            dict_to_df(
                {
                    **base_params,
                    **{
                        "function_name": func.__name__,
                        "execution_time": execution_time,
                        "memory_peak": peak,
                    },
                }
            ).to_csv(path, index=False)
            tracemalloc.stop()
            return result

        return wrapper

    return efficiency_decorator


def get_unix_path(path):
    """
    Returns the input path with unique csv filename
    """
    return path / f"{datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')}.csv"


def df_from_dir(dir_path):
    """
    Returns pd.DataFrame with concatenated files from the given path
    """
    files_read = [
        pd.read_csv(dir_path / filename)
        for filename in os.listdir(dir_path)
        if filename.endswith(".csv")
    ]
    return pd.concat(files_read, axis=0, ignore_index=True)


def get_interactions_subset(
    interactions, fraction_users, fraction_items, random_seed=10
):
    """
    Select subset from interactions based on fraction of users and items
    :param interactions: Original interactions
    :param fraction_users: Fraction of users
    :param fraction_items: Fraction of items
    :param random_seed: Random seed
    :return: Dataframe with subset of interactions
    """

    def _get_subset_by_column(column, fraction):
        column_df = interactions[column].unique()
        subset = set(np.random.choice(column_df, int(len(column_df) * fraction)))
        return interactions[interactions[column].isin(subset)]

    np.random.seed(random_seed)
    if fraction_users < 1:
        interactions = _get_subset_by_column("user", fraction_users)

    if fraction_items < 1:
        interactions = _get_subset_by_column("item", fraction_items)

    return interactions
