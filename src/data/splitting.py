"""
Module for splitting data into train/validation/test sets
"""


import numpy as np
import pandas as pd


def splitting_functions_factory(function_name):
    """Returns splitting function based on name"""
    if function_name == "by_time":
        return split_by_time


def split_by_time(interactions, fraction_test, random_state=30):
    """
    Splits interactions by time. Returns tuple of dataframes: train and test.
    """

    np.random.seed(random_state)

    test_min_timestamp = np.percentile(
        interactions["timestamp"], 100 * (1 - fraction_test)
    )

    train = interactions[interactions["timestamp"] < test_min_timestamp]
    test = interactions[interactions["timestamp"] >= test_min_timestamp]

    return train, test


def filtering_restrict_to_train_users(train, test):
    """
    Returns test DataFrame restricted to users from train set.
    """
    train_users = set(train["user"])
    return test[test["user"].isin(train_users)]


def filtering_already_interacted_items(train, test):
    """
    Filters out (user, item) pairs from the test set if the given user interacted with a given item in train set.
    """
    columns = test.columns
    already_interacted_items = train[["user", "item"]].drop_duplicates()
    merged = pd.merge(
        test, already_interacted_items, on=["user", "item"], how="left", indicator=True
    )
    test = merged[merged["_merge"] == "left_only"]
    return test[columns]


def filtering_restrict_to_unique_user_item_pair(dataframe):
    """
    Returns pd.DataFrame where each (user, item) pair appears only once.
    A list of corresponding events is stores instead of a single event.
    Returned timestamp is the timestamp of the first (user, item) interaction.
    """
    return (
        dataframe.groupby(["user", "item"])
        .agg({"event": list, "timestamp": "min"})
        .reset_index()
    )


def split(
    interactions,
    splitting_config=None,
    restrict_to_train_users=True,
    filter_out_already_interacted_items=True,
    restrict_train_to_unique_user_item_pairs=True,
    restrict_test_to_unique_user_item_pairs=True,
    replace_events_by_ones=True,
):
    """
    Main function used for splitting the dataset into the train and test sets.
    Parameters
    ----------
    interactions: pd.DataFrame
        Interactions dataframe
    splitting_config : dict, optional
        Dict with name and parameters passed to splitting function.
        Currently only name="by_time" supported.
    restrict_to_train_users : boolean, optional
        Whether to restrict users in the test set only to users from the train set.
    filter_out_already_interacted_items : boolean, optional
        Whether to filter out (user, item) pairs from the test set if the given user interacted with a given item
        in the train set.
    restrict_test_to_unique_user_item_pairs
        Whether to return only one row per (user, item) pair in test set.
    """

    if splitting_config is None:
        splitting_config = {
            "name": "by_time",
            "fraction_test": 0.2,
        }

    splitting_name = splitting_config["name"]
    splitting_config = {k: v for k, v in splitting_config.items() if k != "name"}

    train, test = splitting_functions_factory(splitting_name)(
        interactions=interactions, **splitting_config
    )

    if restrict_to_train_users:
        test = filtering_restrict_to_train_users(train, test)

    if filter_out_already_interacted_items:
        test = filtering_already_interacted_items(train, test)

    if restrict_train_to_unique_user_item_pairs:
        train = filtering_restrict_to_unique_user_item_pair(train)

    if restrict_test_to_unique_user_item_pairs:
        test = filtering_restrict_to_unique_user_item_pair(test)

    if replace_events_by_ones:
        train["event"] = 1
        test["event"] = 1

    return train, test
