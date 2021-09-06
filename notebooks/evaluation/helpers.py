import os
import pandas as pd


def overlap(df1, df2):
    """
    Returns the Overlap Coefficient with respect to (user, item) pairs.
    We assume uniqueness of (user, item) pairs in DataFrames
    (not recommending the same item to the same users multiple times)).
    :param df1: DataFrame which index is user_id and column ["items"] is a list of recommended items
    :param df2: DataFrame which index is user_id and column ["items"] is a list of recommended items
    """
    nb_items = min(df1["items"].apply(len).sum(), df2["items"].apply(len).sum())

    merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
    nb_common_items = merged_df.apply(
        lambda x: len(set(x["items_x"]) & set(x["items_y"])), axis=1
    ).sum()

    return 1.00 * nb_common_items / nb_items


def get_recommendations(models_to_evaluate, recommendations_path):
    """
    Returns dictionary with model_names as keys and recommendations as values.
    :param models_to_evaluate: List of model names
    :param recommendations_path: Stored recommendations directory
    """
    models = [
        (file_name.split(".")[0], file_name)
        for file_name in os.listdir(recommendations_path)
    ]

    return {
        model[0]: pd.read_csv(
            os.path.join(recommendations_path, model[1]),
            header=None,
            compression="gzip",
            dtype=str,
        )
        for model in models
        if model[0] in models_to_evaluate
    }


def _get_models(models_to_evaluate, recommendations_path):
    models = [
        (file_name.split(".")[0], file_name)
        for file_name in os.listdir(recommendations_path)
    ]
    if models_to_evaluate:
        return [model for model in models if model[0] in models_to_evaluate]
    return models
