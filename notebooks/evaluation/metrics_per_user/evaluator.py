"""
Module with Evaluator class used for models evaluation
"""


import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from metrics import ranking_metrics


def preprocess_test(test: pd.DataFrame):
    """
    Preprocesses test set to speed up evaluation
    """

    def _map_column(test, column):
        test[f"{column}_code"] = test[column].astype("category").cat.codes
        return dict(zip(test[column], test[f"{column}_code"]))

    test = test.copy()

    test.columns = ["user", "item", "event", "timestamp"]
    user_map = _map_column(test, "user")
    item_map = _map_column(test, "item")

    test_matrix = sparse.csr_matrix(
        (np.ones(len(test)), (test["user_code"], test["item_code"]))
    )
    return user_map, item_map, test_matrix


class Evaluator:
    """
    Class used for models evaluation
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        recommendations_path: Path,
        test_path: Path,
        k,
        models_to_evaluate,
        output_path,
    ):
        self.recommendations_path = recommendations_path
        self.test_path = test_path
        self.k = k
        self.models_to_evaluate = models_to_evaluate
        self.output_path = output_path
        self.located_models = None
        self.test = None
        self.user_map = None
        self.item_map = None
        self.test_matrix = None

        self.evaluation_results = {}

    def prepare(self):
        """
        Prepares test set and models to evaluate
        """

        def _get_models(models_to_evaluate, recommendations_path):
            models = [
                (file_name.split(".")[0], file_name)
                for file_name in os.listdir(recommendations_path)
            ]
            if models_to_evaluate:
                return [model for model in models if model[0] in models_to_evaluate]
            return models

        self.test = pd.read_csv(self.test_path, compression="gzip").astype(
            {"user": str, "item": str}
        )
        self.user_map, self.item_map, self.test_matrix = preprocess_test(self.test)

        self.located_models = _get_models(
            self.models_to_evaluate, self.recommendations_path
        )

    def evaluate_models(self):
        """
        Evaluating multiple models
        """

        def _read_recommendations(file_name):
            return pd.read_csv(
                os.path.join(self.recommendations_path, file_name),
                header=None,
                compression="gzip",
                dtype=str,
            )

        evaluation_results = pd.DataFrame()

        for model, file_name in self.located_models:
            recommendations = _read_recommendations(file_name)
            evaluation_result = self.evaluate(
                original_recommendations=recommendations,
            )
            evaluation_result.insert(0, "model_name", model)

            evaluation_results = pd.concat([evaluation_results, evaluation_result])

        user_code_id = {code: id for id, code in self.user_map.items()}
        evaluation_results["user"] = evaluation_results["user"].apply(user_code_id.get)
        evaluation_results.to_csv(
            self.output_path / "results.gzip", compression="gzip", index=None
        )

    def evaluate(
        self,
        original_recommendations: pd.DataFrame,
    ):
        """
        Evaluate single model
        """

        def _format_recommendations(recommendations, user_id_code, item_id_code):
            users = recommendations.iloc[:, :1].applymap(
                lambda x: user_id_code.setdefault(str(x), -1)
            )
            items = recommendations.iloc[:, 1:].applymap(
                lambda x: -1 if pd.isna(x) else item_id_code.setdefault(x, -1)
            )
            return np.array(pd.concat([users, items], axis=1))

        original_recommendations = original_recommendations.iloc[:, : self.k + 1].copy()

        formatted_recommendations = _format_recommendations(
            original_recommendations, self.user_map, self.item_map
        )

        return ranking_metrics(
            self.test_matrix,
            formatted_recommendations,
            k=self.k,
        )
