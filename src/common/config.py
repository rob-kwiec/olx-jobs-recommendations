"""
Module with Paths class
"""

import os
from pathlib import Path


class Paths:
    """
    Simple class for returning paths needed to load, save and evaluate data for recommendation models.
    """

    def __init__(
        self,
        dataset_name,
        target_users_name,
        model_name=None,
        repository_path=None,
    ):

        self.repository_path = (
            Path(*Path(os.path.realpath(__file__)).parts[:-3])
            if repository_path is None
            else Path(repository_path)
        )
        self.datasets_folder = self.repository_path / "data" / "raw"
        self.dataset_folder = self.datasets_folder / dataset_name
        self.target_path = self.dataset_folder / "target_users"
        self.recommendations_folder = (
            self.repository_path
            / "data"
            / "recommendations"
            / dataset_name
            / target_users_name
        )

        self.results_evaluation_dir = (
            self.repository_path
            / "data"
            / "results"
            / dataset_name
            / target_users_name
            / "evaluation"
        )

        self.results_efficiency_dir = (
            self.repository_path
            / "data"
            / "results"
            / dataset_name
            / target_users_name
            / "efficiency"
        )

        self.tuning_dir = self.repository_path / "data" / "tuning" / dataset_name

        for path in [
            self.repository_path,
            self.datasets_folder,
            self.dataset_folder,
            self.target_path,
            self.recommendations_folder,
            self.results_evaluation_dir,
            self.results_efficiency_dir,
            self.tuning_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.interactions = self.dataset_folder / "interactions.gzip"
        self.train = self.dataset_folder / "train.gzip"
        self.train_and_validation = self.dataset_folder / "train_and_validation.gzip"
        self.validation = self.dataset_folder / "validation.gzip"
        self.test = self.dataset_folder / "test.gzip"

        self.ads = self.datasets_folder / "ads.json.gz"
        self.profiles = self.datasets_folder / "candidate_profiles.json.gz"
        self.target_users = self.target_path / (target_users_name + ".gzip")

        if model_name:
            self.recommendations = self.recommendations_folder / (model_name + ".gzip")
