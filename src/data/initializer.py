"""
Definition of base reader class
"""

import os
from pathlib import Path

import pandas as pd


class DataLoaderSaver:
    """Dataset initialize interface"""

    def __init__(self):
        self.interactions = None

    def load_interactions(self, path):
        """Loads interactions dataset"""

        def _load_interactions(path, compression=None):
            self.interactions = pd.read_csv(
                path,
                compression=compression,
                header=0,
                names=["user", "item", "event", "timestamp"],
            ).astype({"user": str, "item": str, "event": str, "timestamp": int})

        # to support loading datasets provided in csv format
        path_csv = Path(os.path.splitext(path)[0] + ".csv")

        if os.path.exists(path):
            _load_interactions(path, compression="gzip")
        elif os.path.exists(path_csv):
            _load_interactions(path_csv)
        else:
            raise Exception("Interactions file does not exist! Please provide it.")

    def set_interactions(self, interactions):
        """Sets interactions based on input"""
        self.interactions = interactions

    @staticmethod
    def load_target_users(path):
        """Loads target users from file and returns a list"""
        return list(
            pd.read_csv(path, compression="gzip", header=None).astype(str).iloc[:, 0]
        )

    @staticmethod
    def save_recommendations(recommendations, path):
        """Saved recommendations"""
        recommendations.to_csv(path, index=False, header=False, compression="gzip")
