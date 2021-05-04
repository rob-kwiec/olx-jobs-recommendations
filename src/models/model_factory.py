"""Module providing a model factory getter"""

from .als import ALS
from .prod2vec import Prod2Vec
from .rp3beta import RP3Beta
from .slim import SLIM
from .toppop import TopPop
from .random import Random


def initialize_model(model_type: str, **kwargs):
    """
    :param model_type: als
    :return: model instance
    """

    if model_type == "als":
        return ALS(**kwargs)
    if model_type == "prod2vec":
        return Prod2Vec(**kwargs)
    if model_type == "rp3beta":
        return RP3Beta(**kwargs)
    if model_type == "slim":
        return SLIM(**kwargs)
    if model_type == "toppop":
        return TopPop(**kwargs)
    if model_type == "random":
        return Random(**kwargs)
    raise ValueError(model_type)
