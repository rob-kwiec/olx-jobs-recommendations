"""
Module with defined configuration for optimization
"""
from skopt.space import Real, Integer, Categorical

MODEL_SEARCH_SPACES = {
    "rp3beta": [Real(0, 2, name="alpha"), Real(0, 2, name="beta")],
    "prod2vec": [
        Integer(32, 512, name="vector_size"),
        Real(0.01, 0.5, name="alpha"),
        Integer(1, 20, name="window"),
        Integer(1, 20, name="min_count"),
        Real(0, 0.01, name="sample"),
        Real(0, 0.01, name="min_alpha"),
        Categorical([0, 1], name="sg"),
        Categorical([0, 1], name="hs"),
        Integer(1, 200, name="negative"),
        Real(-0.5, 1.5, name="ns_exponent"),
        Categorical([0, 1], name="cbow_mean"),
        Integer(5, 40, name="epochs"),
    ],
    "slim": [
        Real(0, 0.1, name="alpha"),
        Real(0, 1, name="l1_ratio"),
        Integer(1, 3, name="iterations"),
    ],
    "als": [
        Integer(128, 512, name="factors"),
        Real(0.001, 0.1, name="regularization"),
        Integer(5, 20, name="iterations"),
        Integer(10, 500, name="event_weights_multiplier"),
    ],
    "lightfm": [
        Integer(8, 512, name="no_components"),
        Integer(1, 10, name="k"),
        Integer(1, 20, name="n"),
        Categorical(["adagrad", "adadelta"], name="learning_schedule"),
        Categorical(["logistic", "bpr", "warp", "warp-kos"], name="loss"),
        Integer(1, 100, name="max_sampled"),
        Integer(1, 20, name="epochs"),
    ],
}


MODEL_TUNING_CONFIGURATION = {
    "rp3beta": {"n_calls": 100},
    "prod2vec": {"n_calls": 100},
    "slim": {"n_calls": 100, "x0": [0, 0, 1]},
    "als": {"n_calls": 100},
    "lightfm": {"n_calls": 100},
}
