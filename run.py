"""
    Main script for preparing dataset, tuning selected models,
    producing recommendations and evaluating.
"""
import argparse
import json
import logging
import os
from time import time

import numpy as np
from tqdm import tqdm

from src.common import config
from src.common.helpers import (
    df_from_dir,
    efficiency,
    get_interactions_subset,
    get_unix_path,
)
from src.data import splitting
from src.data.initializer import DataLoaderSaver
from src.evaluation.evaluator import Evaluator, preprocess_test
from src.models.model_factory import initialize_model
from src.tuning.bayessian import tune

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def parse_args(parser):
    """
        Parse input arguments into dict args
    Args:
        parser: Parser of the input arguments
    Returns:
        Dict of the arguments
    """
    parser.add_argument("--dataset", type=str, default="jobs_published")
    parser.add_argument("--target_users", type=str, default="all")
    parser.add_argument(
        "--models",
        type=json.loads,
        default=[
            "als",
            "prod2vec",
            "rp3beta",
            "slim",
            "ligthfm",
            "toppop",
            "random",
        ],
    )
    parser.add_argument(
        "--models_skip_tuning",
        type=json.loads,
        default=["toppop", "random", "perfect", "perfect_cf", "perfect_cf_d3"],
    )
    parser.add_argument(
        "--steps", type=json.loads, default=["prepare", "tune", "run", "evaluate"]
    )
    parser.add_argument("--n_recommendations", type=int, default=10)
    parser.add_argument("--validation_target_users_size", type=int, default=30000)
    parser.add_argument("--validation_fraction_users", type=float, default=0.2)
    parser.add_argument("--validation_fraction_items", type=float, default=0.2)

    args = vars(parser.parse_args())
    LOGGER.info("Parameters read")
    return args


def load_data_tune(path_train, path_validation, path_target_users):
    """
    Loads datasets used in tune step.
    """
    data = DataLoaderSaver()

    data.load_interactions(path_train)
    train = data.interactions

    data.load_interactions(path_validation)
    validation = data.interactions

    target_users = data.load_target_users(path_target_users)

    return train, validation, target_users


def load_data_run(path_train_and_validation, path_target_users):
    """
    Loads datasets used in run step.
    """

    data = DataLoaderSaver()

    data.load_interactions(path_train_and_validation)

    target_users = data.load_target_users(path_target_users)

    return data.interactions, target_users


def load_hyperparameters(path_tuning_dir, model_name):
    """
    Loads best hyperparameters found for a given model.
    """

    if os.listdir(path_tuning_dir):
        results = df_from_dir(path_tuning_dir)
        results = results[results["model_name"] == model_name]
        if len(results) > 0:
            return json.loads(
                results.sort_values(by="score", ascending=False).iloc[0][
                    "model_parameters"
                ]
            )

    return {}


def save_recommendations(
    model, recommendations, dataset_name, target_users_name, model_name
):
    """
    Saves recommendations.
    """
    recommendations_path = config.Paths(
        dataset_name=dataset_name,
        target_users_name=target_users_name,
        model_name=model_name,
    ).recommendations

    model.save_recommendations(recommendations, recommendations_path)


def step_prepare(
    path_interactions,
    path_train,
    path_validation,
    path_train_and_validation,
    path_test,
    path_target_path,
    validation_target_users_size,
    validation_fraction_users,
    validation_fraction_items,
    random_seed=10,
):
    """
    Function for executing "prepare" step of the script.
    """

    # load interactions
    data = DataLoaderSaver()
    data.load_interactions(path_interactions)

    # split into train_and_validation and test
    train_and_validation, test = splitting.split(data.interactions)
    train_and_validation.to_csv(
        path_train_and_validation, compression="gzip", index=None
    )
    test.to_csv(path_test, compression="gzip", index=None)

    # split into train and validation
    interactions_subset = get_interactions_subset(
        interactions=train_and_validation,
        fraction_users=validation_fraction_users,
        fraction_items=validation_fraction_items,
    )
    train, validation = splitting.split(interactions_subset)
    train.to_csv(path_train, compression="gzip", index=None)
    validation.to_csv(path_validation, compression="gzip", index=None)

    # prepare target_users
    test["user"].drop_duplicates().to_csv(
        path_target_path / "all.gzip", header=None, index=None, compression="gzip"
    )

    # prepare target_users for validation
    np.random.seed(random_seed)
    validation_users = validation["user"].drop_duplicates()
    validation_users.sample(
        n=min(validation_target_users_size, len(validation_users))
    ).to_csv(
        path_target_path / "subset_validation.gzip",
        header=None,
        index=None,
        compression="gzip",
    )


def step_tune(
    path_train,
    path_validation,
    path_target_users,
    path_tuning_dir,
    n_recommendations,
    model_names,
):
    """
    Function for executing "tune" step of the script.
    """

    train, validation, target_users = load_data_tune(
        path_train=path_train,
        path_validation=path_validation,
        path_target_users=path_target_users,
    )

    preprocessed_validation = preprocess_test(validation)

    pbar = tqdm(model_names)
    for model_name in pbar:
        pbar.set_description(model_name)
        tune(
            model_name=model_name,
            interactions=train,
            target_users=target_users,
            preprocessed_test=preprocessed_validation,
            n_recommendations=n_recommendations,
            output_dir=path_tuning_dir,
        )


def step_run(
    path_train_and_validation,
    path_test,
    path_target_users,
    path_tuning_dir,
    path_efficiency_dir,
    dataset_name,
    target_users_name,
    n_recommendations,
    model_names,
):
    """
    Function for executing "run" step of the script.
    Specified models are trained with best known configuration.
    Recommendations are produced and stored.
    """
    interactions, target_users = load_data_run(
        path_train_and_validation=path_train_and_validation,
        path_target_users=path_target_users,
    )

    model_bar = tqdm(model_names)
    for model_name in model_bar:
        model_bar.set_description(model_name)
        model_parameters = load_hyperparameters(
            path_tuning_dir=path_tuning_dir,
            model_name=model_name,
        )
        if model_name in ["perfect", "perfect_cf", "perfect_cf_d3"]:
            model_parameters.update({"path_test": path_test})

        model = initialize_model(model_name, **model_parameters)
        model.set_interactions(interactions)

        base_params = {
            "model_name": model_name,
            "model_parameters": model_parameters,
        }

        efficiency(path=get_unix_path(path_efficiency_dir), base_params=base_params)(
            model.preprocess
        )()

        efficiency(path=get_unix_path(path_efficiency_dir), base_params=base_params)(
            model.fit
        )()

        recommendations = efficiency(
            path=get_unix_path(path_efficiency_dir), base_params=base_params
        )(model.recommend)(
            target_users=target_users, n_recommendations=n_recommendations
        )
        save_recommendations(
            model,
            recommendations,
            dataset_name=dataset_name,
            target_users_name=target_users_name,
            model_name=model_name,
        )


def step_evaluate(
    path_recommendations_folder,
    path_test,
    path_result_evaluation_dir,
    n_recommendations,
    models_to_evaluate,
):
    """
    Evaluates all models based on saved recommendations, displays the result and saves.
    """
    evaluator = Evaluator(
        recommendations_path=path_recommendations_folder,
        test_path=path_test,
        k=n_recommendations,
        models_to_evaluate=models_to_evaluate,
    )
    evaluator.prepare()
    evaluator.evaluate_models()

    results_file_path = get_unix_path(path_result_evaluation_dir)
    evaluator.evaluation_results.to_csv(results_file_path)

    print("Evaluation results saved", evaluator.evaluation_results)


def steps_factory(step, args, paths):
    """
    Function for executing specific step of the script.
    """

    start = time()
    LOGGER.info(f"Executing step '{step}'")

    if step == "prepare":
        step_prepare(
            path_interactions=paths.interactions,
            path_train=paths.train,
            path_validation=paths.validation,
            path_train_and_validation=paths.train_and_validation,
            path_test=paths.test,
            path_target_path=paths.target_path,
            validation_target_users_size=args["validation_target_users_size"],
            validation_fraction_users=args["validation_fraction_users"],
            validation_fraction_items=args["validation_fraction_items"],
        )

    if step == "tune":
        step_tune(
            path_train=paths.train,
            path_validation=paths.validation,
            path_target_users=(paths.target_path / "subset_validation.gzip"),
            path_tuning_dir=paths.tuning_dir,
            n_recommendations=args["n_recommendations"],
            model_names=[
                model
                for model in args["models"]
                if model not in args["models_skip_tuning"]
            ],
        )

    if step == "run":
        step_run(
            path_train_and_validation=paths.train_and_validation,
            path_test=paths.test,
            path_target_users=paths.target_users,
            path_tuning_dir=paths.tuning_dir,
            path_efficiency_dir=paths.results_efficiency_dir,
            dataset_name=args["dataset"],
            target_users_name=args["target_users"],
            n_recommendations=args["n_recommendations"],
            model_names=args["models"],
        )

    if step == "evaluate":
        step_evaluate(
            path_recommendations_folder=paths.recommendations_folder,
            path_test=paths.test,
            path_result_evaluation_dir=paths.results_evaluation_dir,
            n_recommendations=args["n_recommendations"],
            models_to_evaluate=args["models"],
        )

    LOGGER.info(f"Step '{step}' executed in {time() - start:.2f} seconds")


def main():
    """Entry-point for preparing dataset, tuning selected models,
    producing recommendations and evaluating."""

    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    paths = config.Paths(
        dataset_name=args["dataset"],
        target_users_name=args["target_users"],
    )

    for step in args["steps"]:
        steps_factory(step, args, paths)


if __name__ == "__main__":
    main()
