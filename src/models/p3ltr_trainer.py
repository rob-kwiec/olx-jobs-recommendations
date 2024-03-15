"""Module implementing a P3LTR Trainer"""

import dgl
import dgl.function as fn
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import random
from .base import BaseRecommender
from .preprocessing import FeaturePreprocessor
from ..data.initializer import DataLoaderSaver


class FeatureEncoder(torch.nn.Module):
    """
    A model for producing edge scores based on edge features.
    """

    def __init__(self, event_size, init_degree_param):
        super(FeatureEncoder, self).__init__()

        self.param_destination_degree = nn.Parameter(
            torch.tensor([[init_degree_param]], requires_grad=True)
        )
        self.recency = nn.Parameter(torch.tensor([[0.0]], requires_grad=True))

        self.linear_events = torch.nn.Linear(event_size, 1)
        torch.nn.init.constant_(self.linear_events.weight, 0)
        torch.nn.init.constant_(self.linear_events.bias, 0)

        self.linear_events_count = torch.nn.Linear(1, 1)
        torch.nn.init.constant_(self.linear_events_count.weight, 0)
        torch.nn.init.constant_(self.linear_events_count.bias, 0)

    def forward(self, destination_degree, recency, events):
        events_count = torch.sum(events, axis=1, keepdim=True)
        score = (
            torch.pow(destination_degree, -self.param_destination_degree)
            * torch.exp(-self.recency * recency)
            * torch.sigmoid(self.linear_events(events / events_count))
            * torch.sigmoid(self.linear_events_count(events_count))
        )
        return score


def loss_function_factory(loss):
    """
    A factory of loss functions possible to be used in P3LTR model
    """
    if loss == "ratio":
        return loss_ratio
    if loss == "log_ratio":
        return loss_log_ratio
    if loss == "log_ratio_k":
        return loss_log_ratio_k
    if loss == "log_ratio_boosted":
        return loss_log_ratio_boosted
    raise ValueError(loss)


def loss_ratio(validation_node_score, scores, parameters, regularization, **kwargs):
    dif = torch.mean(torch.flatten(scores[0])) / validation_node_score
    return dif + regularization * torch.sum(parameters**2)


def loss_log_ratio(validation_node_score, scores, parameters, regularization, **kwargs):
    dif = torch.log(torch.mean(torch.flatten(scores[0])) / validation_node_score)
    return dif + regularization * torch.sum(parameters**2)


def loss_log_ratio_k(
    validation_node_score, scores, parameters, regularization, **kwargs
):
    # print(torch.mean(torch.flatten(scores[0])[k:k+1]))
    # print('--------')
    s = torch.flatten(scores[0])
    scores_pos = s[s > 0]
    cutoff = len(scores_pos) - 1
    dif = torch.log(scores_pos[cutoff] / validation_node_score)
    return dif + regularization * torch.sum(parameters**2)


def loss_log_ratio_boosted(
    validation_node_score,
    validation_node_position,
    scores,
    parameters,
    regularization,
    **kwargs,
):
    dif = torch.log(torch.mean(torch.flatten(scores[0])) / validation_node_score)
    return dif * torch.log(validation_node_position) + regularization * torch.sum(
        parameters**2
    )


def get_successors(g, node_ids):
    """
    Get destination nodes for which there exist a directed edge starting from one of specified node_ids
    """
    return torch.unique(dgl.out_subgraph(g, node_ids).edges()[1])


def recommend_node(g, node_id, top_k, validation_node_id):
    """
    Recommends top_k recommended nodes for a given node_id.
    Connection with validation_node_id is not used in recommendation process.
    The gaol of training the model is to score high the validation node.
    """
    with g.local_scope():
        g.ndata["weight"] = torch.zeros(g.ndata["weight"].shape[0], 1)

        g.ndata["weight"][
            node_id
        ] = 1  # it overrides input graph - no impact on results

        # 1st hop
        g.push(
            [node_id], fn.src_mul_edge("weight", "weight_0", "m"), fn.sum("m", "weight")
        )
        g.ndata["weight"][node_id] = 0
        g.ndata["weight"][validation_node_id] = 0

        # 2nd hop
        successors_1 = g.successors(node_id)
        g.push(
            successors_1,
            fn.src_mul_edge("weight", "weight_1", "m"),
            fn.sum("m", "weight"),
        )
        g.ndata["weight"][successors_1] = 0
        g.ndata["weight"][
            node_id
        ] = 0  # to ignore paths which go back to the source node

        # 3rd hop
        successors_2 = get_successors(g, successors_1)
        g.push(
            successors_2,
            fn.src_mul_edge("weight", "weight_2", "m"),
            fn.sum("m", "weight"),
        )
        g.ndata["weight"][successors_2] = 0

        # do not recommend neighbours, except validation node
        g.ndata["weight"][successors_1[successors_1 != validation_node_id]] = 0

        validation_node_score = g.ndata["weight"][validation_node_id]
        validation_node_position = torch.sum(g.ndata["weight"] >= validation_node_score)

        return (
            dgl.topk_nodes(g, "weight", top_k),
            validation_node_score,
            validation_node_position,
        )


def get_validation_user_item(data, min_items=2):
    """
    We restrict to users with min_items (at least 2).
    The item which the user interacted the most recent is taken as a validation node.
    """
    items_per_user = data.groupby(["user_code"])["item_code"].nunique()
    users = set(items_per_user[items_per_user >= min_items].index)

    data = data.sort_values(by="timestamp", ascending=False)
    data = data[data.groupby(["user_code"]).cumcount() == 0]

    data = data[data["user_code"].isin(users)]
    return dict(zip(data["user_code"], data["item_code"]))


class P3LTRTrainer(BaseRecommender, DataLoaderSaver):
    """
    P3LTR model training class.
    """

    def __init__(
        self,
        learning_rate=0.1,
        regularization=0.1,
        batch_size=100,
        iterations=100,
        top_k=10,
        loss="log_ratio",
        user_selection="random",
    ):
        super().__init__()

        # data
        self.interactions = None

        # model parameters
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.iterations = iterations
        self.top_k = top_k
        self.loss = loss
        self.user_selection = user_selection

        self.g = None
        self.features = None
        self.feature_encoders = None
        self.feature_preprocessor = None
        self.validation_user_item = None
        self.monitoring_metrics = {}

    def preprocess(self):
        """
        Prepare interactions dataset for training model
        """

        data = self.interactions.copy()

        user_id_code = {v: k for k, v in enumerate(data["user"].unique())}
        data["user_code"] = data["user"].apply(user_id_code.get)

        nb_users = len(user_id_code)
        item_id_code = {v: k + nb_users for k, v in enumerate(data["item"].unique())}
        data["item_code"] = data["item"].apply(item_id_code.get)

        self.data = data

        # initialize preprocessor and preprocess data
        self.feature_preprocessor = FeaturePreprocessor()
        self.features = self.feature_preprocessor.preprocess(data)

        # calculate target_users
        self.validation_user_item = get_validation_user_item(data)

        # initalize feature_encoder
        self.feature_encoders = [
            FeatureEncoder(
                len(self.feature_preprocessor.event_mapping), init_degree_param=0.0
            )
            for i in range(2)
        ]
        self.feature_encoders = self.feature_encoders + [
            FeatureEncoder(
                len(self.feature_preprocessor.event_mapping), init_degree_param=-1.0
            )
        ]

        # # create a graph
        self.g = dgl.graph(
            (
                self.features["source"],
                self.features["destination"],
            )
        )

        self.g.ndata["weight"] = torch.zeros(self.g.num_nodes(), 1)  # to improve

    def fit(self):
        optimizer = torch.optim.Adam(
            [
                param
                for enc in self.feature_encoders
                for param in list(enc.parameters())
            ],
            lr=self.learning_rate,
        )

        target_users = list(self.validation_user_item.keys())

        self.monitoring_metrics["parameter_change_history"] = []
        self.monitoring_metrics["val_node_positions"] = pd.DataFrame()
        self.monitoring_metrics["losses"] = []

        for iteration in tqdm(range(self.iterations)):
            with torch.no_grad():
                self.monitoring_metrics["parameter_change_history"].append(
                    torch.cat(
                        [
                            param.clone().reshape([-1])
                            for enc in self.feature_encoders
                            for param in enc.parameters()
                        ]
                    ).reshape([-1, 1])
                )

            # update edge weights
            for i in range(3):
                self.g.edata[f"weight_{i}"] = self.feature_encoders[i](
                    destination_degree=self.features["destination_degree"],
                    recency=self.features["recency"],
                    events=self.features["events"],
                )

            # zero grad
            optimizer.zero_grad()
            loss = torch.tensor(0)

            # Forward pass
            for i in range(self.batch_size):
                if self.user_selection == "random":
                    node_id = random.choice(target_users)
                elif self.user_selection == "deterministic":
                    node_id = target_users[i]

                (
                    scores,
                    validation_node_score,
                    validation_node_position,
                ) = recommend_node(
                    g=self.g,
                    node_id=node_id,
                    top_k=self.top_k,
                    validation_node_id=self.validation_user_item[node_id],
                )

                if (
                    validation_node_score == 0
                ):  # if it's not possible to reach validation_node in 3 steps
                    continue

                self.monitoring_metrics["val_node_positions"] = pd.concat(
                    [
                        self.monitoring_metrics["val_node_positions"],
                        pd.DataFrame(
                            {
                                "iteration": [iteration],
                                "position": [int(validation_node_position)],
                            }
                        ),
                    ]
                )
                # Compute Loss
                loss = loss + loss_function_factory(self.loss)(
                    validation_node_score=validation_node_score,
                    validation_node_position=validation_node_position,
                    scores=scores,
                    parameters=torch.cat(
                        [
                            torch.nn.utils.parameters_to_vector(enc.parameters())
                            for enc in self.feature_encoders
                        ]
                    ),
                    regularization=self.regularization,
                )

            # print("Epoch {}: train loss: {}".format(iteration, loss.item()))
            self.monitoring_metrics["losses"].append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()
        self.monitoring_metrics["parameter_change_history"] = torch.cat(
            self.monitoring_metrics["parameter_change_history"], axis=1
        )
