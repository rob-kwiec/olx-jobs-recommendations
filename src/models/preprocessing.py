"""Module implementing preprocessing classes used by models"""

import pandas as pd
import torch


def prepare_event_features(
    df, user_column, item_column, event_column, timestamp_column, event_mapping
):
    event_code_column = "event_code"
    df[event_code_column] = df[event_column].apply(lambda x: event_mapping[x])
    df = pd.get_dummies(df, columns=[event_code_column])
    event_columns = [f"{event_code_column}_{i}" for i in range(len(event_mapping))]
    df = (
        df.groupby([user_column, item_column])
        .agg(
            {
                **{timestamp_column: "max"},
                **{event_column: "sum" for event_column in event_columns},
            }
        )
        .reset_index()
    )
    return df, event_columns


def prepare_source_degree_and_recency_features(df, timestamp_column):
    max_timestamp = (
        df.groupby("source")
        .agg({timestamp_column: "max", "destination": "count"})
        .reset_index()
        .rename(
            columns={
                timestamp_column: "max_timestamp_user",
                "destination": "source_degree",
            }
        )
    )
    df = pd.merge(df, max_timestamp, on="source")
    df["recency"] = (df["max_timestamp_user"] - df[timestamp_column]) / 3600 / 24
    df = df.drop(columns=["max_timestamp_user", timestamp_column])
    return df


class FeaturePreprocessor:
    def __init__(self, event_mapping=None):
        self.event_mapping = event_mapping

    def _get_mappings(self, df):
        self.event_mapping = {v: k for k, v in enumerate(df["event"].unique())}

    def preprocess(
        self,
        df,
        user_column="user_code",
        item_column="item_code",
        event_column="event",
        timestamp_column="timestamp",
    ):
        if self.event_mapping is None:
            self._get_mappings(df)

        df, event_columns = prepare_event_features(
            df=df,
            user_column=user_column,
            item_column=item_column,
            event_column=event_column,
            timestamp_column=timestamp_column,
            event_mapping=self.event_mapping,
        )

        df = df.rename(
            columns={
                user_column: "source",
                item_column: "destination",
            }
        )

        df = prepare_source_degree_and_recency_features(
            df=df, timestamp_column=timestamp_column
        )

        # calculate destination degree
        destination_degrees = (
            df.groupby("destination")
            .size()
            .reset_index()
            .rename(columns={0: "destination_degree"})
        )
        df = pd.merge(df, destination_degrees, on="destination")

        df["is_direction_ui"] = True
        df = df[
            [
                "is_direction_ui",
                "source",
                "destination",
                "source_degree",
                "destination_degree",
                "recency",
            ]
            + event_columns
        ]

        # add interactions from items to users
        df_item_user = df[
            [
                "is_direction_ui",
                "destination",
                "source",
                "destination_degree",
                "source_degree",
                "recency",
            ]
            + event_columns
        ]
        df_item_user["is_direction_ui"] = False
        df_item_user.columns = df.columns

        df = pd.concat([df, df_item_user])

        return {
            "is_direction_ui": df["is_direction_ui"].values,
            "source": df["source"].values,
            "destination": df["destination"].values,
            "destination_degree": torch.Tensor(df[["destination_degree"]].values),
            "recency": torch.Tensor(df[["recency"]].values),
            "events": torch.Tensor(df[event_columns].values),
        }
