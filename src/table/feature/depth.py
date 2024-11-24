import pandas as pd

from src.table.preprocess import create_time_series_features


def add_depth_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, depth_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "max_depth_0",
        "max_depth_1",
        "max_depth_2",
        "max_depth_3",
        "max_depth_4",
        "max_depth_5",
    ]
    depth_df = create_time_series_features(depth_df, features)
    cols = [c for c in depth_df.columns if c != "ID"]

    train_df = pd.merge(train_df, depth_df, on="ID", how="left")
    test_df = pd.merge(test_df, depth_df, on="ID", how="left")
    return train_df, test_df, cols
