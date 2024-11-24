import pandas as pd

from src.table.preprocess import create_time_series_features


def add_count_pixel_per_label_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    count_pixel_per_label_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """セマンティックセグメンテーションのラベルごとのピクセル数に関する特徴量を追加する関数

    Args:
        train_df (pd.DataFrame): 学習データのデータフレーム
        test_df (pd.DataFrame): テストデータのデータフレーム
        count_pixel_per_label_df (pd.DataFrame): ラベルごとのピクセル数が格納されたデータフレーム

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - 特徴量を追加した学習データのデータフレーム
            - 特徴量を追加したテストデータのデータフレーム
            - 追加した特徴量のカラム名のリスト

    Notes:
        以下のラベルについて、時系列特徴量を作成します:
        road, wall, car, building, tree, floor, sky, fence, sidewalk, grass, person

        各ラベルについて、t, t-0.5, t-1.0の3時点の値を特徴量として追加します。
    """
    features = [
        "road",
        "wall",
        "car",
        "building",
        "tree",
        "floor",
        "sky",
        "fence",
        "sidewalk",
        "grass",
        "person",
    ]

    count_pixel_per_label_df = create_time_series_features(
        count_pixel_per_label_df, features
    )
    cols = [c for c in count_pixel_per_label_df.columns if c != "ID"]

    train_df = pd.merge(train_df, count_pixel_per_label_df, on="ID", how="left")
    test_df = pd.merge(test_df, count_pixel_per_label_df, on="ID", how="left")

    return train_df, test_df, cols
