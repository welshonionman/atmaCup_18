import json

import pandas as pd

from src.constants import LIGHT_DIR


def add_traffic_light_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, ids: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """信号機の特徴量を追加する関数

    Args:
        train_df (pd.DataFrame): 学習データ
        test_df (pd.DataFrame): テストデータ
        ids (list): 信号機データのIDリスト

    Returns:
        tuple: 以下の要素を含むタプル
            - train_df (pd.DataFrame): 信号機特徴量を追加した学習データ
            - test_df (pd.DataFrame): 信号機特徴量を追加したテストデータ
    """
    traffic_lights = []
    id_class_list = []
    for id_ in ids:
        path = LIGHT_DIR + f"{id_}"

        traffic_light = json.load(open(path))

        traffic_lights.append(traffic_light)

        for light in traffic_light:
            id_class_list.append((id_.split(".")[0], light["class"]))

    counts = [len(traffic_light) for traffic_light in traffic_lights]
    traffic_lights_df = pd.DataFrame(id_class_list, columns=["ID", "class"])
    traffic_lights_df["class"].value_counts()

    # traffic_lights_dfを作成
    ids = [id_.split(".")[0] for id_ in ids]

    traffic_lights_df = pd.DataFrame({"ID": ids, "traffic_lights_counts": counts})
    train_df = pd.merge(train_df, traffic_lights_df, on="ID", how="left")
    test_df = pd.merge(test_df, traffic_lights_df, on="ID", how="left")

    return train_df, test_df


def add_traffic_light_features_2(
    train_df: pd.DataFrame, test_df: pd.DataFrame, ids: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """信号機の特徴量を追加する関数

    Args:
        train_df (pd.DataFrame): 学習データ
        test_df (pd.DataFrame): テストデータ
        ids (list): 信号機データのIDリスト

    Returns:
        tuple: 以下の要素を含むタプル
            - train_df (pd.DataFrame): 信号機特徴量を追加した学習データ
            - test_df (pd.DataFrame): 信号機特徴量を追加したテストデータ
    """
    traffic_lights = []
    id_class_list = []
    for id_ in ids:
        path = f"{LIGHT_DIR}/{id_}"
        traffic_light = json.load(open(path))
        traffic_lights.append(traffic_light)

        for light in traffic_light:
            scene_id = id_.split(".")[0]
            class_ = light["class"]
            bbox = light["bbox"]
            bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            id_class_list.append((scene_id, class_, bbox_size))

    traffic_lights_df = pd.DataFrame(
        id_class_list, columns=["ID", "class", "bbox_size"]
    )
    traffic_lights_df = traffic_lights_df.pivot_table(
        index="ID", columns="class", values="bbox_size", aggfunc="max"
    )
    train_df = pd.merge(train_df, traffic_lights_df, on="ID", how="left")
    test_df = pd.merge(test_df, traffic_lights_df, on="ID", how="left")
    return train_df, test_df
