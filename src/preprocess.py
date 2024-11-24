import os
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import (
    GroupKFold,
)

from src.constants import IMAGE_DIR, TRAIN_FEATURES


def add_id_scene_id_sec(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームにシーンIDとフレーム番号の列を追加する

    Args:
        df (pd.DataFrame): 入力データフレーム。'ID'列が必要。
            'ID'列は'{シーンID}_{フレーム番号}'の形式である必要がある。
            例: '00066be8e20318869c38c66be466631a_320'

    Returns:
        pd.DataFrame: 以下の2列が追加されたデータフレーム
            - id_scene (str): シーンID ('_'より前の部分)
            - id_sec (int): フレーム番号 ('_'より後の部分を整数型に変換)

    Note:
        - 入力データフレームは変更される
        - 'ID'列の形式が想定と異なる場合、エラーが発生する可能性がある
    """
    df["id_scene"] = df["ID"].str.split("_").str[0]
    df["id_sec"] = df["ID"].str.split("_").str[1].astype(int)
    return df


def get_image(df_feature_train: pd.DataFrame, idx: int) -> Image.Image:
    """データフレームの指定された行に対応する画像を読み込む

    Args:
        df_feature_train (pd.DataFrame): 特徴量データフレーム。'ID'列が必要。
        idx (int): 読み込みたい画像に対応する行のインデックス

    Returns:
        Image.Image: 読み込まれた画像

    Note:
        - 画像は'{IMAGE_DIR}/{ID}/image_t.png'のパスから読み込まれる
        - 画像ファイルが存在しない場合、FileNotFoundErrorが発生する
    """
    row = df_feature_train.iloc[idx]
    id_ = row["ID"]
    return Image.open(f"{IMAGE_DIR}/{id_}/image_t.png")


def get_fold(train, n_fold, group_col):
    """データフレームに交差検証用のfoldを追加する関数

    Args:
        train (pd.DataFrame): 学習データのデータフレーム
        cfg (Config): 設定情報を含むConfigオブジェクト。以下の属性が必要:
            - fold_type (str): 分割方法 ("kf", "skf", "gkf", "sgkf")
            - n_fold (int): 分割数
            - seed (int): 乱数シード
            - target_col (str): 目的変数のカラム名 (KFoldの場合)
            - skf_col (str): 層化に使用するカラム名 (StratifiedKFold, StratifiedGroupKFoldの場合)
            - group_col (str): グループ分けに使用するカラム名 (GroupKFold, StratifiedGroupKFoldの場合)

    Returns:
        pd.DataFrame: fold列が追加されたデータフレーム
    """
    fold = GroupKFold(n_splits=n_fold)
    groups = train[group_col].values
    kf = fold.split(train, train[group_col], groups)

    for n, (train_index, val_index) in enumerate(kf):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)

    return train


def make_train_folds(csv_path: str, output_path: str, n_fold: int) -> None:
    """学習データを読み込み、交差検証用のfold情報を付与してCSVファイルとして保存する関数

    Args:
        csv_path (str): 学習データのCSVファイルパス

    Returns:
        None: 処理結果はCSVファイルとして保存される

    Note:
        - 学習データを読み込み、シーン情報を抽出
        - get_fold()関数でfold情報を付与
        - fold情報付きのデータをCSVファイルとして保存
    """
    train_df = pd.read_csv(csv_path)

    train_df["scene"] = train_df["ID"].str.split("_").str[0]

    train_df = get_fold(train_df, n_fold, "scene")

    os.makedirs(Path(output_path).parent, exist_ok=True)

    train_df.to_csv(output_path, index=False)
