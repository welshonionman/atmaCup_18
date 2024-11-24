import pandas as pd


def common_preprocess(target_df: pd.DataFrame) -> pd.DataFrame:
    """データフレームに対して共通の前処理を行う関数

    Args:
        target_df (pd.DataFrame): 前処理対象のデータフレーム

    Returns:
        pd.DataFrame: 前処理後のデータフレーム

    Notes:
        以下の処理を行う:
        - bool型カラムをint型に変換
        - IDカラムからsceneとscene_secを抽出
        - シーンごとのデータ数をカウントしてscene_countカラムを追加
    """
    bool_cols = ["brakePressed", "gasPressed", "leftBlinker", "rightBlinker"]
    target_df[bool_cols] = target_df[bool_cols].astype(int)

    target_df["scene"] = target_df["ID"].str.split("_").str[0]
    target_df["scene_sec"] = target_df["ID"].str.split("_").str[1].astype(int)

    count_df = target_df.groupby("scene").size()
    target_df["scene_count"] = target_df["scene"].map(count_df)
    return target_df


def create_time_series_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """画像の時系列特徴量を作成する関数

    Args:
        df (pd.DataFrame): 入力データフレーム。以下のカラムが必要:
            - path: 画像パス (例: "scene_id/image_t.png")
            - feature_cols で指定された特徴量カラム
        feature_cols (list[str]): ピボット処理する特徴量カラムのリスト

    Returns:
        pd.DataFrame: 時系列特徴量を含むデータフレーム
            - 各特徴量について t, t-0.5, t-1.0 の3時点の値を含む
            - インデックスは scene_id

    Notes:
        - 入力画像は t.png, t-0.5.png, t-1.0.png の3時点分が必要
        - 出力の各特徴量カラム名は {feature}_t, {feature}_t-0.5, {feature}_t-1.0 となる
    """
    df["ID"] = df["path"].str.split("/").str[0]
    df["fname"] = df["path"].str.split("/").str[1]

    pivot_dfs = []

    for feature in feature_cols:
        pivot = df.pivot_table(
            index=["ID"], columns="fname", values=feature, aggfunc="first"
        ).rename(
            columns={
                "image_t.png": f"{feature}_t",
                "image_t-0.5.png": f"{feature}_t-0.5",
                "image_t-1.0.png": f"{feature}_t-1.0",
            }
        )[[f"{feature}_t", f"{feature}_t-0.5", f"{feature}_t-1.0"]]  # 列の順番を指定
        pivot_dfs.append(pivot)

    result_df = pd.concat(pivot_dfs, axis=1).reset_index()

    return result_df
