import pandas as pd


def add_shift_feature(
    df: pd.DataFrame, col: str, shift: int, shift_cols: list[str]
) -> tuple[pd.DataFrame, list[str], str]:
    """シフト特徴量を追加する関数

    Args:
        df (pd.DataFrame): 入力データフレーム
        col (str): シフトする列名
        shift (int): シフトする数
        shift_cols (list[str]): シフト特徴量の列名リスト

    Returns:
        tuple: 以下の要素を含むタプル
            - df (pd.DataFrame): シフト特徴量を追加したデータフレーム
            - shift_cols (list[str]): 更新されたシフト特徴量の列名リスト
            - shift_col (str): 追加したシフト特徴量の列名
    """
    shift_col = f"{col}_shift{shift}"
    df[shift_col] = df.groupby("scene")[col].shift(shift)
    shift_cols.append(shift_col)
    return df, shift_cols, shift_col


def add_diff_feature(
    df: pd.DataFrame, col: str, shift_col: str, shift: int, shift_cols: list[str]
) -> tuple[pd.DataFrame, list[str], str]:
    """差分特徴量を追加する関数

    Args:
        df (pd.DataFrame): 入力データフレーム
        col (str): 元の列名
        shift_col (str): シフトした列名
        shift (int): シフト数
        shift_cols (list[str]): 特徴量の列名リスト

    Returns:
        tuple: 以下の要素を含むタプル
            - df (pd.DataFrame): 差分特徴量を追加したデータフレーム
            - shift_cols (list[str]): 更新された特徴量の列名リスト
            - diff_col (str): 追加した差分特徴量の列名
    """
    diff_col = f"{col}_diff{shift}"
    df[diff_col] = df[col] - df[shift_col]
    shift_cols.append(diff_col)
    return df, shift_cols, diff_col


def make_shift_diff_feature(
    df: pd.DataFrame, use_feat_cols: list[str], shift_count: int = 1
) -> tuple[pd.DataFrame, list[str]]:
    """シフトと差分の特徴量を作成する関数

    Args:
        df (pd.DataFrame): 入力データフレーム
        use_feat_cols (list[str]): 特徴量を作成する対象の列名リスト
        shift_count (int, optional): シフトする最大数. デフォルトは1

    Returns:
        tuple: 以下の要素を含むタプル
            - df (pd.DataFrame): 特徴量を追加したデータフレーム
            - shift_cols (list[str]): 追加した特徴量の列名リスト
    """
    shift_range = list(range(-shift_count, shift_count + 1))
    shift_range = [x for x in shift_range if x != 0]

    df["ori_idx"] = df.index

    df = df.sort_values(["scene", "scene_sec"]).reset_index(drop=True)

    shift_cols = []
    for shift in shift_range:
        for col in use_feat_cols:
            df, shift_cols, shift_col = add_shift_feature(df, col, shift, shift_cols)
            df, shift_cols, diff_col = add_diff_feature(
                df, col, shift_col, shift, shift_cols
            )

    df = df.sort_values("ori_idx").reset_index(drop=True)
    df = df.drop("ori_idx", axis=1)

    return df, shift_cols
