import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import TARGET_COLUMNS


class AbstractBaseBlock:
    """特徴量エンジニアリングのための基底クラス

    全ての特徴量ブロックはこのクラスを継承する必要があります。
    fit()とtransform()メソッドを実装する必要があります。

    参考: https://www.guruguru.science/competitions/16/discussions/95b7f8ec-a741-444f-933a-94c33b9e66be/
    """

    def __init__(self) -> None:
        pass

    def fit(self, input_df: pd.DataFrame, y=None) -> pd.DataFrame:
        """特徴量の学習を行うメソッド

        Args:
            input_df (pd.DataFrame): 入力データフレーム
            y: 目的変数(デフォルト: None)

        Returns:
            pd.DataFrame: 変換後のデータフレーム
        """
        raise NotImplementedError()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量の変換を行うメソッド

        Args:
            input_df (pd.DataFrame): 入力データフレーム

        Returns:
            pd.DataFrame: 変換後のデータフレーム
        """
        raise NotImplementedError()


def run_block(input_df: pd.DataFrame, blocks: list[AbstractBaseBlock], is_fit):
    """特徴量ブロックを実行する関数

    Args:
        input_df (pd.DataFrame): 入力データフレーム
        blocks (list[AbstractBaseBlock]): 実行する特徴量ブロックのリスト
        is_fit (bool): fitを実行するかどうか

    Returns:
        pd.DataFrame: 全ての特徴量ブロックを適用した後のデータフレーム
    """
    output_df = pd.DataFrame()
    for block in blocks:
        name = block.__class__.__name__

        if is_fit:
            _df = block.fit(input_df)
        else:
            _df = block.transform(input_df)

        output_df = pd.concat([output_df, _df], axis=1)
    return output_df


class NumericBlock(AbstractBaseBlock):
    """数値特徴量をそのまま出力するブロック

    Args:
        col (str): 対象とする列名
    """

    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].copy()
        return output_df


class LabelEncodingBlock(AbstractBaseBlock):
    """カテゴリカル変数をラベルエンコーディングするブロック

    Args:
        col (str): 対象とする列名
    """

    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col
        self.encoder = LabelEncoder()

    def fit(self, input_df):
        self.encoder.fit(input_df[self.col])
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = self.encoder.transform(input_df[self.col])
        return output_df.add_suffix("@le")


class CountEncodingBlock(AbstractBaseBlock):
    """カテゴリカル変数の出現回数をエンコーディングするブロック

    Args:
        col (str): 対象とする列名
    """

    def __init__(self, col: str) -> None:
        super().__init__()
        self.col = col

    def fit(self, input_df):
        self.val_count_dict = {}
        self.val_count = input_df[self.col].value_counts()
        return self.transform(input_df)

    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df[self.col] = input_df[self.col].map(self.val_count)
        return output_df.add_suffix("@ce")


def get_feature_blocks(
    train_df,
    test_df,
    num_cols,
    cat_label_cols,
    cat_count_cols,
):
    blocks = [
        *[NumericBlock(col) for col in num_cols],
        *[LabelEncodingBlock(col) for col in cat_label_cols],
        *[CountEncodingBlock(col) for col in cat_count_cols],
    ]

    train_num = len(train_df)
    whole_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    whole_feat_df = run_block(whole_df, blocks, is_fit=True)

    train_df, test_df = (
        whole_df.iloc[:train_num],
        whole_df.iloc[train_num:].drop(columns=TARGET_COLUMNS).reset_index(drop=True),
    )
    train_feat, test_feat = (
        whole_feat_df.iloc[:train_num],
        whole_feat_df.iloc[train_num:].reset_index(drop=True),
    )

    blocks = []

    _df = run_block(train_df, blocks, is_fit=True)
    train_feat = pd.concat([train_feat, _df], axis=1)
    _df = run_block(test_df, blocks, is_fit=False)
    test_feat = pd.concat([test_feat, _df], axis=1)

    print("use_col len", len(train_feat.columns))
    return train_df, test_df, train_feat, test_feat
