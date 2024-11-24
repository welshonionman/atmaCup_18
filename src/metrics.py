import numpy as np
import pandas as pd

from src.constants import TARGET_COLUMNS


def mae(true_values, pred_values):
    """予測値と実測値の平均絶対誤差(MAE)を計算する関数

    Args:
        true_values (numpy.ndarray): 実測値の配列
        pred_values (numpy.ndarray): 予測値の配列

    Returns:
        float: 平均絶対誤差(MAE)の値
    """
    abs_diff = abs(true_values - pred_values)
    mae = np.mean(abs_diff.reshape(-1))
    return mae


def get_result(result_df: pd.DataFrame) -> float:
    """予測結果のスコアを計算する関数

    予測結果のDataFrameから予測値とラベルを取り出し、平均絶対誤差(MAE)スコアを計算します。
    計算されたスコアはログに出力され、関数の戻り値として返されます。

    Args:
        result_df (pd.DataFrame): 予測結果とラベルを含むDataFrame
            - pred_{i}列: モデルによる予測値。i=0からtarget_size-1までの連番で、
                         各予測対象の座標値に対応します
            - CFG.target_col列: 正解ラベル。予測対象の実際の座標値が格納されています

    処理の流れ:
        1. DataFrameから予測値の列(pred_0 ~ pred_{target_size-1})を抽出
        2. DataFrameから正解ラベルの列(CFG.target_col)を抽出
        3. 予測値と正解値の平均絶対誤差(MAE)を計算
        4. スコアをログに出力
        5. スコアを戻り値として返す

    Returns:
        float: 予測値とラベルのMAEスコア。値が小さいほど予測精度が高いことを示します。
              スコアは0以上の実数値で、完全に一致する場合は0となります。

    Note:
        - スコアの計算にはget_score()関数を使用しています
        - ログ出力には小数点以下4桁までの固定幅フォーマットを使用しています
        - CFG.target_sizeはモデルの予測対象数を指定する設定値です
    """
    pred_cols = [f"pred_{i}" for i in range(len(TARGET_COLUMNS))]
    preds = result_df[pred_cols].values
    labels = result_df[TARGET_COLUMNS].values
    score = mae(labels, preds)
    return score
