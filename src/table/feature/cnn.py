import pandas as pd

from src.constants import TARGET_COLUMNS


def add_cnn_oof_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cnn_exp_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """CNNモデルのOOF予測値を特徴量として追加する関数

    Args:
        train_df (pd.DataFrame): 学習データ
        test_df (pd.DataFrame): テストデータ
        cnn_exp_names (list): CNNの実験名のリスト

    Returns:
        tuple: 以下の要素を含むタプル
            - train_df (pd.DataFrame): OOF特徴量を追加した学習データ
            - test_df (pd.DataFrame): OOF特徴量を追加したテストデータ
            - oof_feat_cols (list): 追加したOOF特徴量のカラム名のリスト
    """
    oof_feat_cols = []

    for cnn_exp_name in cnn_exp_names:
        oof_cv_path = f"/kaggle/outputs/{cnn_exp_name}/submit/oof_cv.csv"
        sub_oof_path = f"/kaggle/outputs/{cnn_exp_name}/submit/submission_oof.csv"

        _oof_feat_cols = [f"{cnn_exp_name}_{c}" for c in TARGET_COLUMNS]
        oof_feat_cols.extend(_oof_feat_cols)

        cnn_train_df = pd.read_csv(oof_cv_path)
        pred_cols = [f"pred_{i}" for i in range(len(TARGET_COLUMNS))]
        train_df[_oof_feat_cols] = cnn_train_df[pred_cols]

        cnn_test_df = pd.read_csv(sub_oof_path)
        test_df[_oof_feat_cols] = cnn_test_df[TARGET_COLUMNS]

    return train_df, test_df, oof_feat_cols
