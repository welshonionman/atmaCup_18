import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import LIGHT_DIR, TARGET_COLUMNS, TEST__FEATURES
from src.metrics import get_result, mae
from src.table.feature import (
    add_cnn_oof_features,
    add_count_pixel_per_label_features,
    add_depth_features,
    add_naive_predicted_positions,
    add_traffic_light_features,
    get_feature_blocks,
    make_shift_diff_feature,
)
from src.table.lgbm import get_fit_params, get_model
from src.table.preprocess import common_preprocess
from src.util import set_seed

warnings.filterwarnings("ignore")


class CFG:
    exp_name = "atmacup_18_gbdt"
    comp_dataset_path = "/kaggle/datasets/"

    submission_dir = f"/kaggle/outputs/{exp_name}/submit/"
    model_dir = f"/kaggle/outputs/{exp_name}/model/"
    log_path = f"/kaggle/outputs/{exp_name}/logs/{exp_name}.txt"
    fold_csv = "/kaggle/outputs/train_folds.csv"

    n_fold = 5


if __name__ == "__main__":
    for dir_ in [
        Path(CFG.submission_dir),
        Path(CFG.model_dir),
        Path(CFG.log_path).parent,
    ]:
        os.makedirs(dir_, exist_ok=True)

    set_seed(seed=41)

    train_df = pd.read_csv(CFG.fold_csv)
    test_df = pd.read_csv(TEST__FEATURES)

    # 共通の前処理
    train_df = common_preprocess(train_df)
    test_df = common_preprocess(test_df)

    ids = os.listdir(LIGHT_DIR)
    train_df, test_df, naive_pred_cols = add_naive_predicted_positions(
        train_df, test_df
    )

    # traffic_lightのデータを特徴量として追加
    train_df, test_df = add_traffic_light_features(train_df, test_df, ids)

    # CNNのOOF予測値を特徴量として追加
    cnn_exp_names = [
        "tf_efficientnet_b7_ns",
    ]
    train_df, test_df, oof_feat_cols = add_cnn_oof_features(
        train_df,
        test_df,
        cnn_exp_names,
    )

    # シフト・差分特徴量を追加
    use_cols = [
        "vEgo",
        "aEgo",
        "brakePressed",
        "gas",
        "gasPressed",
        "leftBlinker",
        "rightBlinker",
    ]
    use_cols += oof_feat_cols
    train_df, shift_feat_cols = make_shift_diff_feature(train_df, use_cols)
    test_df, shift_feat_cols = make_shift_diff_feature(test_df, use_cols)

    # # クラスごとのピクセル数の特徴量を追加
    count_pixel_per_label_df = pd.read_csv("/kaggle/dataset/count_pixels_per_label.csv")
    train_df, test_df, count_pixel_per_label_cols = add_count_pixel_per_label_features(
        train_df, test_df, count_pixel_per_label_df
    )

    # # 深度の特徴量を追加
    depth_df = pd.read_csv("/kaggle/dataset/depth.csv")
    train_df, test_df, depth_cols = add_depth_features(train_df, test_df, depth_df)

    # 数値特徴量の定義
    num_cols = [
        "vEgo",
        "aEgo",
        "steeringAngleDeg",
        "steeringTorque",
        "brakePressed",
        "gas",
        "gasPressed",
        "leftBlinker",
        "rightBlinker",
        "traffic_lights_counts",
    ]
    num_cols += ["scene_sec"]
    num_cols += naive_pred_cols
    num_cols += oof_feat_cols
    num_cols += shift_feat_cols
    # num_cols += count_pixel_per_label_cols
    # num_cols += depth_cols
    num_cols += ["scene_count"]

    # カテゴリカル変数の定義
    cat_label_cols = ["gearShifter"]
    # カテゴリカル変数の出現回数をエンコーディングするブロック
    cat_count_cols = []
    # カテゴリカル変数のテーブルを作成するブロック
    cat_te_cols = []

    train_df, test_df, train_feat, test_feat = get_feature_blocks(
        train_df, test_df, num_cols, cat_label_cols, cat_count_cols
    )

    y = train_df[TARGET_COLUMNS]
    folds = train_df["fold"]

    # LightGBMのモデル

    def main(train_df, X_train, y, folds, test_df):
        oof_predictions = np.zeros((X_train.shape[0], len(TARGET_COLUMNS)))
        test_predictions = np.zeros((test_df.shape[0], len(TARGET_COLUMNS)))

        for target_idx in range(len(TARGET_COLUMNS)):
            print(f"\n\ntarget {target_idx}")
            X_train_ = X_train.copy()
            test_df_ = test_df.copy()

            # X_train_ = drop_other_time_predictions(X_train, target_idx)
            # test_df_ = drop_other_time_predictions(test_df, target_idx)

            # X_train_ = drop_other_direction_predictions(X_train_, target_idx)
            # test_df_ = drop_other_direction_predictions(test_df_, target_idx)
            for fold in range(CFG.n_fold):
                print(f"\nTraining fold {fold}")
                target_col = TARGET_COLUMNS[target_idx]

                model_name = f"lgb_{target_col}"
                model = get_model(model_name, CFG.model_dir)
                fit_params = get_fit_params(model_name)

                trn_ind = folds != fold
                val_ind = folds == fold

                x_train, x_val = X_train_.loc[trn_ind], X_train_.loc[val_ind]
                y_train, y_val = y.loc[trn_ind, target_col], y.loc[val_ind, target_col]
                eval_set = [(x_val, y_val)]

                fit_params_fold = fit_params.copy()
                fit_params_fold["eval_set"] = eval_set

                model.fit(x_train, y_train, **fit_params_fold)

                model.save(fold)

                oof_predictions[val_ind, target_idx] = model.predict(x_val)

                test_predictions[:, target_idx] += model.predict(test_df_)

        score = mae(y.values, oof_predictions)
        print(f"oof result {score}")

        pred_cols = [f"pred_{i}" for i in range(len(TARGET_COLUMNS))]

        oof = train_df.copy()
        oof[pred_cols] = oof_predictions
        oof[TARGET_COLUMNS] = y

        oof_feat = X_train.copy()
        oof_feat[pred_cols] = oof_predictions
        oof_feat[TARGET_COLUMNS] = y

        best_score = get_result(oof)
        print(f"best_score: {best_score:<.4f}")

        # save
        oof_save_path = f"{CFG.submission_dir}/oof_gbdt.csv"
        oof_feat_save_path = f"{CFG.submission_dir}/oof_feat_gbdt.csv"
        test_sub_save_path = f"{CFG.submission_dir}/submission.csv"

        oof.to_csv(oof_save_path, index=False)
        oof_feat.to_csv(oof_feat_save_path, index=False)

        test_predictions /= 5

        test_df[TARGET_COLUMNS] = test_predictions
        test_df.to_csv(test_sub_save_path, index=False)
        test_df[TARGET_COLUMNS].to_csv(test_sub_save_path, index=False)

    main(train_df, train_feat, y, folds, test_feat)
    from src.util import line_notify

    line_notify("end")
