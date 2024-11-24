import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm.callback import record_evaluation

from src.constants import LIGHT_DIR, TARGET_COLUMNS, TEST__FEATURES
from src.metrics import get_result, mae
from src.table.feature import (
    add_cnn_oof_features,
    add_count_pixel_per_label_features,
    add_depth_features,
    add_traffic_light_features,
    get_feature_blocks,
    make_shift_diff_feature,
)
from src.table.lgbm import get_fit_params, get_model
from src.table.preprocess import common_preprocess
from src.util import set_seed

warnings.filterwarnings("ignore")


def save_learning_log(
    eval_result: dict, target_col: str, fold: int, save_dir: str = "./"
) -> None:
    """学習の評価結果をCSVファイルとして保存します。

    Args:
        eval_result (dict): 学習の評価結果を含む辞書
        fold (int): 交差検証のフォールド番号
        save_dir (str, optional): 保存先ディレクトリ. デフォルトは"./"
    """
    train_df = pd.DataFrame(eval_result["train"]).rename(
        columns={"l1": "train_mae", "l2": "train_mse"}
    )
    valid_df = pd.DataFrame(eval_result["valid"]).rename(
        columns={"l1": "valid_mae", "l2": "valid_mse"}
    )

    result_df = pd.concat([train_df, valid_df], axis=1)
    output_file = f"{save_dir}/{target_col}_fold_{fold}_result.csv"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)


class CFG:
    exp_name = "atmacup_18_gbdt"
    comp_dataset_path = "/kaggle/datasets/"

    submission_dir = f"/kaggle/outputs/{exp_name}/submit/"
    model_dir = f"/kaggle/outputs/{exp_name}/model/"
    log_path = f"/kaggle/outputs/{exp_name}/logs/{exp_name}.txt"
    fold_csv = "/kaggle/outputs/train_folds.csv"

    n_fold = 5
    optuna_n_trials = 25


if __name__ == "__main__":
    for dir_ in [
        Path(CFG.submission_dir),
        Path(CFG.model_dir),
        Path(CFG.log_path).parent,
    ]:
        os.makedirs(dir_, exist_ok=True)

    set_seed()

    train_df = pd.read_csv(CFG.fold_csv)
    test_df = pd.read_csv(TEST__FEATURES)

    # 共通の前処理
    train_df = common_preprocess(train_df)
    test_df = common_preprocess(test_df)

    ids = os.listdir(LIGHT_DIR)

    # traffic_lightのデータを特徴量として追加
    train_df, test_df = add_traffic_light_features(train_df, test_df, ids)

    # CNNのOOF予測値を特徴量として追加
    cnn_exp_names = [
        "tf_efficientnet_b7_ns",
        # "maxvit_base_tf_384",
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
        "brake",
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
        "brake",
        "brakePressed",
        "gas",
        "gasPressed",
        "leftBlinker",
        "rightBlinker",
        "traffic_lights_counts",
    ]
    num_cols += ["scene_sec"]
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
        param_template = {
            "learning_rate": 0.01,
            "n_estimators": 10000000000,
            "num_threads": 8,
        }

        def objective(trial):
            # チューニング対象のパラメータ
            param = {
                "objective": "regression",
                "metric": "mae",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.4, 0.8, step=0.05
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.4, 0.8, step=0.05
                ),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            }

            # 固定パラメータとチューニングパラメータを結合
            param.update(param_template)

            # 学習データセットの作成
            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

            # モデルの学習
            model = lgb.train(
                param,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=param["n_estimators"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=2000, verbose=True),
                ],
            )

            # 検証データでの予測と評価
            pred_val = model.predict(x_val)
            mae_score = mae(y_val.values.reshape(-1, 1), pred_val.reshape(-1, 1))

            return mae_score

        oof_predictions = np.zeros((X_train.shape[0], len(TARGET_COLUMNS)))
        test_predictions = np.zeros((test_df.shape[0], len(TARGET_COLUMNS)))

        for target_idx in range(len(TARGET_COLUMNS)):
            print(f"\n\ntarget {target_idx}")
            best_params = {}  # 各ターゲットの最適パラメータを保存
            eval_result = {}
            for fold in range(CFG.n_fold):
                print(f"\nTraining fold {fold}")
                target_col = TARGET_COLUMNS[target_idx]

                trn_ind = folds != fold
                val_ind = folds == fold

                x_train, x_val = X_train.loc[trn_ind], X_train.loc[val_ind]
                y_train, y_val = y.loc[trn_ind, target_col], y.loc[val_ind, target_col]

                # Optunaによるハイパーパラメータ最適化
                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=CFG.optuna_n_trials)

                # 最適化されたパラメータの取得
                best_params = study.best_params
                best_params.update(param_template)
                print(f"Best params for target {target_col}, fold {fold}:", best_params)

                print()
                model = lgb.LGBMRegressor(**best_params)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_train, y_train), (x_val, y_val)],
                    eval_names=["train", "valid"],
                    eval_metric=["mae"],
                    callbacks=[
                        record_evaluation(eval_result),
                        lgb.early_stopping(stopping_rounds=2000, verbose=True),
                    ],
                )

                # 予測
                val_pred = model.predict(x_val)
                oof_predictions[val_ind, target_idx] = val_pred
                test_predictions[:, target_idx] += model.predict(test_df)

                # OOFスコアの計算と表示
                val_score = mae(y_val.values.reshape(-1, 1), val_pred.reshape(-1, 1))
                print(f"Fold {fold} OOF MAE Score: {val_score:.4f}")
                save_dir = f"/kaggle/outputs/{CFG.exp_name}/logs"
                save_learning_log(eval_result, target_col, fold, save_dir)

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

    # train_dfの列と、train_featの列を比較
    main(train_df, train_feat, y, folds, test_feat)
