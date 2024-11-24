import lightgbm as lgb


class LightGBM:
    def __init__(
        self,
        lgb_params,
        save_dir=None,
        categorical_feature=None,
        model_name="lgb",
        stopping_rounds=2000,
    ) -> None:
        self.save_dir = save_dir
        self.lgb_params = lgb_params
        self.categorical_feature = categorical_feature

        # saveの切り替え用
        self.model_name = model_name

        self.stopping_rounds = stopping_rounds

    def fit(self, x_train, y_train, **fit_params) -> None:
        X_val, y_val = fit_params["eval_set"][0]
        del fit_params["eval_set"]

        train_dataset = lgb.Dataset(
            x_train, y_train, categorical_feature=self.categorical_feature
        )

        val_dataset = lgb.Dataset(
            X_val, y_val, categorical_feature=self.categorical_feature
        )

        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.stopping_rounds, verbose=True),
                # lgb.log_evaluation(500),
            ],
            **fit_params,
        )

    def save(self, fold):
        save_to = f"{self.save_dir}/lgb_fold_{fold}_{self.model_name}.txt"
        self.model.save_model(save_to)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


def get_model(model_name, model_dir):
    lgb_params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "verbose": -1,
        "n_jobs": 8,
        "seed": 42,
        "learning_rate": 0.01,
        "metric": "mae",
        "num_leaves": 64,
        "max_depth": 5,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "drop_seed": 42,
    }
    model = LightGBM(
        lgb_params=lgb_params,
        save_dir=model_dir,
        model_name=model_name,
    )

    return model


def get_fit_params(model_name):
    params = {"num_boost_round": 100000}

    return params
