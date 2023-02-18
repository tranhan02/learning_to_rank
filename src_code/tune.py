import pandas as pd
import optuna
import optuna.integration.lightgbm as lgb
from parse import get_tune_args
from preprocess import get_data

def main():
    args = get_tune_args()
    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(
        args["data_path"]
    )

    def objective(trial):

        param = {
            "num_leaves": trial.suggest_int("num_leaves", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        }

        gbm = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100, #số lượng cây mà mô hình sẽ sử dụng trong quá trình huấn luyện.
            num_leaves=param["num_leaves"],
            learning_rate=param["learning_rate"],
            reg_lambda=param["reg_lambda"],
            reg_alpha=param["reg_alpha"],
            #ndcg_eval_at=[1,2,3,4,5],
        )

        gbm.fit(
            X_train,
            y_train,
            group=group_train,
            eval_group=[group_vali],
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,#50,
        )

        return gbm.best_score_["valid_0"]["ndcg@1"]

    study = optuna.create_study(direction="maximize") #hướng tối ưu hóa ndch@1 max
    study.optimize(objective, n_trials=10) #trial" thường được hiểu là một lần thử nghiệm hoặc đào tạo mô hình trên một tập dữ liệu nhất định
    best_params = study.best_trial.params

    print("Best trial:", best_params)

if __name__ == "__main__":
    main()
