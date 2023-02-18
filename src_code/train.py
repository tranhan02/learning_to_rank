import pandas as pd
import optuna
import optuna.integration.lightgbm as lgb
from parse import get_train_args
from preprocess import get_data

def main():
    args = get_train_args()
    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(
        args["data_path"]
    )

    gbm1 = lgb.Booster(model_file="model4.txt")
    gbm = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=10000,
        num_leaves=args["num_leaves"],
        learning_rate=args["learning_rate"],
        reg_lambda=args["reg_lambda"],
        reg_alpha=args["reg_alpha"],
    )
    
    gbm.fit(
        X_train,
        y_train,
        group=group_train,
        eval_group=[group_vali],
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=150,#200,
        init_model = gbm1,
    )
   
    gbm.booster_.save_model(args["output_file_name"], num_iteration=gbm.best_iteration_) #Mô hình sẽ được lưu trữ chỉ với số lần lặp tốt nhất (best_iteration_) được xác định trong quá trình huấn luyện

if __name__ == "__main__":
    main()

#model1.txt
# Best is trial 8 with value: 0.49003333333333343.
# Best trial: {'num_leaves': 91, 'learning_rate': 0.0693789884886769, 'reg_lambda': 4.222656475228009, 'reg_alpha': 1.260163148375562}
