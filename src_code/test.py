import pandas as pd
import optuna
import optuna.integration.lightgbm as lgb
from parse import get_test_args
from sklearn.metrics import ndcg_score
from preprocess import get_data

def main():
    args = get_test_args()
    X_train, X_test, X_val, y_train, y_test, y_val, group_vali, group_train = get_data(
        args["data_path"]
    )

    gbm = lgb.Booster(model_file=args["model_path"])
    #gbm = lgb.Booster(model_file="E:\Lambdarank\src_code\model.txt")
    true_relevance = y_test.sort_values(ascending=False) #sắp xếp giảm dần

    y_pred = gbm.predict(X_test)
    #pd.DataFrame(y_pred).to_csv("output2.txt")
    y_score = pd.DataFrame({"relevance_score": y_test, "predicted_ranking": y_pred})
   
    y_score = y_score.sort_values("predicted_ranking", ascending=False)
    #score = ndcg_score(y_true, y_score)
    score = ndcg_score(
            [true_relevance.to_numpy()], [y_score["relevance_score"].to_numpy()]
        ),
    print("nDCG score: ", score[0])

if __name__ == "__main__":
    main()

#model1
#nDCG score:  0.9331625226697012

# python test.py ../data/fold5 ../src_code/model5.txt, output2
# nDCG score:  0.9398968218931246
# (py37) E:\Lambdarank\src_code>python test.py ../data/Fold4 ../src_code/model5.txt
# nDCG score:  0.9467609303594786
# (py37) E:\Lambdarank\src_code>python test.py ../data/Fold3 ../src_code/model5.txt
# nDCG score:  0.9457426185721445