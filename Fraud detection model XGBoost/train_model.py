import json
import pickle

import pandas as pd
import xgboost as xgb

hyperparameters = {
    "max_depth": 3,
    "eta": 0.2,
    "objective": "binary:logistic",
    "num_round": 100,
    "early_stopping_rounds": 10,
    "nfold": 5
}

data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")

train = data_train.drop(["fraud", "customer_education"], axis=1)
test = data_test.drop(["fraud", "customer_education"], axis=1)

label_train = pd.DataFrame(data_train["fraud"])
label_test = pd.DataFrame(data_test["fraud"])

dtrain = xgb.DMatrix(train, label=label_train)
dtest = xgb.DMatrix(test, label=label_test)

params = {"max_depth": hyperparameters.get("max_depth"), "eta": hyperparameters.get("eta"),
          "objective": hyperparameters.get("objective")}
num_boost_round = hyperparameters['num_round']
nfold = hyperparameters['nfold']
early_stopping_rounds = hyperparameters['early_stopping_rounds']

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    nfold=nfold,
    early_stopping_rounds=early_stopping_rounds,
    metrics=("auc"),
    seed=0,
)

print(f"[0]#011train-auc:{cv_results.iloc[-1]['train-auc-mean']}")
print(f"[1]#011train-auc std:{cv_results.iloc[-1]['train-auc-std']}")
print(f"[1]#011validation-auc:{cv_results.iloc[-1]['test-auc-mean']}")
print(f"[1]#011validation-auc std:{cv_results.iloc[-1]['test-auc-std']}")

metrics_data = {
    "binary_classification_metrics": {
        "validation:auc": {
            "value": cv_results.iloc[-1]["test-auc-mean"],
            "standard_deviation": cv_results.iloc[-1]["test-auc-std"]
        },
        "train:auc": {
            "value": cv_results.iloc[-1]["train-auc-mean"],
            "standard_deviation": cv_results.iloc[-1]["train-auc-std"]
        },
    }
}

print(len(cv_results))
model = xgb.train(params=params, dtrain=dtrain, evals=[(dtrain, "train"), (dtest, "validation")],
                  num_boost_round=len(cv_results))

# Save the model to the location specified by ``model_dir``
metrics_location = "./output/metrics.json"
model_location = "./output/xgboost-model"
feature_names_location = "./output/feature_names"

with open(metrics_location, "w") as f:
    json.dump(metrics_data, f)

with open(model_location, "wb") as f:
    pickle.dump(model, f)

print(model.best_ntree_limit)

list_items = model.feature_names

result = []
for _item in sorted(list_items):
    result.append({
        "id": _item,
        "dtype": str(test[_item].dtype)
    })

print(json.dumps(result))
"""# Data Preprocessing"""
