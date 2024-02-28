#lgbm

import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import data



def run():
    df = data.load_train_data()

    target = df.NObeyesdad.values
    enc_lbl = LabelEncoder()
    target = enc_lbl.fit_transform(target)

    df = df.drop(columns = "NObeyesdad")

    stf = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

    accuracy_list = []
    for train_idx, test_idx in stf.split(X=df, y=target):
        train_df = df.loc[train_idx,:]
        train_target = target[train_idx]
        test_df = df.loc[test_idx,:]
        test_target = target[test_idx]

        params = {"objective": "multiclass",
                  "boosting_type": "gbdt",
                  "learning_rate": 0.05,
                  "reg_alpha": 1, #L^1
                  "reg_lambda": 1, #L^2
                  "subsample": 0.75,
                  "colsample_bytree": 0.6,
                  "max_depth": -1,
                  "n_estimators": 200,
                  "min_child_weight": 1e-3,
                  "verbose": -1,
                  "random_state": 73
                 }
                  
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X=train_df, y=train_target)
        preds = clf.predict(test_df)
        accuracy = accuracy_score(test_target, preds)
        print(f"Accuracy: {accuracy}")
        accuracy_list.append(accuracy)

    mean_accuracy = np.mean(np.array(accuracy_list))
    print(f"Average accuracy: {mean_accuracy}")


def objective(trial, data, target):
    params = {"objective": trial.suggest_categorical("objective",["multiclass","multiclassova"]),
                  "num_class": 7,
                  "boosting_type": "gbdt", 
                  "metric": trial.suggest_categorical("metric", ["multi_logloss", "multi_error"]),
                  "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                  "reg_alpha": trial.suggest_int("reg_alpha", 0, 5), #L^1
                  "reg_lambda": trial.suggest_int("reg_lambda", 1, 5), #L^2
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                  "max_depth": trial.suggest_int("max_depth", 3, 12),
                  "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                  "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 1.0),
                  "verbose": trial.suggest_categorical("verbose", [-1]),
                  "random_state": 73
                 }
    
    model = lgb.LGBMClassifier(**params)
    cv_score = cross_val_score(model, X=data, y=target, scoring = "accuracy", cv=5)
    mean_accuracy = np.mean(cv_score)

    return mean_accuracy
    
def get_best_params(data, target):
    enc_lbl = LabelEncoder()
    target = enc_lbl.fit_transform(target)
    
    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, data, target), n_trials = 10)

    best_params = study.best_params

    return best_params
    
    
def get_predictions(train_data, test_data, train_target, params):

    enc_lbl = LabelEncoder()
    train_target = enc_lbl.fit_transform(train_target)
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X=train_data, y=train_target)
    
    preds = clf.predict(test_data)
    preds = enc_lbl.inverse_transform(preds)

    return preds
  


if __name__ == "__main__":
    run()
    
    




    