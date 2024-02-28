#main


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from collections import Counter

import data
import xgb
import lgbm
import nn

if __name__ == "__main__":
    print("Loading the data...")
    df_1 = data.load_train_data()
    df_2 = data.load_extra_data()

    df = pd.concat((df_1, df_2), axis=0)
    df.reset_index(inplace=True, drop=True)

    target = df.NObeyesdad.values   
    df = df.drop(columns = "NObeyesdad")

    df_test = data.load_test_data()
    id = df_test.id
    df_test.drop(columns="id", inplace=True)

    print("Getting the best params for xgb...")
    best_params_xgb = xgb.get_best_params(df, target)

    print("Getting the best params for lgbm...")
    best_params_lgbm = lgbm.get_best_params(df, target)

    print("Predictions of xgb...")
    xgb_preds = xgb.get_predictions(df, df_test, target, best_params_xgb)
    
    print("Predictions of lgbm...")
    lgbm_preds = lgbm.get_predictions(df, df_test, target, best_params_lgbm)
    
    print("Predictions of nn...")
    nn_preds = nn.get_predictions(hidden_dim=25, num_epochs=40)

    final_preds = []

    for index in range(len(xgb_preds)):
        data = Counter([xgb_preds[index], lgbm_preds[index], nn_preds[index]])
        majority_vote = data.most_common()[0][0]
        final_preds.append(majority_vote)

    index = id.tolist()
    df_result = pd.DataFrame(final_preds, index = index, columns =["NObeyesdad"]) 
    df_result = df_result.rename_axis("id").reset_index()
    df_result.to_csv("../submission/sub.csv", index=False)

    

    

    

    
