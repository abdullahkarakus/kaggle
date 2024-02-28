#Data

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



def load_train_data():

    df = pd.read_csv("../input/train.csv")

    df.drop(columns = "id", inplace=True)

    str_cols = [col for col in df.columns if col != "NObeyesdad" and col != "MTRANS" and df[col].dtype == "object"]
    num_cols = [col for col in df.columns if df[col].dtype == "float64"]

    gender_cats = ["Female", "Male"]
    f_hist_cats = ["yes", "no"]
    favc_cats = ["yes", "no"]
    caec_cats = ["no", "Sometimes", "Frequently", "Always"]
    smoke_cats = ["yes", "no"]
    scc_cats = ["yes", "no"]
    calc_cats = ["no", "Sometimes", "Frequently", "Always"]

    enc_ord = OrdinalEncoder(categories = [gender_cats, f_hist_cats, favc_cats, caec_cats, smoke_cats, scc_cats, calc_cats])
    df_str = df[str_cols]
    df_str_ord = enc_ord.fit_transform(df_str)
    df_str_ord = pd.DataFrame(df_str_ord, columns=enc_ord.feature_names_in_, dtype=int)
    
    enc_ht = OneHotEncoder(sparse_output = False)
    mt = df[["MTRANS"]]
    mt = enc_ht.fit_transform(mt)
    mt = pd.DataFrame(mt, columns=enc_ht.get_feature_names_out(), dtype=int)

    df_num = df[num_cols]
    df_target = df[["NObeyesdad"]]

    df = pd.concat((df_num, df_str_ord, mt, df_target), axis=1)

    return df

def load_test_data():

    df = pd.read_csv("../input/test.csv")

    id = df.id
    df.drop(columns = "id", inplace=True)

    str_cols = [col for col in df.columns if col != "MTRANS" and df[col].dtype == "object"]
    num_cols = [col for col in df.columns if df[col].dtype == "float64"]

    gender_cats = ["Female", "Male"]
    f_hist_cats = ["yes", "no"]
    favc_cats = ["yes", "no"]
    caec_cats = ["no", "Sometimes", "Frequently", "Always"]
    smoke_cats = ["yes", "no"]
    scc_cats = ["yes", "no"]
    calc_cats = ["no", "Sometimes", "Frequently", "Always"]

    enc_ord = OrdinalEncoder(categories = [gender_cats, f_hist_cats, favc_cats, caec_cats, smoke_cats, scc_cats, calc_cats])
    df_str = df[str_cols]
    df_str_ord = enc_ord.fit_transform(df_str)
    df_str_ord = pd.DataFrame(df_str_ord, columns=enc_ord.feature_names_in_, dtype=int)
    
    enc_ht = OneHotEncoder(sparse_output = False)
    mt = df[["MTRANS"]]
    mt = enc_ht.fit_transform(mt)
    mt = pd.DataFrame(mt, columns=enc_ht.get_feature_names_out(), dtype=int)

    df_num = df[num_cols]
    df_id = pd.DataFrame(id, columns=["id"])

    df = pd.concat((df_num, df_str_ord, mt, df_id), axis=1)

    return df

def load_extra_data():

    df = pd.read_csv("../input/ObesityDataSet.csv")

    str_cols = [col for col in df.columns if col != "NObeyesdad" and col != "MTRANS" and df[col].dtype == "object"]
    num_cols = [col for col in df.columns if df[col].dtype == "float64"]

    gender_cats = ["Female", "Male"]
    f_hist_cats = ["yes", "no"]
    favc_cats = ["yes", "no"]
    caec_cats = ["no", "Sometimes", "Frequently", "Always"]
    smoke_cats = ["yes", "no"]
    scc_cats = ["yes", "no"]
    calc_cats = ["no", "Sometimes", "Frequently", "Always"]

    enc_ord = OrdinalEncoder(categories = [gender_cats, f_hist_cats, favc_cats, caec_cats, smoke_cats, scc_cats, calc_cats])
    df_str = df[str_cols]
    df_str_ord = enc_ord.fit_transform(df_str)
    df_str_ord = pd.DataFrame(df_str_ord, columns=enc_ord.feature_names_in_, dtype=int)
    
    enc_ht = OneHotEncoder(sparse_output = False)
    mt = df[["MTRANS"]]
    mt = enc_ht.fit_transform(mt)
    mt = pd.DataFrame(mt, columns=enc_ht.get_feature_names_out(), dtype=int)

    df_num = df[num_cols]
    df_target = df[["NObeyesdad"]]

    df = pd.concat((df_num, df_str_ord, mt, df_target), axis=1)

    return df
    
