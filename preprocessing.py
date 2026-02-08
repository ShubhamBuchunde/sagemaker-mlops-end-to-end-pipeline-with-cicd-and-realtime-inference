import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Because this is a header less csv file, specify the column name here
feature_column_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight"]

label_column = "rings"

feature_column_dtype ={
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64}

label_column_dtype = {"rings": np.float64}

def merge_two_dicts(x,y):
    z= x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(f"{base_dir}/input/abalone-dataset.csv",header=None,
        names= feature_column_names+[label_column],
        dtype = merge_two_dicts(feature_column_dtype,label_column_dtype)
                    )
    numeric_features = list(feature_column_names)
    numeric_features.remove("sex")
    
    numeric_transformer = Pipeline(steps=[("imputer",
                                           SimpleImputer(strategy="median")),
                                          ("scaler", StandardScaler())])
    categoric_cols= ["sex"]
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant",fill_value="missing")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocess = ColumnTransformer(transformers=[
    ("num",numeric_transformer,numeric_features),
    ("cat",categorical_transformer,categoric_cols)
])
    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y),1)

    X= np.concatenate((y_pre,X_pre),axis=1)

    np.random.shuffle(X)
    train,validation,test = np.split(X,[int(.7*len(X)),int(0.85*len(X))])
    
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv",header=False,index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation/validation.csv",header=False,index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv",header=False,index=False)







    
