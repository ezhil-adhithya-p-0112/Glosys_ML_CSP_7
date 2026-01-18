import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_data(path):
    return pd.read_csv(path)

def simple_impute(df):
    imputer = SimpleImputer(strategy="mean")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def knn_impute(df, k=5):
    imputer = KNNImputer(n_neighbors=k)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def iterative_impute(df):
    imputer = IterativeImputer(random_state=42)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
