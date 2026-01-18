from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    RobustScaler, MaxAbsScaler,
    PowerTransformer
)
import numpy as np
import pandas as pd

def apply_scalers(df):
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler()
    }

    return {name: pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            for name, scaler in scalers.items()}

def fix_skewness(df):
    df_log = np.log1p(df)
    pt = PowerTransformer(method="yeo-johnson")
    df_yeo = pd.DataFrame(pt.fit_transform(df), columns=df.columns)
    return df_log, df_yeo
