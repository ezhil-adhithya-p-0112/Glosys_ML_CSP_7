import pandas as pd
from scipy.stats import skew

def symmetry_report(df):
    report = []
    for col in df.select_dtypes(include='number').columns:
        mean = df[col].mean()
        median = df[col].median()
        skewness = skew(df[col].dropna())

        if abs(skewness) < 0.5:
            nature = "Symmetric"
        elif skewness > 0:
            nature = "Right-Skewed"
        else:
            nature = "Left-Skewed"

        report.append([col, mean, median, skewness, nature])

    return pd.DataFrame(
        report,
        columns=["Feature", "Mean", "Median", "Skewness", "Distribution"]
    )
