import matplotlib.pyplot as plt
import seaborn as sns

def plot_univariate(df, col):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(df[col], ax=ax[0], kde=True)
    sns.boxplot(x=df[col], ax=ax[1])
    sns.barplot(x=df[col].value_counts().index,
                y=df[col].value_counts().values, ax=ax[2])

    plt.tight_layout()
    plt.show()
