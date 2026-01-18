from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import entropy
import numpy as np

def balance_smote(X, y):
    return SMOTE().fit_resample(X, y)

def balance_under(X, y):
    return RandomUnderSampler().fit_resample(X, y)

def calculate_entropy(y):
    values, counts = np.unique(y, return_counts=True)
    return entropy(counts)
