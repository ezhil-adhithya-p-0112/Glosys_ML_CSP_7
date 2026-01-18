from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

def id3_classifier():
    return DecisionTreeClassifier(criterion="entropy")

def bayes_classifier():
    return GaussianNB()

def mlp_classifier():
    return MLPClassifier(hidden_layer_sizes=(64,32),
                         max_iter=500)

def linear_regression():
    return LinearRegression()
