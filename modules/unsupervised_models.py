from sklearn.cluster import KMeans, DBSCAN, MeanShift
from minisom import MiniSom

def kmeans(X, k=3):
    return KMeans(n_clusters=k).fit_predict(X)

def dbscan(X):
    return DBSCAN(eps=0.5).fit_predict(X)

def meanshift(X):
    return MeanShift().fit_predict(X)

def som(X):
    som = MiniSom(10,10,X.shape[1])
    som.train(X,100)
    return som
