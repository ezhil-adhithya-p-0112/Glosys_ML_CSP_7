from sklearn.metrics import confusion_matrix, roc_auc_score, silhouette_score

def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Accuracy": (tp+tn)/(tp+tn+fp+fn),
        "Precision": tp/(tp+fp),
        "Recall": tp/(tp+fn),
        "Specificity": tn/(tn+fp),
        "F1": 2*tp/(2*tp+fp+fn)
    }

def clustering_score(X, labels):
    return silhouette_score(X, labels)
