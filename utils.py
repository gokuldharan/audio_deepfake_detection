from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def accuracies(Y, pred):
    accuracy = accuracy_score(Y, pred)
    f1 = f1_score(Y, pred)
    precision = precision_score(Y, pred)
    recall = recall_score(Y, pred)
    return accuracy, f1, precision, recall
   