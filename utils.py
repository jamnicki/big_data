import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import mode


class KNN():
    def __init__(self, N=5):
        self.N = N

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.m, self.n = X_train.shape

    def predict(self, X_test):
        self.X_test = X_test
        self.m_test, self.n = X_test.shape

        Y_predict = np.zeros(self.m_test)
        for i in range(self.m_test):
            x = self.X_test[i]
            neighbors = np.zeros(self.N)
            neighbors = self.find_neighbors(x)
            Y_predict[i] = mode(neighbors)[0][0]

        return Y_predict

    def find_neighbors(self, x):
        euclidean_distances = np.zeros(self.m)
        for i in range(self.m):
            d = self.euclidean(x, self.X_train[i])
            euclidean_distances[i] = d

        inds = euclidean_distances.argsort()
        Y_train_sorted = self.Y_train[inds]

        return Y_train_sorted[:self.N]

    def euclidean(self, x, x_train):
        return np.sqrt(np.sum(np.square(x-x_train)))


def get_unique_chars(txt_filepath):
    with open(txt_filepath) as f:
        raw_txt = f.read()

    unique_chars = []
    for char in raw_txt:
        if char not in unique_chars:
            unique_chars.append(char)

    return sorted(unique_chars)


def extract_lines_val(txt_filepath):
    with open(txt_filepath) as f:
        lines = f.readlines()

    lines_extracted_values = []
    for line in lines:
        line_values = []
        line_split = line.split(' ')
        for string in line_split:
            try:
                value = float(string)
                line_values.append(value)
            except Exception:
                pass
        lines_extracted_values.append(line_values)

    return lines_extracted_values


def extract_labels(txt_filepath):
    with open(txt_filepath) as f:
        raw_txt = f.read()

    new_lines_free = raw_txt.replace('\n', '')
    labels = [char for char in new_lines_free]

    return labels


def get_features(txt_filepath):
    with open(txt_filepath) as f:
        lines = f.readlines()

    features = []
    for line in lines:
        new_line_free = line.replace('\n', '')
        feature = new_line_free.split(' ')[1]
        features.append(feature)

    return features


def calculate_classes_accuracy(y_test, predicted):
    cnf_matrix = confusion_matrix(y_test, predicted)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN) / (TP+FP+FN+TN)

    return ACC
