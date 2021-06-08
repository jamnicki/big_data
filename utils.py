import numpy as np
from sklearn.metrics import confusion_matrix


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
    TPR = TP / (TP+FN)
    # Specificity or true negative rate
    TNR = TN / (TN+FP) 
    # Precision or positive predictive value
    PPV = TP / (TP+FP)
    # Negative predictive value
    NPV = TN / (TN+FN)
    # Fall out or false positive rate
    FPR = FP / (FP+TN)
    # False negative rate
    FNR = FN / (TP+FN)
    # False discovery rate
    FDR = FP / (TP+FP)

    # Overall accuracy
    ACC = (TP+TN) / (TP+FP+FN+TN)
    
    return ACC
