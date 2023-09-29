import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
import os

def generate_small_samples(X, y, small_sample_threshold):
    small_X = X[y == 1]
    small_y = y[y == 1]
    large_X = X[y == 0]
    large_y = y[y == 0]
    if len(small_X) < small_sample_threshold:
        small_X = X
        small_y = y
        large_X = np.array([])
        large_y = np.array([])

    return small_X, small_y, large_X, large_y

def evaluate_model(data_file):
    data = pd.read_csv(data_file)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    small_X, small_y, large_X, large_y = generate_small_samples(X, y, 15000)

    X_train = small_X
    y_train = small_y

    if len(large_X) > 0:
        X_train = np.concatenate((X_train, large_X))
        y_train = np.concatenate((y_train, large_y))

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbosity=-1)

    model.fit(X_train, y_train)

    model_name = data_file.split("\\")[-1].split(".csv")[0]
    model_filename = saved_model_path + model_name + '_lgb_model.pth'
    model.booster_.save_model(model_filename)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    specificity = TN / (TN + FP)

    FPR = FP / (FP + TN)

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    accuracy = accuracy_score(y_test, y_pred)

    return [accuracy, recall, precision, specificity, FPR]


if __name__ == '__main__':
    root_path = 'data/merged_pr_related/'
    saved_model_path = root_path + "merged_related_Model/"

    target_column = 'target'
    folder_path = glob.glob(root_path + "*.csv")
    for file in folder_path:
        accuracy, recall, precision, specificity, FPR = evaluate_model(file)
        print(file.split(".")[0])
