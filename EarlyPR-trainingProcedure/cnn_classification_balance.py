import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
import os

def generate_small_samples(X, y, small_sample_threshold):
    # 根据设定的阈值将数据集拆分为小样本量和大样本量
    small_X = X[y == 1]
    small_y = y[y == 1]
    large_X = X[y == 0]
    large_y = y[y == 0]

    # 如果小样本量数量小于阈值，将所有样本作为小样本量
    if len(small_X) < small_sample_threshold:
        small_X = X
        small_y = y
        large_X = np.array([])
        large_y = np.array([])

    return small_X, small_y, large_X, large_y

def evaluate_model(data_file):
    # 加载数据集
    data = pd.read_csv(data_file)

    # 分割特征和标签
    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # 生成小样本量
    small_X, small_y, large_X, large_y = generate_small_samples(X, y, 15000)

    # 将小样本量作为训练集的一部分
    X_train = small_X
    y_train = small_y

    # 将大样本量与小样本量合并构成训练集
    if len(large_X) > 0:
        X_train = np.concatenate((X_train, large_X))
        y_train = np.concatenate((y_train, large_y))

    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # 创建LGBMClassifier对象
    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbosity=-1)

    # 训练模型
    model.fit(X_train, y_train)

    # 保存模型
    model_name = data_file.split("\\")[-1].split(".csv")[0]
    model_filename = saved_model_path + model_name + '_lgb_model.pth'
    model.booster_.save_model(model_filename)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # 计算特异度
    specificity = TN / (TN + FP)

    # 计算假阳性率
    FPR = FP / (FP + TN)

    # 计算精确度
    precision = TP / (TP + FP)

    # 计算召回率
    recall = TP / (TP + FN)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 返回评估指标结果
    return [accuracy, recall, precision, specificity, FPR]


if __name__ == '__main__':
    # 使用方法示例
    root_path = 'data/merged_pr_related/'
    saved_model_path = root_path + "merged_related_Model/"

    target_column = 'target'
    folder_path = glob.glob(root_path + "*.csv")
    for file in folder_path:
        accuracy, recall, precision, specificity, FPR = evaluate_model(file)
        print(file.split(".")[0])
        print("特异度:", specificity)
        print("假阳性率:", FPR)
        print("精确度:", precision)
        print("召回率:", recall)
        print("Accuracy:", accuracy)
        print()
