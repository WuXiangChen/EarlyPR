import glob
import pickle
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from decision_Sequential_tree.model_defined import DecisionTree_clf,ConvolutionalNetwork,RandomForestClassifier,xgb_trainer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.impute import SimpleImputer
import os
import joblib
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
# 首先 按照group_Id进行十折划分
# 然后 判断划分后的数据是否正负样本均衡，是否比例均衡
# 最后预测的时候，进行 cherry-pick检测验证

random.seed(123)
def evaluate_model_three(y_test, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    f1_score_ = f1_score(y_test, y_pred,average='macro')
    # 返回评估指标结果
    return accuracy, recall, precision, f1_score_, np.array(cm)

def evaluate_model_two(y_test, y_pred):
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
    f1_score_ = f1_score(y_test, y_pred)
    return accuracy, recall, precision, specificity, FPR, f1_score_, np.array(cm)

def print_evaluation_(param):
    for key, value in param.items():
        print(f"{key}: {value}")

def train_and_evaluate_decision_tree_model(data_file):
    # 加载数据集
    print(data_file)
    data = pd.read_csv(data_file)
    data['mergedPR'].fillna(-1, inplace=True)
    # 删除含有缺失值的行
    data.dropna(inplace=True)
    #data.drop("#Churn", inplace=True, axis=1)
    # data['mergedPR'].fillna(-1, inplace=True)
    data = data[data["label"]==1]
    # data["pr_1Mon_files_Kth_JS"] = data["pr_1Mon_files_Kth_JS"].map(eval).map(lambda x: max(x))
    # data.reset_index(inplace=True,drop=True)

    # 分割特征和标签
    # 更换mergedPR测试
    X = data.iloc[:, 4:]
    y = data.iloc[:, 0]

    #y = pd.Series(le.fit_transform(y))
    classes = len(y.unique())

    # 计算 1 和 0 的数量
    num_ones = y[y == 1].count()
    num_zeros = y[y == 0].count()
    num_Mone = y[y == -1].count()
    print("Number of 1's:", num_ones)
    print("Number of 0's:", num_zeros)
    print("Number of -1's:", num_Mone)
    # # 处理缺失值
    # imputer = SimpleImputer(strategy='mean')
    # X = imputer.fit_transform(X)
    group_id = X["group_id"].drop_duplicates().reset_index(drop=True)
    group_id = group_id.sample(frac=1, random_state=42).reset_index(drop=True)
    len_ = int(len(group_id)/10)
    group_test_IDindex = [group_id[i*len_:(i+1)*len_] for i in range(10)]
    # 创建决策树分类器对象
    clf = RandomForestClassifier()
    pro_all = []
    i = 0
    # 将数据集拆分为训练集和测试集
    for test_groupId in group_test_IDindex:
        test_index = pd.Series(X[X["group_id"].isin(test_groupId)].index)
        train_index = pd.Series(X[~X["group_id"].isin(test_groupId)].index)

        test_x = X.loc[test_index,:].iloc[:,1:]
        test_y = y.loc[test_index]

        train_x = X.loc[train_index].iloc[:,1:]
        train_y = y.loc[train_index]

        # 使用训练集对模型进行训练
        clf.fit(train_x, train_y)
        # 在这里完成它的Sequential predict过程
        # 对测试集进行预测
        y_pred = clf.predict(test_x)
        # 计算混淆矩阵
        # print(i)
        i+=1
        y_test = test_y.to_numpy()
        pro_all.append([y_test, y_pred])

    accuracy, recall, precision, f1_score_,specificity, FPR = 0,0,0,0,0,0
    tmp_re = None
    for y_test, y_pred in pro_all:
        if classes==3:
            tmp_accuracy, tmp_recall, tmp_precision, tmp_f1_score_,cm = evaluate_model_three(y_test, y_pred)
            accuracy+=tmp_accuracy
            recall+=tmp_recall
            precision+=tmp_precision
            f1_score_+=tmp_f1_score_
            if tmp_re is None:
                tmp_re = cm
            else:
                tmp_re+=cm
        elif classes==2:
            tmp_accuracy, tmp_recall, tmp_precision, tmp_specificity, tmp_FPR,tmp_f1_score_,cm = evaluate_model_two(y_test, y_pred)
            accuracy += tmp_accuracy
            recall += tmp_recall
            if not np.isnan(tmp_precision):
                precision += tmp_precision
            specificity += tmp_specificity
            f1_score_ += tmp_f1_score_
            FPR += tmp_FPR
            if tmp_re is None:
                tmp_re = cm
            else:
                tmp_re+=cm
    if classes == 3:
        tmp = {"accuracy": format(accuracy / 10,'.4f'), "recall": format(recall / 10,'.4f'), "precision": format(precision / 10,'.4f'),
               "f1_score_": format(f1_score_ / 10,'.4f')}
        print_evaluation_(tmp)
    else:
        tmp = {"accuracy": format(accuracy / 10,'.4f'), "recall": format(recall / 10,'.4f'), "precision": format(precision / 10,'.4f'),
               "specificity":format(specificity / 10,'.4f'), "FPR":format(FPR / 10,'.4f'),"f1_score_": format(f1_score_ / 10,'.4f')}
        print_evaluation_(tmp)

    print("=======================")
    return tmp,tmp_re
    #print(tmp_re)
    # repo_name = data_file.split("\\")[-1].split(".csv")[0]
    # save_path = saved_model_path+repo_name+"_Random_Forest_model.pickle"
    # with open(save_path,"wb") as f:
    #     pickle.dump(clf, f)
    # print("======================")

if __name__ == '__main__':
    root_path = 'data//prRelated_data//'
    saved_model_path = root_path + "prRelated_data_Model/"
    #save_name = "Extend_Features_Predict_PR_Random_10_folder_ByOwner.csv"
    save_name = "CNN_Extend_Features_Known_PR_PredictMergedPR_KnownPR_10_folder_ByOwner.csv"
    pro_all = []
    folder_path = glob.glob(root_path + "*.csv")
    output = np.array([[0, 0], [0, 0]])
    for file in folder_path:
        tmp,tmp_re = train_and_evaluate_decision_tree_model(file)
        output += tmp_re
        pro_all.append(tmp)
    tmp = pd.json_normalize(pro_all)
    tmp.to_csv(save_name)
    print("最终累计的结果为:")
    print(output)

