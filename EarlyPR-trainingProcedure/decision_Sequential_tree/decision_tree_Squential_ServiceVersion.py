import argparse
import glob
import sys
import warnings

from sklearn.impute import SimpleImputer

script_directory_path = "../"
sys.path.append(script_directory_path)
sys.path.append("DataPreproAndModelExecutor")
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from statistic_PR_ForREPO_decisionSequential_tree import Test_Sequential_Pos_commits,find_element,Test_Sequential_All_commits,\
    Test_Sequential_Neg_commits
import joblib
from DataPreproAndModelExecutor.ModelExecutor import modelexecutor
from DataPreproAndModelExecutor.datapreprocessor import datapreprocessor
from model_defined import ConvolutionalNetwork,NeuralNetwork,LSTMModel,RandomForestClassifier,DecisionTreeClassifier,LGBMClassifier
import numpy as np
from decision_Seq_utils import calculate_and_print_metrics, calculate_metrics

warnings.filterwarnings("ignore")

csv_values = pd.read_csv("repo_data/repo_owner_name.csv")
Test_ownerShas_ls = glob.glob("../data/Test_auxiliary_information/OwnerShas_for_generate_SequentialtestData/*.csv")
def transfor_eval(ls):
    if isinstance(ls,str):
       return eval(ls)
    elif isinstance(ls,list):
        return ls

def iter_to_ls(ls):
    tmp = []
    for l in ls:
        tmp.extend(l)
    return tmp

def count_unique_elements(input_list):
    unique_elements = set(input_list)
    return len(unique_elements)

def filtered_balanced_data(owner_test_Commits):
    owner_test_Commits["data_type_len"] = owner_test_Commits["label"].map(count_unique_elements)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_type_len"] == 2]
    owner_test_Commits = owner_test_Commits.drop("data_type_len", axis=1)
    owner_test_Commits.reset_index(drop=True, inplace=True)
    return owner_test_Commits

def generate_owner_test_commits(merged_data,filtered_balanced_flag):
    owner_test_Commits = merged_data[["owner", "commits","label","merged_PR"]]
    owner_test_Commits.loc[:,"commits"] = owner_test_Commits["commits"].map(transfor_eval)
    agg_functions = {
        "commits": list,
        "label": list,
        "merged_PR": list
    }
    owner_test_Commits = owner_test_Commits[["owner","commits","label","merged_PR"]].groupby("owner").agg(agg_functions).reset_index()
    if filtered_balanced_flag:
        owner_test_Commits = filtered_balanced_data(owner_test_Commits)
    owner_test_Commits["true_flag"] = owner_test_Commits["commits"]
    owner_test_Commits["commits"] = owner_test_Commits["commits"].map(iter_to_ls)
    owner_test_Commits["data_len"] = owner_test_Commits["commits"].map(len)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_len"] <= 120]
    return owner_test_Commits

def compare_hex_strings(item):
    sorted_hex_values = sorted(item)
    return sorted_hex_values

def train_and_evaluate_decision_tree_model(data_file,clf_name):
    print(data_file)
    data = pd.read_csv(data_file)[0:100]
    data['mergedPR'].fillna(-1, inplace=True)
    data.insert(1,'mergedPR',data.pop("mergedPR"))
    data.rename(columns={"sha":"commits","mergedPR":"merged_PR"},inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True,inplace=True)

    data.dropna(axis=0, inplace=True)
    owner_group_ID = data.iloc[:, 3:5]
    owner_group_ID = owner_group_ID.drop_duplicates().reset_index(drop=True)

    X_ = data[data['label'] == 1].iloc[:, 1:]
    y_ = data[data['label'] == 1].iloc[:, 1]

    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    group_id = X["group_id"].drop_duplicates().reset_index(drop=True)
    group_id = group_id.sample(frac=1, random_state=42).reset_index(drop=True)
    len_ = int(len(group_id) / 10)
    group_test_IDindex = [group_id[i * len_:i * len_ + len_] for i in range(10)]

    clf = eval(clf_name)()
    pr_all = []
    random_pr_all = []
    repo_owner = {}
    repo_owner["repo_name"] = data_file.split("//")[-1].split(".csv")[0]
    repo_owner["owner_name"] = csv_values[csv_values["repo_names"] == repo_owner["repo_name"]]["owner_names"].iloc[0]
    repo_owner["PR_start_time"] = csv_values[csv_values["repo_names"] == repo_owner["repo_name"]]["PR_start_time"].iloc[0]
    neg_OwnerShas_path = find_element(Test_ownerShas_ls, repo_owner["repo_name"], repo_owner["owner_name"])

    dp = datapreprocessor()
    modelex = modelexecutor(clf, repo_owner, clf_name)
    conf_matrix_pr_per = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_matrix_randompr_per = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_matrix_random_per = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for i, test_groupId in enumerate(group_test_IDindex):
        if i < 0:
            continue
        train_index = pd.Series(X[~X["group_id"].isin(test_groupId)].index)
        train_x = X.loc[train_index].iloc[:, 4:]
        train_y = y.loc[train_index]

        clf.fit(train_x, train_y)

        merged_train_index = pd.Series(X_[~X_["group_id"].isin(test_groupId)].index)
        mergedPR_train_x = X_.loc[merged_train_index].iloc[:, 4:]
        mergedPR_train_y = y_.loc[merged_train_index]
        modelex.mergedPR_predictor.merged_clf.fit(mergedPR_train_x,mergedPR_train_y)

        test_owner = owner_group_ID[owner_group_ID["group_id"].isin(test_groupId)].reset_index(drop=True)["owner"]

        print(test_owner)
        if  test_owner.empty:
            continue
        pos_test_df = Test_Sequential_Pos_commits(test_owner, repo_owner)
        neg_test_df = Test_Sequential_Neg_commits(test_owner, repo_owner)
        neg_test_data = dp.preprocess_samples(neg_test_df, test_owner, pos_samples=False)
        pos_test_data = dp.preprocess_samples(pos_test_df, test_owner, pos_samples=True)
        merged_data = pd.concat([pos_test_data, neg_test_data], axis=0).reset_index()

        filtered_balanced_flag = False
        testpr_IoU_all = []
        testrandompr_IoU_all = []
        filtered_test_Commits = generate_owner_test_commits(merged_data, filtered_balanced_flag)
        for index, row in filtered_test_Commits.iterrows():
            print(index)
            IoU_,randomcocoPr_IoU, predicted_conf_matrix,randomcocoPr_predicted_conf_matrix,\
            randomcoco_predicted_conf_matrix = modelex.performPredictProcess(row)
            testpr_IoU_all.append(IoU_)
            testrandompr_IoU_all.append(randomcocoPr_IoU)

            conf_matrix_pr_per = conf_matrix_pr_per + predicted_conf_matrix
            conf_matrix_randompr_per = conf_matrix_randompr_per + randomcocoPr_predicted_conf_matrix
            conf_matrix_random_per = conf_matrix_random_per + randomcoco_predicted_conf_matrix


        evaluate_pr = calculate_and_print_metrics(testpr_IoU_all,conf_matrix_pr_per,i=i)

        evaluate_randompr = calculate_and_print_metrics(testrandompr_IoU_all, conf_matrix_randompr_per,conf_matrix_random_per, i=i)

        random_pr_all.append(evaluate_randompr)
        pr_all.append(evaluate_pr)

        # joblib.dump(clf, model_filename)

    print()
    print(f"*************Sequential {data_file} ******************")
    pr_df_IoU = pd.json_normalize(pr_all)
    randompr_df_IoU = pd.json_normalize(random_pr_all)
    calculate_metrics(pr_df_IoU, conf_matrix_pr_per)
    calculate_metrics(randompr_df_IoU, conf_matrix_randompr_per, conf_matrix_random_per)

if __name__ == '__main__':
    root_path = '../data/prRelated_data//'
    saved_model_path = root_path + "prRelated_data_Model/"
    folder_path = glob.glob(root_path + "*.csv")
    parser = argparse.ArgumentParser(description='Train and evaluate decision tree model on input CSV file.')
    parser.add_argument('file', type=str, help='Path to the input CSV file.')
    parser.add_argument('model_name', type=str, help='Path to the input CSV file.')

    args = parser.parse_args()

    file_path = args.file
    clf_name = args.model_name

    train_and_evaluate_decision_tree_model(file_path,clf_name)

