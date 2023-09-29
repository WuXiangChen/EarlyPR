
from statistic_PR_ForREPO import PR_related_mian
import pandas as pd
import os
import glob
import numpy as np
from utils import generate_adjacent_combinations,find_element,calculate_entropy
from generate_TestData.MongoDB_Related.utils import recursive_replace
from querySets_GenerateTestCommits import query_pPR
from CONSTANT import database
import joblib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score


def selectForksFromCmRepository():
    owners = ['VSChina',  'alikins']
    return owners

def getCommitsFromPRRepository(owners):
    PR_related_mian(owners)

def createLabelsForCommits():
    root_path = "Test_PR_Data_Related/_1_Commits_pSpecificFork/*.csv"
    dir_ls = glob.glob(root_path)
    result = []
    for path in dir_ls[0:]:
        print(path)
        df = pd.read_csv(path)
        pr_commitSHA = set(df["pr_commitSHA"].map(eval)[0])
        commitsAll = set(df["commitsAll"].map(eval)[0])
        del df
        diff_commits = commitsAll - pr_commitSHA
        positive_Cm = np.ones(shape=(len(pr_commitSHA),))
        negative_Cm = np.zeros(shape=(len(diff_commits),))
        df_P = pd.DataFrame([pr_commitSHA,positive_Cm]).T
        df_N = pd.DataFrame([diff_commits,negative_Cm]).T
        df_data = pd.concat([df_P,df_N],axis=0).reset_index(drop=True)
        df_data.columns = ["testD","label"]
        accuracy, recall, precision, specificity, FPR = performPredictProcess(df_data,path)
        result.append([accuracy, recall, precision, specificity, FPR])
    return result

def expand_comRE_TO_TestD(com_re,len_x,len_):
    result = [0]*len_x
    indices = [index for index, element in enumerate(com_re) if element == 1]
    for index in indices:
        result[index:index + len_] = [1] * len_
    return result

def performPredictProcess(df_data,file_):
    y_test = df_data["label"]
    y_test_copy = y_test.copy()
    model_rootPH = '../data/prRelated_data/prRelated_data_Model/'
    x_testD = df_data["testD"]
    query = query_pPR["_2_statistics_pNCM"]
    len_ = 0
    csv_values = file_.split("_pulls")[0]
    repo_owner = {"repo":csv_values.split("-")[0].split("\\")[1],
                 "owner":csv_values.split("-")[1]}
    model_filename = model_rootPH + repo_owner["repo"] + '_decision_tree_model.pkl'
    print(model_filename)
    loaded_model = joblib.load(filename=model_filename)
    j = 0
    y_pred_final = pd.DataFrame()
    while x_testD.notna:
        j += 1
        len_+=1
        com_re = []
        com = list(x_testD[0:len_])
        true_removed = []
        i = 0
        while i<len(x_testD):
            statistcsCom = SearchStaForCommitCom(com,query,repo_owner)
            if len(statistcsCom)!=0:
                obj = statistcsCom[0]
                tmp_testD = np.asarray([obj['CommitsNum'],obj['PR_FilesChangedNum'],obj['Churn_Num'],
                             obj['diff_FilesChangedRate'],obj['TS_entropy'],obj['TS_INTwoCommits']]).reshape((1,-1))
                tmp_ypred = loaded_model.predict(tmp_testD)
                com_re.extend(tmp_ypred)
            else:
                print("查询结果为空"+str(i))
                tmp_ypred = 0
                com_re.extend([tmp_ypred])
            if tmp_ypred==1:
                true_removed.extend([i for i in range(i,i+len_)])
                i += len_
                com = list(x_testD[i:i+len_])

            else:
                i += 1
                if i + len_>len(x_testD):
                    break
                com = list(x_testD[i:i + len_])
        if j>1:
            com_re = expand_comRE_TO_TestD(com_re,len(x_testD),len_)

        if len_+1>len(x_testD) or j>4:
            index_com = x_testD.index
            df_com = pd.DataFrame(com_re,index=index_com)
            y_pred_final = pd.concat([y_pred_final, df_com])
            y_pred_final = y_pred_final.sort_index()
            break

        x_testD_indices = x_testD.index[true_removed]
        x_testD = x_testD.drop(x_testD_indices)

        tmp_ones = pd.DataFrame(np.ones(shape=(len(true_removed),)),index=x_testD_indices)
        y_pred_final = pd.concat([y_pred_final, tmp_ones])

        y_indices = y_test_copy.index[true_removed]
        y_test_copy = y_test_copy.drop(y_indices)

    print("____final____")
    return  calculateTop1Accuracy(y_pred_final.iloc[:,0].tolist(), y_test.tolist())


def SearchStaForCommitCom(test_com,query,repo_owner,suffix="commits"):
    coll_names = database.list_collection_names()
    repo_name = repo_owner["repo"]
    owner_name = repo_owner["owner"]
    repo_name_INmongo = find_element(coll_names, repo_name, C=suffix)
    referenced_name = find_element(coll_names, repo_name, C="commits")

    coll = database[repo_name_INmongo]

    replacement = {"owner": owner_name, "repo": repo_name, "referenced_name": referenced_name}
    repla_query = recursive_replace(query, replacement)
    repla_query[1]["$match"]["sha"]["$in"] = test_com
    # print(repla_query)
    try:
        query_result = list(coll.aggregate(repla_query, allowDiskUse=True))
    except Exception as e:
        print(e)
        return  []

    if len(query_result)>0:
        obj = query_result[0]
        commits_Time = obj.get('Commtis_Time')
        if commits_Time is not None:
            entropy = calculate_entropy(commits_Time)
            del obj['Commtis_Time']
            obj["TS_entropy"] = entropy[0]

    else:
        #print(repla_query)
        print("query_result"+str(test_com))
        return []


    return [obj]

# 5.1 top 1 accuracy
# 5.2 top 5 accuracy
def calculateTop1Accuracy(y_pred,y_test):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    specificity = TN / (TN + FP)
    FPR = FP / (FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, recall, precision, specificity, FPR


def calculateTop5Accuracy():
    pass

if __name__ == '__main__':
    #owners = selectForksFromCmRepository()
    #PR_related_mian(owners)
    result = createLabelsForCommits()
    print(result)
