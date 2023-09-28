'''
    本节的目标是建立pr_related的评估过程
'''
'''导包区'''
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


# 1. 进行筛选，确定待选的fork名，通过Cm仓库其中所有的commits全部拿下来，以sha作为存储标志
def selectForksFromCmRepository():
    owners = ['VSChina',  'alikins']
    return owners

# 2. 通过PR仓库将指定fork名下的所有commits全部拿下来，以sha作为存储标志
def getCommitsFromPRRepository(owners):
    PR_related_mian(owners)

# 3. 利用以上信息做标签 在PR下的commit集合全部置1，做正样本；反之，置0做负样本
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
        # 至此建立完了正负样本
        accuracy, recall, precision, specificity, FPR = performPredictProcess(df_data,path)
        result.append([accuracy, recall, precision, specificity, FPR])
    return result

# 4. 在进行预测的时候，建立搜索过程
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
        # 接下来就是挨个进行查询与预测工作了
        com = list(x_testD[0:len_])
        true_removed = []
        i = 0
        while i<len(x_testD):
            statistcsCom = SearchStaForCommitCom(com,query,repo_owner)
            if len(statistcsCom)!=0:
                # 加载模型进行判断
                # CommitsNum	PR_FilesChangedNum	Churn_Num	diff_FilesChangedRate	TS_entropy	TS_INTwoCommits	label
                obj = statistcsCom[0]
                tmp_testD = np.asarray([obj['CommitsNum'],obj['PR_FilesChangedNum'],obj['Churn_Num'],
                             obj['diff_FilesChangedRate'],obj['TS_entropy'],obj['TS_INTwoCommits']]).reshape((1,-1))
                tmp_ypred = loaded_model.predict(tmp_testD)
                com_re.extend(tmp_ypred)
            else:
                print("查询结果为空"+str(i))
                tmp_ypred = 0
                com_re.extend([tmp_ypred])
            # 当一次判断中tmp_ypred为1时，去除该组合对应在x_testD中的值，并且修正com的指向
            if tmp_ypred==1:
                true_removed.extend([i for i in range(i,i+len_)])
                i += len_
                com = list(x_testD[i:i+len_])

            else:
                i += 1
                if i + len_>len(x_testD):
                    break
                com = list(x_testD[i:i + len_])
        # 跑通一轮 进行简单的预测 以及 遍历剩下的集合
        # 这里封装着所有的预测结果, 但是还是要处理一下的
        # 由com_re的结果拓展到全结果上
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
        # print("*************************")
        # print("y_pred_final_的累次变化")
        # print(y_pred_final.value_counts())
        # print("*************************")
        # print(j)

    print("____final____")
    return  calculateTop1Accuracy(y_pred_final.iloc[:,0].tolist(), y_test.tolist())


# 4.1 首先对其中每一个commit进行搜索，对搜索结果进行无放回的采样，
# 4.2 接着，对剩下的commits进行按序组合，重复以上工作
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
            # 这边需要根据时间列表计算熵值
            entropy = calculate_entropy(commits_Time)
            del obj['Commtis_Time']
            obj["TS_entropy"] = entropy[0]

    else:
        #print(repla_query)
        print("query_result_查询错误"+str(test_com))
        return []


    return [obj]


# 5. 设计评估结果：
# 5.1 top 1 accuracy
# 5.2 top 5 accuracy
def calculateTop1Accuracy(y_pred,y_test):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
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

    print("特异度:", specificity)
    print("假阳性率:", FPR)
    print("精确度:", precision)
    print("召回率:", recall)
    print("Accuracy:", accuracy)
    print("===================================================")

    return accuracy, recall, precision, specificity, FPR


def calculateTop5Accuracy():
    pass

if __name__ == '__main__':
    #owners = selectForksFromCmRepository()
    #PR_related_mian(owners)
    result = createLabelsForCommits()
    print(result)