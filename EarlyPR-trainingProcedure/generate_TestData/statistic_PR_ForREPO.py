# 本节用于根据确定repo生成对应的数据集
'''导包区'''
from generate_TestData.MongoDB_Related.MongoDB_utils import mongo_conn
from querySets_GenerateTestCommits import query_pPR_
import pandas as pd
from generate_TestData.MongoDB_Related.utils import recursive_replace, save_commits_to_pickle
import os

def query_repo_names(csv_file_path):
    csv_values = pd.read_csv(csv_file_path)
    return csv_values

def query_MongDB(owners,i, database,query,csv_values,save_folder_name, save_type):
    # 根据CSV列的值执行查询操作，并将结果保存在对应的文件中
    coll_names = database.list_collection_names()
    for repo_name,owner_name in csv_values.values[i:i+1]:
        for owner_specific in owners:
            repo_name_INmongo = find_element(coll_names,owner_name)
            referenced_name = find_element(coll_names,owner_name,C="commits")
            if repo_name_INmongo==None:
                continue
            coll = database[repo_name_INmongo]
            replacement={"owner":owner_name,"repo":repo_name,"referenced_name":referenced_name,
                         "owner_specific":owner_specific}
            repla_query = recursive_replace(query,replacement)
            print(repla_query)
            print(owner_specific)
            try:
                query_result = list(coll.aggregate(repla_query,allowDiskUse = True))
            except Exception as e:
                print(e)
                continue
            if len(query_result)!=0:
                print(len(query_result[0]['pr_commitSHA']))
                print(len(query_result[0]['commitsAll']))
                print()
                save_commits_to_pickle(repo_name, owner_specific,query_result,save_folder_name, save_type)

def find_element(A, B, C="pull_requests"):
    for element in A:
        if element.startswith(B) and element.endswith(C):
            return element
    return None

def PR_related_mian(owners,i):
    # CSV文件路径和列名
    csv_file_path = "../decision_Sequential_tree/repo_data/repo_owner_name.csv"
    database = mongo_conn()
    csv_values = query_repo_names(csv_file_path)
    save_folder_name, save_type = "Test_PR_Data_Related","csv"
    for query_name in list(query_pPR_.keys()):
        save_fold_path = os.path.join(save_folder_name,str(query_name))
        query_content = query_pPR_[query_name]
        if not os.path.exists(save_fold_path):
            os.mkdir(save_fold_path)
        query_MongDB(owners,i, database, query_content , csv_values, save_fold_path, save_type)

# owner_repo = []
# owner_shals = "owner_shals.csv"
# shadf = pd.read_csv(owner_shals,header=None)
# for sha in shadf.values:
#     owners = sha.tolist()
#     PR_related_mian(owners,0)
#     print(str(0)+"=============")