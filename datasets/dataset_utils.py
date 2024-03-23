
import random
import sys

script_directory_path = "../"
sys.path.append(script_directory_path)
from utils.MongoDBUtils import  mongo_conn
from datasets.Test_auxiliary_information.querySets_GenerateTestCommits import _1_CommitsPos_pSpecificFork, \
    _1_CommitSqe_pSpecificFork,_1_CommitNeg_pSpecificFork
import pandas as pd
from utils.SaveUtils import recursive_replace


def normalize_column(df, column_name):
    column = df[column_name]
    min_value = column.min()
    max_value = column.max()

    normalized_column = (column - min_value) / (max_value - min_value)

    return normalized_column

def replace_value_recursive(data, old_value, new_value):
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if value == old_value:
                value = new_value
            new_data[key] = replace_value_recursive(value, old_value, new_value)
        return new_data
    elif isinstance(data, list):
        return [replace_value_recursive(item, old_value, new_value) for item in data]
    else:
        return data

def non_repeated_sampling(data,y_pred):
    # 确定随机抽取的数量
    y_pre_T = y_pred.loc[y_pred["predict_flag"] != 0]
    y_pre_T.drop_duplicates(inplace=True)
    random_n = len(y_pre_T)+1
    n = max(1, random_n)
    samples = []
    for i in range(n):
        if len(data) == 1:
            sample_size = 1
        elif len(data) == 0:
            break
        else:
            sample_size = min(len(data), len(data) // (n - i) + 1)
        sample = random.sample(data, sample_size)
        data = list(set(data) - set(sample))
        samples.append(sample)

    return samples


def non_repeated_sampling_(data,n):
    samples = []
    options = [1, 2]
    if n==0:
        return []
    if n==1:
        return [data[0:1]]
    elif n==2:
        return [data[0:2]]
    else:
        sample_size = random.choice(options)
        sample = random.sample(data, sample_size)
        n = n-sample_size
        data = list(set(data) - set(sample))
        samples.append(sample)
        samples.extend(non_repeated_sampling(data,n))
    return samples


def query_repo_names(csv_file_path):
    csv_values = pd.read_csv(csv_file_path)
    return csv_values

def query_MongDB_ForSpecificOwner(owners, database,query,csv_values,suffix):

    coll_names = database.list_collection_names()
    repo_name, owner_name = csv_values["repo_name"],csv_values["owner_name"]
    owners_Cms = pd.DataFrame(columns=["owner","commits"])
    for owner_specific in owners:
        repo_name_INmongo = find_element(coll_names,repo_name,C=suffix)
        referenced_name = find_element(coll_names,repo_name,C="commits")
        if repo_name_INmongo==None:
            continue
        coll = database[repo_name_INmongo]
        replacement={"owner":owner_name,"repo":repo_name,"referenced_name":referenced_name,
                     "owner_specific":owner_specific}
        repla_query = recursive_replace(query,replacement)
        try:
            query_result = list(coll.aggregate(repla_query,allowDiskUse = True))
        except Exception as e:
            print(e)
            continue
        if len(query_result)!=0:
            df_query_result = pd.json_normalize(query_result)
            owners_Cms = pd.concat([owners_Cms, df_query_result], ignore_index=True)
    if suffix == "pull_requests":
        owners_Cms["merged_PR"] = owners_Cms["merged_PR"].astype(int)

    return owners_Cms

def find_element(A, B, C="pull_requests"):
    for element in A:
        if B in element  and C in element:
            return element
    return None

def Test_Sequential_Pos_commits(owners,csv_values):
    database = mongo_conn()
    query_content = _1_CommitsPos_pSpecificFork
    suffix = "pull_requests"
    query_re = query_MongDB_ForSpecificOwner(owners, database, query_content , csv_values,suffix)

    return query_re

def Test_Sequential_All_commits(owners,csv_values):
    database = mongo_conn()
    query_content = _1_CommitSqe_pSpecificFork
    suffix = "commits"
    query_re = query_MongDB_ForSpecificOwner(owners, database, query_content , csv_values,suffix)
    return query_re

def Test_Sequential_Neg_commits(owners,csv_values):
    database = mongo_conn()
    query_content = _1_CommitNeg_pSpecificFork
    suffix = "commits"
    query_re = query_MongDB_ForSpecificOwner(owners,database, query_content, csv_values, suffix)
    return query_re


def FilterCommits(df, column, threshold):
    value_counts = df[column].value_counts()
    indices_to_remove = value_counts[value_counts > threshold].index
    df_filtered = df[~df[column].isin(indices_to_remove)]
    return df_filtered