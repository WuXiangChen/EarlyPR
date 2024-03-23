
import pickle

def recursive_replace(query, replacement):
    if isinstance(query, list):
        replaced_query = []
        for item in query:
            replaced_item = recursive_replace(item, replacement)
            replaced_query.append(replaced_item)
        return replaced_query
    elif isinstance(query, dict):
        replaced_query = {}
        for key, value in query.items():
            replaced_value = recursive_replace(value, replacement)
            replaced_query[key] = replaced_value
        return replaced_query
    elif isinstance(query, str):
        if query=="%owner":
            return query.replace("%owner", replacement["owner"])
        if query=="%repo":
            return query.replace("%repo", replacement["repo"])
        if query=="%referenced_name":
            return query.replace("%referenced_name", replacement["referenced_name"])
        if query=="%owner_specific":
            return query.replace("%owner_specific", replacement["owner_specific"])
        return query
    else:
        return query

import pandas as pd
def save_commits_to_pickle(repo_name, owner_name, query_result, save_folder_name, save_type):
    output_file_path = f"{repo_name}-{owner_name}_pulls." + save_type # 以仓库名作为文件名，使用pickle扩展名
    save_file_path = save_folder_name + "/" + output_file_path
    if save_type=="pickle":
        with open(save_file_path, "wb") as output_file:
            pickle.dump(query_result, output_file)
    elif save_type=="csv":
        df_q = pd.DataFrame(query_result)
        df_q.to_csv(save_file_path,index=False)


import os
def merge_csv_files(folder_path, output_file_path):

    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    merged_data = pd.DataFrame()

    tmp_columns = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        if "_id" in data.columns:
            data = data.drop(columns="_id",axis=1)

        if "pr_commits" in data.columns:
            data = data.drop(columns="pr_commits", axis=1)

        merged_data = pd.concat([merged_data,data],axis=1, ignore_index=False)
        tmp_columns.extend([str(file).split("_pulls.csv")[0] + "_" + cn for cn in data.columns])


    merged_data.columns = tmp_columns
    merged_data.to_csv(output_file_path, index=False)

import os
from pathlib import Path
def save_results(df, repo_name, type_):

    if df is None:
        return
    else:
        output_file_path = Path("save_runningResutls/" + type_+"/"+repo_name +".csv")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file_path, index=False)