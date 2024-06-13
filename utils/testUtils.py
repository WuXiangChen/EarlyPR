from pprint import pprint

import torch


def iter_to_ls(ls):
    tmp = []
    for l in ls:
        tmp.extend(l)
    return tmp

def transfor_eval(ls):
    if isinstance(ls,str):
       return eval(ls)
    elif isinstance(ls,list):
        return ls
def count_unique_elements(input_list):
    unique_elements = set(input_list)
    return len(unique_elements)

def filtered_balanced_data(owner_test_Commits):
    owner_test_Commits["data_type_len"] = owner_test_Commits["label"].map(count_unique_elements)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_type_len"] == 2]
    owner_test_Commits = owner_test_Commits.drop("data_type_len", axis=1)
    owner_test_Commits.reset_index(drop=True, inplace=True)
    return owner_test_Commits

def generate_owner_commits(merged_data,filtered_balanced_flag):
    owner_test_Commits = merged_data[["owner", "commits","label","merged_PR"]]
    owner_test_Commits.loc[:,"commits"] = owner_test_Commits["commits"].map(transfor_eval)
    agg_functions = {
        "commits": list,
        "label": list,
        "merged_PR": list,
    }
    owner_test_Commits = owner_test_Commits[["owner","commits","label","merged_PR"]].groupby("owner").agg(agg_functions).reset_index()
    if filtered_balanced_flag:
        owner_test_Commits = filtered_balanced_data(owner_test_Commits)
    owner_test_Commits["true_flag"] = owner_test_Commits["commits"]
    owner_test_Commits["commits"] = owner_test_Commits["commits"].map(iter_to_ls)
    owner_test_Commits["data_len"] = owner_test_Commits["commits"].map(len)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_len"] <= 120]
    return owner_test_Commits

def generate_owner_test_commits(owner_test_Commits):
    owner_test_Commits["data_len"] = owner_test_Commits["commits"].map(transfor_eval).map(len)
    owner_test_Commits = owner_test_Commits[owner_test_Commits["data_len"] <= 6]
    owner_test_Commits["commits"] = owner_test_Commits["commits"].map(transfor_eval)
    owner_test_Commits.drop("data_len", axis=1, inplace=True)
    owner_test_Commits.reset_index(drop=True, inplace=True)
    return owner_test_Commits

import random
import pandas as pd

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
        if query == "%owner":
            return query.replace("%owner", replacement["owner"])
        if query == "%repo":
            return query.replace("%repo", replacement["repo"])
        if query == "%referenced_name":
            return query.replace("%referenced_name", replacement["referenced_name"])
        if query == "%owner_specific":
            return query.replace("%owner_specific", replacement["owner_specific"])
        return query
    else:
        return query

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

def find_element(A, B, C="pull_requests"):
    for element in A:
        if B in element  and C in element:
            return element
    return None

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

def calculate_metrics(conf_matrix_per, i=-1, random_flag = False):
    print(f"===========Sequential Decision Tree第{i}次的IoU_均值===========")

def calculate_and_print_metrics(testpr_IoU_all, conf_matrix_per, i=-1, random_flag = False):
    if len(testpr_IoU_all)==0:
        return  {
        "IoU":0,
        "IoU_false_avg":0,
        "IoU_filesBased":0,
    }

    df_i_IoU = pd.DataFrame(testpr_IoU_all)
    mean_IoU_avg = df_i_IoU['IoU'].mean()
    mean_IoU_false_avg = df_i_IoU['IoU_false_avg'].mean()
    mean_IoU_filesBased_avg = df_i_IoU['IoU_filesBased'].mean()

    evaluate = {
        "IoU": mean_IoU_avg,
        "IoU_false_avg": mean_IoU_false_avg,
        "IoU_filesBased": mean_IoU_filesBased_avg,
    }
    pprint(evaluate)


    return evaluate

def compare_models(model1, model2, rtol=1e-05, atol=1e-08):
    params1 = model1.state_dict()
    params2 = model2.state_dict()

    for key in params1:
        try:
            torch.testing.assert_allclose(params1[key], params2[key], rtol=rtol, atol=atol)
        except AssertionError:
            print(f"Parameter '{key}' does not match.")
            return None

    print("All parameters match.")
