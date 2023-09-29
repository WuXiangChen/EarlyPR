import random
import math

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

def non_repeated_sampling(data,n):
    samples = []
    for i in range(n):
        if len(data) == 1:
            sample_size = 1
        elif len(data)==0:
            break
        else:
            sample_len = len(data) // (n-i)
            if n-i > 1:
                options = list(range(sample_len , sample_len+1))
            else:
                options = list(range(len(data),len(data)+1))
            sample_size = random.choice(options)
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


import pandas as pd
import numpy as np


def calculate_and_print_metrics(testpr_IoU_all, conf_matrix_per, conf_matrix_random_per=None, i=-1):
    if len(testpr_IoU_all)==0:
        return  {
        "mean_i_IoU_ground_truth_true_avg": 0,
        "mean_i_IoU_predicted_truth_true_avg": 0,
        "mean_i_IoU_false_avg": 0,
        "mean_i_IoU_ground_truth_filesBased_true_avg": 0,
        "mean_i_IoU_predicted_truth_filesBased_true_avg": 0}

    df_i_IoU = pd.DataFrame(testpr_IoU_all)
    mean_i_IoU_ground_truth_true_avg = df_i_IoU['IoU_ground_truth_true_avg'].mean()
    mean_i_IoU_predicted_truth_true_avg = df_i_IoU['IoU_predicted_truth_true_avg'].mean()
    mean_i_IoU_false_avg = df_i_IoU['IoU_false_avg'].mean()
    mean_i_IoU_ground_truth_filesBased_true_avg = df_i_IoU['IoU_ground_truth_filesBased_true_avg'].mean()
    mean_i_IoU_predicted_truth_filesBased_true_avg = df_i_IoU['IoU_predicted_truth_filesBased_true_avg'].mean()

    evaluate = {
        "mean_i_IoU_ground_truth_true_avg": mean_i_IoU_ground_truth_true_avg,
        "mean_i_IoU_predicted_truth_true_avg": mean_i_IoU_predicted_truth_true_avg,
        "mean_i_IoU_false_avg": mean_i_IoU_false_avg,
        "mean_i_IoU_ground_truth_filesBased_true_avg": mean_i_IoU_ground_truth_filesBased_true_avg,
        "mean_i_IoU_predicted_truth_filesBased_true_avg": mean_i_IoU_predicted_truth_filesBased_true_avg
    }

    print(f"===========Sequential Decision Tree第{i}次的IoU_均值===========")
    print(f"mean_IoU_ground_truth_true_avg:", str(mean_i_IoU_ground_truth_true_avg))
    print(f"mean_IoU_predicted_truth_true_avg:", str(mean_i_IoU_predicted_truth_true_avg))
    print(f"mean_IoU_false_avg:", str(mean_i_IoU_false_avg))
    print(f"mean_IoU_ground_truth_filesBased_true_avg:", str(mean_i_IoU_ground_truth_filesBased_true_avg))
    print(f"mean_IoU_predicted_truth_filesBased_true_avg:", str(mean_i_IoU_predicted_truth_filesBased_true_avg))
    print("mergedPR Confusion Matrix:\n", conf_matrix_per)
    print("Random Cm Confusion Matrix:\n", conf_matrix_random_per)

    return evaluate

def calculate_metrics(testpr_IoU_all, conf_matrix_per, conf_matrix_random_per=None, i=-1):
    if len(testpr_IoU_all)==0:
        return  {
        "mean_i_IoU_ground_truth_true_avg": 0,
        "mean_i_IoU_predicted_truth_true_avg": 0,
        "mean_i_IoU_false_avg": 0,
        "mean_i_IoU_ground_truth_filesBased_true_avg": 0,
        "mean_i_IoU_predicted_truth_filesBased_true_avg": 0}

    df_i_IoU = pd.DataFrame(testpr_IoU_all)
    mean_i_IoU_ground_truth_true_avg = df_i_IoU['mean_i_IoU_ground_truth_true_avg'].mean()
    mean_i_IoU_predicted_truth_true_avg = df_i_IoU['mean_i_IoU_predicted_truth_true_avg'].mean()
    mean_i_IoU_false_avg = df_i_IoU['mean_i_IoU_false_avg'].mean()
    mean_i_IoU_ground_truth_filesBased_true_avg = df_i_IoU['mean_i_IoU_ground_truth_filesBased_true_avg'].mean()
    mean_i_IoU_predicted_truth_filesBased_true_avg = df_i_IoU['mean_i_IoU_predicted_truth_filesBased_true_avg'].mean()

    evaluate = {
        "mean_i_IoU_ground_truth_true_avg": mean_i_IoU_ground_truth_true_avg,
        "mean_i_IoU_predicted_truth_true_avg": mean_i_IoU_predicted_truth_true_avg,
        "mean_i_IoU_false_avg": mean_i_IoU_false_avg,
        "mean_i_IoU_ground_truth_filesBased_true_avg": mean_i_IoU_ground_truth_filesBased_true_avg,
        "mean_i_IoU_predicted_truth_filesBased_true_avg": mean_i_IoU_predicted_truth_filesBased_true_avg
    }

    print(f"===========Sequential Decision Tree第{i}次的IoU_均值===========")
    print(f"mean_IoU_ground_truth_true_avg:", str(mean_i_IoU_ground_truth_true_avg))
    print(f"mean_IoU_predicted_truth_true_avg:", str(mean_i_IoU_predicted_truth_true_avg))
    print(f"mean_IoU_false_avg:", str(mean_i_IoU_false_avg))
    print(f"mean_IoU_ground_truth_filesBased_true_avg:", str(mean_i_IoU_ground_truth_filesBased_true_avg))
    print(f"mean_IoU_predicted_truth_filesBased_true_avg:", str(mean_i_IoU_predicted_truth_filesBased_true_avg))
    print("mergedPR Confusion Matrix:\n", conf_matrix_per)
    print("Random Cm Confusion Matrix:\n", conf_matrix_random_per)
    return evaluate
