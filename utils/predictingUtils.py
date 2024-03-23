import glob
import random
import math
from functools import reduce

import networkx as nx
import pandas as pd
import numpy as np
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
    random_n = len(data)
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

def calculate_and_print_metrics(testpr_IoU_all, conf_matrix_per, i=-1, random_flag = False):
    if len(testpr_IoU_all)==0:
        return  {
            "mean_i_IoU_ground_truth_true_avg": 0,
            "mean_i_IoU_predicted_truth_true_avg": 0,
            "mean_i_IoU_false_avg": 0,
            "mean_i_IoU_ground_truth_filesBased_true_avg": 0,
            "mean_i_IoU_predicted_truth_filesBased_true_avg": 0
        }

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

    return evaluate


def loads_fileLS(con):
    serialized_bytes = eval(con)
    loaded_data = pickle.loads(serialized_bytes)
    return loaded_data

def get_unique_elements_with_labels(dataframe):

    all_lists = reduce(lambda x, y: x + y, dataframe)
    all_lists_ = reduce(lambda x, y: x + y, all_lists)

    union_set = set(all_lists_)
    unique_elements = {f'{element}': idx for idx, element in enumerate(union_set)}
    return unique_elements

def get_unique_elements_from_pickle(repo_name):
    fileNameLS = f"../data/Test_auxiliary_information/FilenameLS/{repo_name}_*.pickle"
    fileNamePath = glob.glob(fileNameLS)
    if fileNamePath!=None:
        with open(fileNamePath[0], 'rb') as file:
            loaded_dict = pickle.load(file)
        return loaded_dict
    return None


def set_unique_for_element(ls):
    ls_ = []
    for tmp in ls:
        t = []
        for l in tmp:
            if l in unique_elements:
                t.append(unique_elements[l])
        ls_.append(t)
    return ls_

def transform_filename_To_Node(df_file,repo_name):
    df_file = df_file.map(loads_fileLS)
    global unique_elements
    unique_elements = get_unique_elements_from_pickle(repo_name)
    df_file_ = df_file.apply(set_unique_for_element)
    return df_file_


def create_fully_connected_directed_graph(numSet):
    if isinstance(numSet[0], (int, str, float)):
        dimension = 1
    else:
        dimension =  2

    if dimension==2:
        flat_list = [item for sublist in numSet for item in sublist]
    else:
        flat_list = set([item for item in numSet])

    edges = []

    for start in flat_list:
        tmp = []
        for end in flat_list:
                tmp.append((start, end, 1))
        edges.append(tmp)

    G = nx.DiGraph()
    for i,num in enumerate(flat_list):
        G.add_node(num)
        G.add_weighted_edges_from(edges[i])
    return G

def merged_whole_graph(df):

    G3 = nx.DiGraph()

    for i,G1 in enumerate(df.values):

        for edge in G1.edges():
            A = edge[0]
            B = edge[1]
            if G3.has_edge(A,B):
                G1[A][B]['weight'] = G3[A][B]['weight'] + 1
                G1[B][A]['weight'] = G3[B][A]['weight'] + 1
        G3 = nx.compose(G3,G1)
    return G3

