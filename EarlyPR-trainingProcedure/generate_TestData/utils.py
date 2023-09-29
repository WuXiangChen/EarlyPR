import pandas as pd
import numpy as np

def generate_adjacent_combinations(lst, n):
    combinations = []
    for i in range(len(lst) - n + 1):
        combinations.append(list(lst[i:i+n]))
    return combinations

def find_element(A, B, C="pull_requests"):
    for element in A:
        if B in element and C in element:
            return element
    return None


def query_repo_names(csv_file_path):
    csv_values = pd.read_csv(csv_file_path)
    return csv_values

def calculate_entropy(obj):
    column_data = [obj]
    entropies = []

    for i,row in enumerate(column_data):
        timestamps = row

        intervals = np.abs(np.diff([pd.to_datetime(timestamp) for timestamp in timestamps]))
        intervals = intervals[intervals != pd.Timedelta(0)]
        if len(intervals) == 0:
            entropies.append(0)
            continue

        sum_intervals = np.sum(intervals)
        if sum_intervals == pd.Timedelta(0):
            entropy = 0
        else:
            entropy = -np.sum((intervals / sum_intervals) * np.log2(intervals / sum_intervals))
        entropies.append(entropy)

    return entropies
