import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class datapreprocessor:
    def __init__(self):
        pass
    def compare_hex_strings(self, item):
        hex_int_values = [int(element, 16) for element in item if isinstance(element, str)]
        sorted_hex_values = sorted(hex_int_values)
        sorted_hex_strings = [hex(value).strip("0x") for value in sorted_hex_values]
        return sorted_hex_strings

    def iter_to_Neglabel(self, ls):
        return 0

    def iter_to_Poslabel(self, ls):
        return 1

    def transfor_eval(self, ls):
        if isinstance(ls, str):
            return eval(ls)
        elif isinstance(ls, list):
            return ls
        elif isinstance(ls, int):
            return ls

    def calculate_entropy(self, obj):
        column_data = [obj]
        entropies = []
        for i, row in enumerate(column_data):
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

    def preprocess_samples(self, test_df, test_owner, pos_samples=True):
        if not pos_samples:
            samples = test_df[test_df["owner"].isin(test_owner)].reset_index(drop=True)
            #samples.rename(columns={"owner": "owner", "sha": "commits"}, inplace=True)
            samples["commits"] = samples["commits"].map(self.transfor_eval)
            samples["label"] = samples["commits"].map(self.iter_to_Neglabel)
            samples["merged_PR"] = pd.Series([-1]*samples.shape[0])
            return samples
        else:
            test_df["commits"] = test_df["commits"].map(self.transfor_eval)
            test_df["label"] = test_df["commits"].map(self.iter_to_Poslabel)
            return test_df

    def expand_comRE_TO_TestD(self, com_re, len_x, len_):
        result = [0] * len_x
        indices = [index for index, element in enumerate(com_re) if element == 1]
        for index in indices:
            result[index:index + len_] = [1] * len_
        return result

    def cm_combinations(self,tuple_rows, coco_len):
        len_ = coco_len-1
        ls = []
        index_ = len_
        while True:
            i = index_ - len_
            j = index_ + len_ + 2
            tmp_rows = tuple_rows.loc[i:j]
            ls.extend(list(itertools.combinations(tmp_rows, coco_len)))
            if j>len(tuple_rows):
                break
            index_+=1
        com_ls = pd.Series(ls)
        com_ls = com_ls.drop_duplicates().reset_index(drop=True)
        return com_ls

    def findNext_coco(self, x_testD, coco_len):
        coco = x_testD.loc[x_testD["predict_flag"].isin([0])]
        index_coco = coco.reset_index().drop("predict_flag", axis=1)
        tuple_rows = index_coco.apply(lambda row: {"index_": row[0], "commits": row[1]}, axis=1)
        if len(tuple_rows) == 0:
            return []
        combinations = self.cm_combinations(tuple_rows, coco_len)
        com_ls = []
        for com in combinations:
            tmp = pd.DataFrame(com)
            first_index_element = tmp.index_.values[0]
            last_index_element = tmp.index_.values[-1]
            max_index_between_elements = last_index_element - first_index_element
            if max_index_between_elements < coco_len + 1:
                com_ls.append(tmp["commits"].tolist())
        return com_ls
