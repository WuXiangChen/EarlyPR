import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter

class datapreprocessor:
    def __init__(self):
        pass
    def compare_hex_strings(self, item):
        hex_int_values = [int(element, 16) for element in item if isinstance(element, str)]
        # 按从小到大的顺序排序并返回列表
        sorted_hex_values = sorted(hex_int_values)
        # 将整数列表转换为 16 进制字符串列表
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
            # 解析列表中的字典数据
            timestamps = row
            # 计算时间间隔
            intervals = np.abs(np.diff([pd.to_datetime(timestamp) for timestamp in timestamps]))
            intervals = intervals[intervals != pd.Timedelta(0)]
            if len(intervals) == 0:
                entropies.append(0)
                continue

            sum_intervals = np.sum(intervals)
            if sum_intervals == pd.Timedelta(0):
                entropy = 0
            else:
                # 计算时间间隔的熵值
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
            # 约束有效范围
            i = index_ - len_
            j = index_ + len_ + 2
            if j - 7 > i:
                j = i + 7
            tmp_rows = tuple_rows.loc[i:j]
            ls.extend(list(itertools.combinations(tmp_rows, coco_len)))
            if j>len(tuple_rows):
                break
            index_ += 1
        com_ls = pd.Series(ls)
        com_ls = com_ls.drop_duplicates().reset_index(drop=True)
        return com_ls

    def find_max_duplicate_count(self,filesLS):
        # 使用Counter来计算元素的重复次数
        element_count = Counter(filesLS)
        # 找到最大的重复次数
        max_count = np.max(list(element_count.values()))
        # 返回最大的重复次数
        if max_count>3:
            count_shredhold = 0
            return count_shredhold

        count_shredhold = 0.3 * 1/max_count
        return count_shredhold

    def find_max_ProFileCom(self,filesLS,train_filesLS):
        set1 = set(filesLS)
        js = []
        for coco_fileLS in train_filesLS:
            set2 = set(coco_fileLS)
            # 计算两个集合的交集
            intersection = len(set1.intersection(set2))
            # 计算两个集合的并集
            union = len(set1.union(set2))
            # 计算交并比
            js.append(intersection / union)

        if np.max(js) < 0.05:
            return 0

        return 0.3 * np.max(js)


    # 这里要将内容空间添加到查询coco的过程中，包括三个部分的搜索
    # 1. PR中最大重复文件数量的限制，这里是单纯的参数限制
    # 2. 文件组合构成PR的概率, 这里需要用到所构建的图结构进行设计
    # 3. 基于度计算构成PR的概率，这里需要用到所构建的带权无向图进行设计
    def findNext_coco(self, x_testD, coco_len):
        coco = x_testD.loc[x_testD["predict_flag"].isin([0])]
        if coco.shape[0] < coco_len:
            return []
        index_coco = coco.reset_index().drop("predict_flag", axis=1)
        tuple_rows = index_coco.apply(lambda row: {"index_": row[0], "commits": row[1]}, axis=1)
        if len(tuple_rows) == 0:
            return []
        combinations = self.cm_combinations(tuple_rows, coco_len)
        com_ls = []
        for com in combinations:
            tmp = pd.DataFrame(com)
            first_index_element = tmp["index_"].values[0]
            last_index_element = tmp["index_"].values[-1]
            max_index_between_elements = last_index_element - first_index_element
            # 最大跨越数+2
            if max_index_between_elements < coco_len + 1:
                com_ls.append(tmp["commits"].tolist())
        return com_ls


    def preprocess_cocofileLS(self, coco, com_ls):
        coco_fileLS = {"coco": [], "filesLS": []}
        for com in com_ls:
            com_fileLS = coco[coco["commits"].isin(com)]["fileLS"]
            flat_fileLS = [item for sublist in com_fileLS for item in sublist]
            coco_fileLS["coco"].append(com)
            coco_fileLS["filesLS"].append(flat_fileLS)
        df_coco_fileLS = pd.DataFrame(coco_fileLS)
        return df_coco_fileLS

'''
    combinations = list(itertools.combinations(tuple_rows, coco_len))
    com_ls = []
    # 现在这里应该写对这些组合施加条件
    # 1。所有长度为coco_len的com，并对其施加条件为组内所有元素之间的index跨度不得大于coco_len + 1
    for com in combinations:
        tmp = pd.DataFrame(com)
        first_index_element = tmp.index_.values[0]
        last_index_element = tmp.index_.values[-1]
        max_index_between_elements = last_index_element - first_index_element
        if max_index_between_elements < coco_len + 1:
            com_ls.append(tmp["commits"].tolist())
    print(len(com_ls))
    print("=========")
'''