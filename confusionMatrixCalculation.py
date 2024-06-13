import copy
import glob
import warnings
import numpy as np
import pandas as pd
from datasets.GenerateDataSetForMPTrain import GenerateDataSetForMPTrainAndTest
from utils.testUtils import generate_owner_test_commits, calculate_and_print_metrics, compare_models, \
    generate_owner_commits
# Disable all warnings
warnings.filterwarnings("ignore")
from utils.runUtils import load_opt_sched_from_ckptTest
from datasets.GenerateDataSetForPRTrain import GenerateDataSetForPRTrainAndTest
from CONSTANT.CONSTANT import *
from CONSTANT.CONSTANT import CB_max_Output_length
from performPredict.performPredictProcess import performPredict
import torch
from glob import glob

def calculate_hittingAndmissingRate(matrix):
    predicted_true_PR = matrix[0][0] + matrix[0][1]+matrix[1][0] + matrix[1][1]
    predicted_PR = np.sum(matrix[:, 0:2])

    missing_rate = 1 - (predicted_true_PR/predicted_PR)
    return missing_rate

def calculateAccuracy_IoU(y_true_T, pre_cm,threshold=0.5):
    result = []
    pre_cm = eval(pre_cm)
    for j, cm in enumerate(y_true_T.values):
        cm = eval(cm)
        IOU = len(set(cm).intersection(set(pre_cm))) / len(set(cm).union(set(pre_cm)))
        result.append(IOU)

    re = np.asarray(result, dtype=np.float32)
    argmax = np.argmax(re)
    max_iou = re[argmax]

    if max_iou >= threshold:
        return argmax, max_iou

    return -1,-1

def fusion_test(predicted_df, true_df, threshold=0.5):
    # 补Nan
    predicted_df.fillna(0, inplace=True)
    confusion_matrix = np.zeros((3, 3))

    hitting_rank = []
    for index, row in predicted_df.iterrows():
        cm_, predicted_flag = row["predict_flag"], int(row["predicted_flag"])
        true_cm_ = true_df["cm_"]
        true_series_flag = true_df["merged_PR"]
        argmax,  max_iou = calculateAccuracy_IoU(true_cm_, cm_,threshold)
        hitting_rank.append(argmax)

        if argmax != -1:
                true_flag = int(true_series_flag[argmax])
                if true_flag == predicted_flag == 1:
                        confusion_matrix[0][0] += 1
                elif true_flag == predicted_flag == 0:
                        confusion_matrix[1][1] += 1
                elif true_flag == 1:
                    if predicted_flag == 0:
                        confusion_matrix[0][1] += 1
                elif true_flag == 0:
                    if predicted_flag == 1:
                        confusion_matrix[1][0] += 1
        else:
                if predicted_flag == 1:
                    confusion_matrix[2][0] += 1
                else:
                    confusion_matrix[2][1] += 1

    true_positive = true_df[true_df["merged_PR"] == 1]
    true_negtive = true_df[true_df["merged_PR"] == 0]

    confusion_matrix[0][2] = abs(len(true_positive)-(confusion_matrix[0][0] + confusion_matrix[0][1]))
    confusion_matrix[1][2] = abs(len(true_negtive)-(confusion_matrix[1][0] + confusion_matrix[1][1]))
    hitting_num = len(set(hitting_rank))
    total_num = len(true_df)

    return confusion_matrix,hitting_num, total_num

def check_exisits(predicted_filePath):
    if not os.path.exists(predicted_filePath):
        print(f"文件{predicted_filePath}不存在")
        return False
    return True

if __name__ == '__main__':
    type = "Main_Experiments"

    EarlyPR_root_path = f"save_runningResutls/{type}/EarlyPR/"
    True_root_path = f"save_runningResutls/{type}/TruePR/"
    Random_root_path = f"save_runningResutls/{type}/RandomPR/"
    judge_metrics = np.zeros((11, 2), dtype=np.float32)

    folder_path = True_root_path + "*.csv"
    data_files = glob(folder_path)

    for _ in range(1, 11):
        confusion_matrix = np.zeros((3, 3))
        hitting_num, total_num = 0, 0
        threshold = _ / 10

        project_metrics = np.zeros((14, 2), dtype=np.float32)
        for i, file_path in enumerate(data_files):
            data_file = file_path.split("TruePR\\")[-1].split(".csv")[0]
            #print(data_file)
            if data_file == "julia":
                continue
            predicted_filePath = EarlyPR_root_path + data_file + ".csv"
            true_filePath = True_root_path + data_file + ".csv"
            random_filePath = Random_root_path + data_file + ".csv"

            flag = check_exisits(predicted_filePath)
            if not flag:
                true_df = pd.read_csv(true_filePath)
                confusion_ = np.zeros((3, 3))
                confusion_[0][2] = len(true_df[true_df["merged_PR"] == 1])
                confusion_[1][2] = len(true_df[true_df["merged_PR"] == 0])
                hitting_ = 0
                total_ = len(true_df)
            else:
                predicted_df = pd.read_csv(predicted_filePath)
                random_df = pd.read_csv(random_filePath)
                true_df = pd.read_csv(true_filePath)
                confusion_,hitting_, total_ = fusion_test(predicted_df, true_df,threshold)

            missing_rate = calculate_hittingAndmissingRate(confusion_)
            hitting_rate = hitting_ / total_
            project_metrics[i][0] = hitting_rate
            project_metrics[i][1] = missing_rate

            confusion_matrix += confusion_
            hitting_num += hitting_
            total_num += total_

        #np.savetxt(f"{type}_project_metrics.csv",project_metrics,delimiter=",")
        missing_rate = calculate_hittingAndmissingRate(confusion_matrix)
        hitting_rate = hitting_num / total_num
        judge_metrics[_][0] = hitting_rate
        judge_metrics[_][1] = missing_rate
        if _ == 5:
            merged_precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
            merged_recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
            rejected_precision = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
            rejected_recall = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
            rejected_acc = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[1][1] + confusion_matrix[0][1] + confusion_matrix[0][0] + confusion_matrix[1][0])

            print(f"MergedPR predicted results when threshold set at {threshold}")
            print(f"Lable-Merged Precison: {merged_precision:.4f}")
            print(f"Lable-Merged Recall: {merged_recall:.4f}")
            print(f"Lable-Rejected Precison: {rejected_precision:.4f}")
            print(f"Lable-Rejected Recall: {rejected_recall:.4f}")
            print(f"Lable-Rejected Acc: {rejected_acc:.4f}")

    df = pd.DataFrame(judge_metrics, columns=["hitting_rate", "missing_rate"])
    df.to_csv(f"{type}_judge_metrics.csv", index=False)
    print("Save Correctly")