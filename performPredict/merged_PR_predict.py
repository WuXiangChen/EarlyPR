import json
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from .evaluation_process import DataPre_evaluate_predictions
class MergedPRPredictor:
    def __init__(self,redis_st,extendFeaturesIndex,clf_trained):
        # self.clf = self.load_model_from_pkl(self.model_path)
        self.redis_st = redis_st
        self.class_labels = [1,0,-1]
        self.extendFeaturesIndex = extendFeaturesIndex
        self.merged_clf = clf_trained

    def calculate_iou(self, x, pre):
        intersection = len(set(x["cm_"]).intersection(set(pre)))
        iou = intersection / len(set(x["cm_"]).union(set(pre)))
        return iou

    def calculateAccuracy_OneToOne_IoU(self, y_true_T, y_pre_T):
        result = []
        for i, co in enumerate(y_pre_T.values):
            re = []
            for j, cm in enumerate(y_true_T.values):
                IOU = len(set(cm).intersection(set(co))) / len(set(cm).union(set(co)))
                re.append(IOU)
            result.append(re)
        npResult = np.asarray(result, dtype=np.float32)
        npResult_ = np.where(npResult < 0.5,  0, 1)
        npResult_ = 1 / (npResult_ + 1)
        row, col = linear_sum_assignment(npResult_)
        index_ls = list(zip(row, col))
        iou = []
        for index in index_ls:
            iou.append(npResult[index])
        return iou,index_ls

    def calculateAccuracy_IoU(self, y_true_T, y_pre_T):
        result = []
        for i, co in enumerate(y_pre_T.values):
            re = []
            for j, cm in enumerate(y_true_T.values):
                IOU = len(set(cm).intersection(set(co))) / len(set(cm).union(set(co)))
                re.append(IOU)
            result.append(re)

        # 这里不再要求一一配对了
        npResult = np.asarray(result, dtype=np.float32)
        return npResult

    def binding_mergedPR_label(self, y_pre_coco, y_mergedPR_data):
        pre_merged_result = pd.DataFrame(columns=["pre", "merged_label"])
        pre_merged_result["pre"] = y_pre_coco
        if y_mergedPR_data.empty:
            pre_merged_result["merged_label"] = [-1]*len(y_pre_coco)
            return pre_merged_result["merged_label"]

        npResult = self.calculateAccuracy_IoU(y_mergedPR_data["cm_"],y_pre_coco)

        len_pre = len(y_pre_coco)
        for i in range(len_pre):
            index = int(npResult[i].argmax())
            re = npResult[i][index]
            if re > 0.5:
                pre_merged_result.iloc[i,1] =y_mergedPR_data.iloc[index]["merged_PR"]

        pre_merged_result["merged_label"] = pre_merged_result["merged_label"].fillna(-1)
        return pre_merged_result["merged_label"]

    def judge_mergedPR_re(self, actual_labels, predicted_labels):
        confusion = confusion_matrix(actual_labels, predicted_labels, labels=self.class_labels)
        return confusion

    def get_testSamples_FromRedis(self,cm_):
        cm_ = cm_.values
        cm_and_trainingData = pd.DataFrame(columns=["cm_","testData"])
        for i,cm in enumerate(cm_):

            cm.sort()
            str_com = ",".join(cm)
            redis_com_value = self.redis_st.get_value(str_com)
            if redis_com_value and json.loads(redis_com_value) != None:
                statistcsCom = json.loads(redis_com_value)
                cm_and_trainingData.loc[i] = [cm,statistcsCom]
            else:
                print(f"在测试merged阶段，{str_com}作为键值，训练数据不存在于Redis中")
                return []
        return cm_and_trainingData


    def load_model_from_pkl(self, model_path):
        print(model_path)
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def predict_mergedPR(self, y_pre_T_, y_mergedPR_data):
        y_pre_T = y_pre_T_["predict_flag"]
        true_merged_num = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 1])
        true_rejected_num = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 0])
        if y_pre_T.shape[0] >1:
            print()
        if y_mergedPR_data.empty and len(y_pre_T) != 0:
            true_merged_re = [-1] * len(y_pre_T)
        else:
            true_merged_re = self.binding_mergedPR_label(y_pre_T, y_mergedPR_data)

        y_pre_T_["true_flag"] = true_merged_re
        test_data = self.get_testSamples_FromRedis(y_pre_T)

        if len(test_data)==0:
            matrix_evaluation = np.zeros((3, 3)).astype(np.int32)
            matrix_evaluation[0, 2] = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 1])
            matrix_evaluation[1, 2] = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 0])
            return matrix_evaluation

        test_data = pd.json_normalize(test_data["testData"])
        test_data = test_data.reindex(columns=self.extendFeaturesIndex)
        if "changed_files_LS" in test_data.columns:
            test_data.drop("changed_files_LS", inplace=True, axis=1)

        pre_merged_re = []
        staCom = test_data.iloc[:, :-2]
        staCom = staCom.astype(float)
        staCom = torch.tensor(staCom.values, dtype=torch.float32)

        for i in range(test_data.shape[0]):
            _ = staCom[i].unsqueeze(0)
            MsgAndCodeDiff = pd.DataFrame(test_data.iloc[i, -2:]).T
            pre_merged_re.extend(self.merged_clf.predict(_,MsgAndCodeDiff).tolist()[0])

        y_pre_T_["predicted_flag"] =  pd.Series([0 if i<0.5 else 1 for i in pre_merged_re])
        print("pre_merged_re:",pre_merged_re)
        print("true_merged_re:",true_merged_re)
        pre_merged_re = [1 if i > 0.5 else 0 for i in pre_merged_re]

        conf_matrix = self.judge_mergedPR_re(true_merged_re, pre_merged_re)
        conf_matrix[0,2] = true_merged_num - conf_matrix[0,0] - conf_matrix[0,1]
        conf_matrix[1,2] = true_rejected_num - conf_matrix[1,0] - conf_matrix[1,1]

        return conf_matrix, y_pre_T_, y_mergedPR_data

    def mergedPredictProcess(self, y_true, y_pred):
        y_pre_T, y_pre_F, y_true_T, y_true_F = DataPre_evaluate_predictions(y_true, y_pred)
        y_mergedPR_data = y_true.loc[y_true["y_"] == 1][["cm_", "merged_PR"]]
        # 放弃讨论预测为负样本的集合
        if y_pre_T.empty:
            matrix_evaluation = np.zeros((3, 3)).astype(np.int32)
            matrix_evaluation[0, 2] = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 1])
            matrix_evaluation[1, 2] = len(y_mergedPR_data[y_mergedPR_data["merged_PR"] == 0])
            return matrix_evaluation,None,y_mergedPR_data

        else:
            # 先对预测为正样本的集合进行mergedPR预测
            matrix_evaluation,y_pre_T_, y_mergedPR_data = self.predict_mergedPR(y_pre_T,y_mergedPR_data)

            return matrix_evaluation,y_pre_T_, y_mergedPR_data
