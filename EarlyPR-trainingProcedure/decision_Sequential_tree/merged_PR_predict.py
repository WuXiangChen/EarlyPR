import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model_defined import ConvolutionalNetwork,NeuralNetwork,LSTMModel,RandomForestClassifier,DecisionTreeClassifier,LGBMClassifier
from evaluation_process import DataPre_evaluate_predictions


class MergedPRPredictor:
    def __init__(self,redis_st,extendFeaturesIndex,clf_name):
        # self.clf = self.load_model_from_pkl(self.model_path)
        self.redis_st = redis_st
        self.class_labels = [1,0,-1]
        self.extendFeaturesIndex = extendFeaturesIndex
        self.merged_clf = eval(clf_name)()

    def calculate_iou(self, x, pre):
        intersection = len(set(x["cm_"]).intersection(set(pre)))
        iou = intersection / len(set(pre))
        return iou

    def binding_mergedPR_label(self, y_pre_coco, y_mergedPR_data):
        pre_merged_result = pd.DataFrame(columns=["pre", "merged_label"])
        if y_mergedPR_data.empty:
            pre_merged_result["merged_label"] = [-1]*len(y_pre_coco)
            return pre_merged_result["merged_label"]
        for i, pre in enumerate(y_pre_coco.values):
            IOU_Map = y_mergedPR_data.apply(lambda x: self.calculate_iou(x, pre), axis=1)
            if IOU_Map.max()> 0.3:
                IoU_argmax = IOU_Map.argmax()
                pre_truth_MergedPRlabel = y_mergedPR_data.iloc[IoU_argmax]["merged_PR"]
            else:
                pre_truth_MergedPRlabel = -1
            pre_merged_result.loc[i] = [pre, pre_truth_MergedPRlabel]
        return pre_merged_result["merged_label"]

    def judge_mergedPR_re(self, actual_labels, predicted_labels):
        confusion = confusion_matrix(actual_labels, predicted_labels, labels=self.class_labels)
        return confusion

    def get_testSamples_FromRedis(self,cm_):
        cm_ = cm_.values
        cm_and_trainingData = pd.DataFrame(columns=["cm_","testData"])
        for i,cm in enumerate(cm_):
            str_com = ",".join(cm)
            redis_com_value = self.redis_st.get_value(str_com)
            if redis_com_value and json.loads(redis_com_value) != None and 'JS_files' in json.loads(redis_com_value):
                statistcsCom = json.loads(redis_com_value)
                cm_and_trainingData.loc[i] = [cm,statistcsCom]
            else:
                print(f"{str_com}")
                return []
        return cm_and_trainingData

    def load_model_from_pkl(self, model_path):
        print(model_path)
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def predict_mergedPR(self, y_pre_T, y_mergedPR_data):
        y_pre_T = y_pre_T["predict_flag"]
        test_data = self.get_testSamples_FromRedis(y_pre_T)
        if len(test_data)==0:
            return []
        test_data = pd.json_normalize(test_data["testData"])
        test_data = test_data.reindex(columns=self.extendFeaturesIndex)
        if y_mergedPR_data.empty and len(y_pre_T) != 0:
            true_merged_re = [-1] * len(y_pre_T)
        else:
            true_merged_re = self.binding_mergedPR_label(y_pre_T, y_mergedPR_data)

        if "changed_files_LS" in test_data.columns:
            test_data.drop("changed_files_LS", inplace=True, axis=1)
        pre_merged_re = self.merged_clf.predictall(test_data).tolist()
        conf_matrix = self.judge_mergedPR_re(true_merged_re, pre_merged_re)
        return conf_matrix
    def mergedPredictProcess(self, y_true, y_pred):
        y_pre_T, y_pre_F, y_true_T, y_true_F = DataPre_evaluate_predictions(y_true, y_pred)
        y_mergedPR_data = y_true.loc[y_true["y_"] == 1][["cm_", "merged_PR"]]
        if y_pre_F.empty:
            conf_matrix_31  = np.zeros((3, 1)).astype(np.int32)
        else:
            true_F_31_Tflag = self.binding_mergedPR_label(y_pre_F,y_mergedPR_data)
            conf_matrix_31 = self.judge_mergedPR_re(true_F_31_Tflag, np.asarray([-1]*y_pre_F.shape[0]))[:,2].reshape(3,1)

        if y_pre_T.empty:
            conf_matrix_32  = np.zeros((3, 2)).astype(np.int32)
        else:
            conf_matrix_32 = self.predict_mergedPR(y_pre_T,y_mergedPR_data)
            if len(conf_matrix_32)==0:
                conf_matrix_32 = np.zeros((3, 2)).astype(np.int32)
            else:
                conf_matrix_32 = conf_matrix_32[:, 0: 2]

        predicted_conf_matrix = np.hstack((conf_matrix_32,conf_matrix_31))
        return predicted_conf_matrix

    def randomcoco_mergedPredictProcess(self, y_true, random_testD):
        y_mergedPR_data = y_true.loc[y_true["y_"] == 1][["cm_", "merged_PR"]]
        conf_matrix_32 = self.predict_mergedPR(y_pre_T,y_mergedPR_data)
        if len(conf_matrix_32) == 0:
            conf_matrix_32 = np.zeros((3, 3)).astype(np.int32)
        return conf_matrix_32
