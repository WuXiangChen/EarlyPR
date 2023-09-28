import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model_defined import ConvolutionalNetwork,NeuralNetwork,LSTMModel,RandomForestClassifier,DecisionTreeClassifier,LGBMClassifier
from evaluation_process import DataPre_evaluate_predictions


# 用以预测mergedPR的完整途径
# 参数: y_pre_T,dataframe格式,列名为predict_flag
#    : y_mergedPR_data,列名为 "cm_","merged_PR"

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

    # 第一个主要任务是为y_pre_T中每一个predict_true PR的内容找到最近的一个true_PR,作为正样本标签的例子
    # 预测本身作为一个集合不能更改，在进行标签配置的时候一个基本的准则是，pr_bias,mergedPR_bias
    # 计算每个p_hat_j的标签时，统计与它交集最大的群体是什么，分别有merged_pr,no_merged_pr和no_match三类
    # 赋予标签的时候，应该考虑到这三类分别占据的最大值作为该预测commits集合的标签
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
        # pre_F_31_Tflag是Series结构，param是list结构
        confusion = confusion_matrix(actual_labels, predicted_labels, labels=self.class_labels)
        return confusion

    def get_testSamples_FromRedis(self,cm_):
        cm_ = cm_.values
        cm_and_trainingData = pd.DataFrame(columns=["cm_","testData"])
        for i,cm in enumerate(cm_):
            # 这里有点问题
            str_com = ",".join(cm)
            redis_com_value = self.redis_st.get_value(str_com)
            if redis_com_value and json.loads(redis_com_value) != None and 'JS_files' in json.loads(redis_com_value):
                statistcsCom = json.loads(redis_com_value)
                cm_and_trainingData.loc[i] = [cm,statistcsCom]
            else:
                print(f"在测试merged阶段，{str_com}作为键值，训练数据不存在于Redis中")
                return []
        return cm_and_trainingData

    # 导入预训练好的模型
    def load_model_from_pkl(self, model_path):
        print(model_path)
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def predict_mergedPR(self, y_pre_T, y_mergedPR_data):
        y_pre_T = y_pre_T["predict_flag"]
        # 生成对应的训练数据
        # 这里也需要判断一下y_mergedPR_data是否为空 为空的话 将所有正确标签置为-1，然后和预测出来的结果计算混淆矩阵
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

        # 这里的信息需要修正,已经在源端修正
        # 导入模型,进行预测
        pre_merged_re = self.merged_clf.predictall(test_data).tolist()
        # 计算混淆矩阵
        conf_matrix = self.judge_mergedPR_re(true_merged_re, pre_merged_re)
        return conf_matrix

    # 这里需要判断一下y_pre_T和y_pre_F分别是什么内容
    # 如果置空的话 如何形成判断？ 这里面需要分两个方面，
    # 第一种情况：
    # 对于第一种y_pre_F，如果y_pre_F为空的话，这时将输出一个全0列
    # 对于第二种y_pre_T，如果y_pre_T为空的话，这时将输出一个3*2的全0矩阵

    # 第二种情况，如果y_pre_F不为空，那经过binding过程，形成一个3*1的矩阵
    # 二者是y_pre_T上需要进行mergedPR的评估 形成一个3*2的矩阵
    def mergedPredictProcess(self, y_true, y_pred):
        y_pre_T, y_pre_F, y_true_T, y_true_F = DataPre_evaluate_predictions(y_true, y_pred)
        y_mergedPR_data = y_true.loc[y_true["y_"] == 1][["cm_", "merged_PR"]]
        if y_pre_F.empty:
            conf_matrix_31  = np.zeros((3, 1)).astype(np.int32)
        else:
            # y_mergedPR_data = pd.DataFrame(columns=["cm_","merged_PR"]) 测试 空值
            true_F_31_Tflag = self.binding_mergedPR_label(y_pre_F,y_mergedPR_data)
            # PR预测为负样本说明 其预测值皆为 -1
            conf_matrix_31 = self.judge_mergedPR_re(true_F_31_Tflag, np.asarray([-1]*y_pre_F.shape[0]))[:,2].reshape(3,1)

        if y_pre_T.empty:
            conf_matrix_32  = np.zeros((3, 2)).astype(np.int32)
        else:
            # 先对预测为正样本的集合进行mergedPR预测
            conf_matrix_32 = self.predict_mergedPR(y_pre_T,y_mergedPR_data)
            if len(conf_matrix_32)==0:
                conf_matrix_32 = np.zeros((3, 2)).astype(np.int32)
            else:
                conf_matrix_32 = conf_matrix_32[:, 0: 2]

        predicted_conf_matrix = np.hstack((conf_matrix_32,conf_matrix_31))
        return predicted_conf_matrix

    def randomcoco_mergedPredictProcess(self, y_true, random_testD):
        y_pre_T = random_testD
        # 在这种情况下y_pre_T一定不为空
        y_mergedPR_data = y_true.loc[y_true["y_"] == 1][["cm_", "merged_PR"]]
        # 先对预测为正样本的集合进行mergedPR预测
        conf_matrix_32 = self.predict_mergedPR(y_pre_T,y_mergedPR_data)
        if len(conf_matrix_32) == 0:
            conf_matrix_32 = np.zeros((3, 3)).astype(np.int32)
        # 为了统一这里返回的是3*3的矩阵，但是最后一列一定为0
        return conf_matrix_32

# 在得到正确标签和训练好的模型以后，进行测试并评估
# 其中关键在两点，第一是如何获取对应的训练数据
# 第二是如何进行有效的评估（还是打印一下混淆矩阵）
'''测试用例'''
'''
def test_predict_mergedPR():
    # 测试用例1：正常情况下的输入和预期输出
    y_pre_T = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    y_mergedPR_data = pd.DataFrame({
        "merged_PR": [1, 0, 1],
        "data": [[1, 2, 3, 10], [4, 5, 6, 11], [12, 13, 14]]
    })
    expected_output = pd.DataFrame({
        "pre": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "merged_label": [1, 0, -1]
    })
    assert predict_mergedPR(y_pre_T, y_mergedPR_data).equals(expected_output)

    # 测试用例2：空的y_pre_T和y_mergedPR_data
    empty_y_pre_T = []
    empty_y_mergedPR_data = pd.DataFrame(columns=["merged_PR", "data"])
    empty_expected_output = pd.DataFrame(columns=["pre", "merged_label"])
    assert predict_mergedPR(empty_y_pre_T, empty_y_mergedPR_data).equals(empty_expected_output)

    #测试用例3：IOU_Map中没有大于0.5的交集
    y_pre_T_no_match = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    y_mergedPR_data_no_match = pd.DataFrame({
        "merged_PR": [1, 0, -1],
        "data": [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    })
    no_match_expected_output = pd.DataFrame({
        "pre": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "merged_label": [-1, -1, -1]
    })
    assert predict_mergedPR(y_pre_T_no_match, y_mergedPR_data_no_match).equals(no_match_expected_output)

    # 测试用例4：部分交集大于0.5，部分交集小于0.5的情况
    y_pre_T_partial_match = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    y_mergedPR_data_partial_match = pd.DataFrame({
        "merged_PR": [1, 0, 1],
        "data": [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
    })
    partial_match_expected_output = pd.DataFrame({
        "pre": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "merged_label": [1, 0, 1]
    })
    assert predict_mergedPR(y_pre_T_partial_match, y_mergedPR_data_partial_match).equals(partial_match_expected_output)

    # 测试用例5：多个 merged_PR 的交集都大于0.5
    y_pre_T_multiple_matches = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    y_mergedPR_data_multiple_matches = pd.DataFrame({
        "merged_PR": [1, 0, 2],
        "data": [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
    })
    multiple_matches_expected_output = pd.DataFrame({
        "pre": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "merged_label": [1, 0, 2]
    })
    assert predict_mergedPR(y_pre_T_multiple_matches, y_mergedPR_data_multiple_matches).equals(
        multiple_matches_expected_output)

    print("所有测试用例通过！")
'''

# 运行扩展的测试用例
# test_predict_mergedPR()
