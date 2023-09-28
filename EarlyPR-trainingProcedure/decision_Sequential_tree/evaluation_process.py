# 用作评估过程
import copy
from functools import reduce
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from data.Test_auxiliary_information.querySets_GenerateTestCommits import _12_getFilesLS_By_cmsha
from decision_Seq_utils import recursive_replace,replace_value_recursive
import json

from functools import reduce

def transfor_eval(ls):
    if isinstance(ls,str):
        return eval(ls)
    elif isinstance(ls,list):
        return ls
    elif isinstance(ls,int):
        return ls

def calculateAccuracy_pre_true_values_tmp_for_files(y_true_T, y_pre_T):
    IoU_ground_truth_true_sum = 0
    IoU_predicted_truth_true_sum = 0
    for cocm in y_true_T.values:
        # print(cocm)
        # intersection_lengths = [len(set(x).intersection(set(cocm))) for x in y_pre_T]
        # union_lengths = [len(set(x).union(set(cocm))) for x in y_pre_T]
        # print("intersection_lengths:")
        # print(intersection_lengths)
        # print("union_lengths:")
        # print(union_lengths)

        IOU = y_pre_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(cocm))).max()
        IoU_ground_truth_true_sum+=IOU

    for cocm in y_pre_T.values:
        IOU = y_true_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(cocm))).max()
        IoU_predicted_truth_true_sum+=IOU

    IoU_predicted_truth_true_avg = IoU_predicted_truth_true_sum/len(y_pre_T)

    IoU_ground_truth_true_avg =IoU_ground_truth_true_sum / len(y_true_T)
    return IoU_ground_truth_true_avg,IoU_predicted_truth_true_avg

def calculateAccuracy_pre_true_values(y_true_T, y_pre_T):
    IoU_ground_truth_true_sum = 0
    IoU_predicted_truth_true_sum = 0
    for cocm in y_true_T.values:
        # print(cocm)
        IOU = y_pre_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(cocm))).max()
        IoU_ground_truth_true_sum+=IOU

    for cocm in y_pre_T.values:
        IOU = y_true_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(cocm))).max()
        IoU_predicted_truth_true_sum+=IOU

    IoU_predicted_truth_true_avg = IoU_predicted_truth_true_sum/len(y_pre_T)

    IoU_ground_truth_true_avg =IoU_ground_truth_true_sum / len(y_true_T)
    return IoU_ground_truth_true_avg,IoU_predicted_truth_true_avg

def calculateAccuracy_pre_true_values_rp_hat(y_true_T, y_pre_T):
    IoU_ground_truth_true_sum = 0
    IoU_predicted_truth_true_sum = 0
    for cocm in y_true_T.values:
        IOU = y_pre_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(x).union(set(cocm)))).max()
        IoU_ground_truth_true_sum+=IOU

    for cocm in y_pre_T.values:
        IOU = y_true_T.map(lambda x: len(set(x).intersection(set(cocm)))/len(set(x).union(set(cocm)))).max()
        IoU_predicted_truth_true_sum+=IOU

    IoU_predicted_truth_true_avg = IoU_predicted_truth_true_sum/len(y_pre_T)

    IoU_ground_truth_true_avg =IoU_ground_truth_true_sum / len(y_true_T)
    return IoU_ground_truth_true_avg,IoU_predicted_truth_true_avg

def calculateAccuracy_pre_false_values(y_true_F, y_pre_F):
    IoU_false_sum = 0
    cocm = y_true_F.iloc[0]
    for cm in cocm:
        if cm in y_pre_F.values:
            IoU_false_sum+=1
    IoU_false_avg = IoU_false_sum / len(cocm)
    return IoU_false_avg

def get_filesChanged_bySha(y_true_T, y_pre_T,queryex):
    replacement = queryex.replacement
    redis_st = queryex.redis_st
    repla_query = recursive_replace(_12_getFilesLS_By_cmsha,replacement)
    # 在这里利用queryex完成查询过程应该就可以了
    # 根据现有得规则true_cocm应该本身即是有序的
    ground_truth_files_changed = []
    for true_cocm in y_true_T:
        redis_key_true_cocm = ",".join(true_cocm)
        #print(redis_key_true_cocm)
        if redis_st.exists_key(redis_key_true_cocm) and json.loads(redis_st.get_value(redis_key_true_cocm))!=None \
                and "changed_files_LS" in json.loads(redis_st.get_value(redis_key_true_cocm)):
            values = redis_st.get_value(redis_key_true_cocm)
            values_dict = json.loads(values)
            if "changed_files_LS" in values_dict:
                changed_files_LS = values_dict["changed_files_LS"]
                ground_truth_files_changed.append(changed_files_LS)
            #得到了true_cocm 对应的 changed_files_LS
        else:
            tmp_repla_query = repla_query.copy()
            tmp_repla_query[0] = replace_value_recursive(tmp_repla_query[0],"%sha_LS",["$sha",true_cocm])
            try:
                query_result = list(queryex.coll.aggregate(tmp_repla_query, allowDiskUse=True))
            except Exception as e:
                print("query_result")
                print(e)
                continue
            if len(query_result)!=0 and query_result[0]["changed_files_LS"]!=[]:
                if not redis_st.exists_key(redis_key_true_cocm) or json.loads(redis_st.get_value(redis_key_true_cocm))==None:
                    # 生成对应的存储键形式
                    changed_files_LS = query_result[0]["changed_files_LS"]
                    new_add_values = {"changed_files_LS": changed_files_LS}
                    new_add_str = json.dumps(new_add_values)
                    redis_st.set_key_value(redis_key_true_cocm,new_add_str)
                    ground_truth_files_changed.append(changed_files_LS)
                else:
                    changed_files_LS = query_result[0]["changed_files_LS"]
                    new_add_values = {"changed_files_LS": changed_files_LS}
                    redis_st.append_value(redis_key_true_cocm, new_add_values)
                    ground_truth_files_changed.append(changed_files_LS)
            else:
                print(f"在评估阶段commits filesname:{true_cocm}的结果为不存在")
                continue

    predicted_files_changed = []
    for predict_cocm in y_pre_T:
        redis_key_predict_cocm = ",".join(predict_cocm)
        if redis_st.exists_key(redis_key_predict_cocm) and json.loads(redis_st.get_value(redis_key_predict_cocm))!=None and \
                "changed_files_LS" in json.loads(redis_st.get_value(redis_key_predict_cocm)) and \
            json.loads(redis_st.get_value(redis_key_predict_cocm))["changed_files_LS" ]!=[] :
            values = redis_st.get_value(redis_key_predict_cocm)
            values_dict = json.loads(values)
            changed_files_LS = values_dict["changed_files_LS"]
            predicted_files_changed.append(changed_files_LS)
            # 得到了true_cocm 对应的 changed_files_LS
        else:
            tmp_repla_query = repla_query.copy()
            tmp_repla_query[0] = replace_value_recursive(tmp_repla_query[0], "%sha_LS", ["$sha",predict_cocm])
            try:
                query_result = list(queryex.coll.aggregate(tmp_repla_query, allowDiskUse=True))
            except Exception as e:
                print("query_result")
                print(e)
                continue
            if len(query_result)!=0 and query_result[0]["changed_files_LS"]!=[]:
                # 生成对应的存储键形式
                # 对于predict_cocm来说，所有存储key，则意味必然存储着changed_files_LS，所以这里不太需要判空
                changed_files_LS = query_result[0]["changed_files_LS"]
                if changed_files_LS==[]:
                    print()
                new_add_values = {"changed_files_LS": changed_files_LS}
                redis_st.append_value(redis_key_predict_cocm, new_add_values)
                predicted_files_changed.append(changed_files_LS)
            else:
                print(f"在评估阶段commits filesname:{predict_cocm}的结果为不存在")
                continue

    gt_f = {"gt_files_Changed":ground_truth_files_changed}
    pre_f = {"pre_files_Changed":predicted_files_changed}
    gt_f = pd.DataFrame(gt_f)
    pre_f = pd.DataFrame(pre_f)

    return gt_f,pre_f

def calculateAccuracy_pre_filesBased_true_values(y_true_T, y_pre_T,queryex):
    y_true_Files,y_pre_Files = get_filesChanged_bySha(y_true_T, y_pre_T,queryex)
    if len(y_true_Files)==0:
        return np.nan,np.nan
    if len(y_pre_Files)==0:
        return 0,0
    IoU_ground_truth_filesBased_true_avg, IoU_predicted_truth_filesBased_true_avg = \
        calculateAccuracy_pre_true_values_tmp_for_files(y_true_Files["gt_files_Changed"],y_pre_Files["pre_files_Changed"])
    return IoU_ground_truth_filesBased_true_avg, IoU_predicted_truth_filesBased_true_avg

def DataPre_evaluate_predictions(y_true, y_pred):
    # 首先判断总的数量 对比是否正确
    # 计算y_true中所有"cm_"的总长度
    y_true_num = y_true["cm_"].map(len).sum()

    # 计算y_pred中"predict_flag"的长度
    y_pred_num = len(y_pred["predict_flag"])

    # 如果y_true_num与y_pred_num不一致，引发异常
    # 添加随机组合的预测以后 这里的内容不能加上取了
    # if y_true_num != y_pred_num:
    #     raise ValueError("预测结果与真实结果不一致，查看错误")

    # 评估预测正确的结果
    y_pre_T = y_pred.loc[y_pred["predict_flag"] != 0]
    y_pre_T.drop_duplicates(inplace=True)
    y_true_T = y_true.loc[y_true["y_"] == 1]["cm_"]
    all_cocm_union_set = set(reduce(lambda x, y: set(x).union(set(y)), y_true["cm_"]))

    if y_pre_T.empty:
        y_pre_F = list(all_cocm_union_set)
    else:
        pre_true_set = set(reduce(lambda x, y: set(x).union(y), y_pre_T["predict_flag"]))
        y_pre_F = list(all_cocm_union_set.difference(pre_true_set))
    y_pre_F = pd.DataFrame(y_pre_F,columns=["predict_flag"])
    y_true_F = y_true.loc[y_true["y_"] == 0]["cm_"]

    # 返回预测结果
    return y_pre_T, y_pre_F, y_true_T, y_true_F

def calculateAccuracy(y_pred, y_true,modelex):
    y_pre_T, y_pre_F, y_true_T, y_true_F = DataPre_evaluate_predictions(y_true, y_pred)
    # 评估merged_PR的计算结果
    if y_pre_T.empty and not y_true_T.empty :
        IoU_ground_truth_true_avg = 0
        IoU_predicted_truth_true_avg = 0
        IoU_ground_truth_filesBased_true_avg = 0
        IoU_predicted_truth_filesBased_true_avg =0
    elif y_true_T.empty:
        IoU_ground_truth_true_avg = np.nan
        IoU_predicted_truth_true_avg = np.nan
        IoU_ground_truth_filesBased_true_avg = np.nan
        IoU_predicted_truth_filesBased_true_avg = np.nan
    else:
        # 对预测进行去重
        #y_pre_T = y_pre_T.drop_duplicates()
        IoU_ground_truth_true_avg, IoU_predicted_truth_true_avg = calculateAccuracy_pre_true_values(y_true_T, y_pre_T[
            "predict_flag"])
        IoU_ground_truth_filesBased_true_avg, IoU_predicted_truth_filesBased_true_avg = calculateAccuracy_pre_filesBased_true_values(
            y_true_T, y_pre_T["predict_flag"],modelex)

    # 因为两个集合互补，所以可以用这里真实的集合 计算得到 预测集合中0的实际表示，且获得顺序不重要
    if y_pre_F.empty and not y_true_F.empty:
        IoU_false_avg = 0
    elif y_true_F.empty:
        IoU_false_avg = np.nan
    # 评估预测错误的结果
    else:
        # 对预测结果进行去重
        y_pre_F = y_pre_F.drop_duplicates()
        IoU_false_avg = calculateAccuracy_pre_false_values(y_true_F, y_pre_F)

    Iou_ = {
        "IoU_ground_truth_true_avg":IoU_ground_truth_true_avg,
        "IoU_predicted_truth_true_avg":IoU_predicted_truth_true_avg,
        "IoU_false_avg":IoU_false_avg,
        "IoU_ground_truth_filesBased_true_avg":IoU_ground_truth_filesBased_true_avg,
        "IoU_predicted_truth_filesBased_true_avg":IoU_predicted_truth_filesBased_true_avg
    }
    return Iou_

def calculate_accuracy(queryex,filtered_test_Commits,y_pred,direct = False):
    true_flag = {
        'y_': filtered_test_Commits["label"],
        'cm_': filtered_test_Commits["true_flag"],
        "merged_PR":filtered_test_Commits["merged_PR"]}
    y_true = pd.DataFrame(true_flag)
    y_true["cm_"] = y_true["cm_"].map(transfor_eval)
    # 在进行真实计算的时候其实不需要知道 预测为负样本的内容是如何进行组织的，但是在mergedPR预测过程中，这些信息是有必要知道嘛？
    # 负样本不存在集合的概念！
    y_pred["predict_flag"] = y_pred["predict_flag"].map(transfor_eval)

    if not direct:
        IoU_ = calculateAccuracy(y_pred,y_true,queryex)
        # 这里成立一个新的方法 专门做predictedPR的mergedPR的预测工作
        predicted_conf_matrix = queryex.mergedPR_predictor.mergedPredictProcess(y_true,y_pred)
    else:
        IoU_ = None
        predicted_conf_matrix = queryex.mergedPR_predictor.randomcoco_mergedPredictProcess(y_true, y_pred)
    return IoU_,predicted_conf_matrix

# 5. 设计评估结果：
def calculateTop1Accuracy(y_pred,y_test):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    # 计算特异度
    specificity = TN / (TN + FP)
    # 计算假阳性率
    FPR = FP / (FP + TN)
    # 计算精确度
    precision = TP / (TP + FP)
    # 计算召回率
    recall = TP / (TP + FN)
    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)

    print("特异度:", specificity)
    print("假阳性率:", FPR)
    print("精确度:", precision)
    print("召回率:", recall)
    print("Accuracy:", accuracy)
    print("===================================================")

    return accuracy, recall, precision, specificity, FPR