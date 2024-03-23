
import copy

import pandas as pd
import numpy as np
from datasets.Test_auxiliary_information.querySets_GenerateTestCommits import _12_getFilesLS_By_cmsha
from utils.predictingUtils import recursive_replace,replace_value_recursive
import json
from scipy.optimize import linear_sum_assignment
from functools import reduce

def transfor_eval(ls):
    if isinstance(ls,str):
        return eval(ls)
    elif isinstance(ls,list):
        return ls
    elif isinstance(ls,int):
        return ls

def calculateAccuracy_IoU(y_true_T, y_pre_T):
    result = []
    for i, co in enumerate(y_pre_T.values):
        re = []
        for j, cm in enumerate(y_true_T.values):
            IOU = len(set(cm).intersection(set(co))) / len(set(cm).union(set(co)))
            re.append(IOU)
        result.append(re)
    npResult = np.asarray(result,dtype=np.float32)
    npResult_ = 1/(npResult+1)
    row, col = linear_sum_assignment(npResult_)
    index_ls = list(zip(row,col))
    iou = []
    for index in index_ls:
        iou.append(npResult[index])
    iou_average = np.mean(iou)
    return iou_average
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
    ground_truth_files_changed = []
    for true_cocm in y_true_T:
        redis_key_true_cocm = ",".join(true_cocm)
        if redis_st.exists_key(redis_key_true_cocm) and json.loads(redis_st.get_value(redis_key_true_cocm))!=None \
                and "changed_files_LS" in json.loads(redis_st.get_value(redis_key_true_cocm)):
            values = redis_st.get_value(redis_key_true_cocm)
            values_dict = json.loads(values)
            if "changed_files_LS" in values_dict:
                changed_files_LS = values_dict["changed_files_LS"]
                ground_truth_files_changed.append(changed_files_LS)
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
                changed_files_LS = query_result[0]["changed_files_LS"]
                if changed_files_LS==[]:
                    print()
                new_add_values = {"changed_files_LS": changed_files_LS}
                redis_st.append_value(redis_key_predict_cocm, new_add_values)
                predicted_files_changed.append(changed_files_LS)
            else:
                continue

    gt_f = {"gt_files_Changed":ground_truth_files_changed}
    pre_f = {"pre_files_Changed":predicted_files_changed}
    gt_f = pd.DataFrame(gt_f)
    pre_f = pd.DataFrame(pre_f)

    return gt_f,pre_f

def calculateAccuracy_pre_filesBased_true_values(y_true_T, y_pre_T,queryex):
    y_true_Files,y_pre_Files = get_filesChanged_bySha(y_true_T, y_pre_T,queryex)
    if len(y_true_Files)==0:
        return np.nan
    if len(y_pre_Files)==0:
        return 0
    IoU_filesBased = calculateAccuracy_IoU(y_true_Files["gt_files_Changed"],y_pre_Files["pre_files_Changed"])
    return IoU_filesBased

def DataPre_evaluate_predictions(y_true, y_pred):
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

    return y_pre_T, y_pre_F, y_true_T, y_true_F

def calculateAccuracy(y_pred, y_true,modelex):
    y_pre_T, y_pre_F, y_true_T, y_true_F = DataPre_evaluate_predictions(y_true, y_pred)
    if y_pre_T.empty and not y_true_T.empty :
        IoU = 0
        IoU_filesBased = 0
    elif y_true_T.empty:
        IoU = np.nan
        IoU_filesBased = np.nan
    else:
        IoU= calculateAccuracy_IoU(y_true_T, y_pre_T[
            "predict_flag"])
        IoU_filesBased = calculateAccuracy_pre_filesBased_true_values(
            y_true_T, y_pre_T["predict_flag"],modelex)

    if y_pre_F.empty and not y_true_F.empty:
        IoU_false_avg = 0
    elif y_true_F.empty:
        IoU_false_avg = np.nan
    else:
        y_pre_F = y_pre_F.drop_duplicates()
        IoU_false_avg = calculateAccuracy_pre_false_values(y_true_F, y_pre_F)
    IoU_ = {
        "IoU":IoU,
        "IoU_false_avg":IoU_false_avg,
        "IoU_filesBased":IoU_filesBased,
    }
    return IoU_

def calculate_accuracy(queryex,filtered_test_Commits,y_pred,mergedPRFlag):
    true_flag = {
        'y_': filtered_test_Commits["label"],
        'cm_': filtered_test_Commits["true_flag"],
        "merged_PR":filtered_test_Commits["merged_PR"]}
    y_true = pd.DataFrame(true_flag)
    y_true["cm_"] = y_true["cm_"].map(transfor_eval)
    y_pred["predict_flag"] = y_pred["predict_flag"].map(transfor_eval)

    predicted_conf_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    y_pre_T_, y_mergedPR_data = None, None
    if mergedPRFlag:
        predicted_conf_matrix,y_pre_T_, y_mergedPR_data = queryex.mergedPR_predictor.mergedPredictProcess(y_true,y_pred)

    return predicted_conf_matrix, y_pre_T_, y_mergedPR_data