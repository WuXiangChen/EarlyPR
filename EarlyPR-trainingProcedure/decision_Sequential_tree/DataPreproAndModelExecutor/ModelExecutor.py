import numpy as np
import pandas as pd
import copy
import sys
import torch

from decision_Sequential_tree.merged_PR_predict import MergedPRPredictor

sys.path.append("DataPreproAndModelExecutor")
from data.Test_auxiliary_information.querySets_GenerateTestCommits import _2_statistics_pNCM
from generate_TestData.MongoDB_Related.MongoDB_utils import RedisStorage
from generate_TestData.CONSTANT import db_id_Torepo,extendFeaturesIndex
from decision_Sequential_tree.evaluation_process import calculate_accuracy
import json
from decision_Sequential_tree.decision_Seq_utils import non_repeated_sampling
from datapreprocessor import datapreprocessor
from QueryExecutor import queryexecutor
dp = datapreprocessor()

class modelexecutor:
    def __init__(self, clf, csv_values, clf_name):
        self.clf = clf
        self.repo_owner = {"repo": csv_values["repo_name"], "owner": csv_values["owner_name"],
                           "PR_start_time":csv_values["PR_start_time"]}
        self.db_id = db_id_Torepo[self.repo_owner["repo"]]
        self.redis_st = RedisStorage(db=self.db_id)
        self.queryex = queryexecutor(self.repo_owner,extendFeaturesIndex, suffix="commits")
        self.queryex.redis_st = self.redis_st
        self.mergedPR_predictor = MergedPRPredictor(self.redis_st,extendFeaturesIndex,clf_name)
        self.queryex.mergedPR_predictor = self.mergedPR_predictor
        self.extendFeaturesIndex = extendFeaturesIndex

    def predict_PRcoco(self, x_testD):
        loaded_model = self.clf
        len_ = 0
        i = 0
        while True:
            len_ += 1
            com_ls = dp.findNext_coco(x_testD, len_)
            if len(com_ls) == 0 or len_ > 6:
                break
            coco = com_ls.pop(0)
            print("len_:",len_)
            while True:
                i+=1
                # if len_>5:
                #     print()
                # print(i)
                statistcsCom = self.queryex.search_and_store(coco)
                if statistcsCom is not None and len(statistcsCom) != 0:
                    y_pred = loaded_model.predict(statistcsCom).astype(np.int32)[0]
                else:
                    print("query return NULL:" + str(coco))
                    y_pred = 0
                if y_pred == 1:
                    x_testD.loc[x_testD["commits"].isin(coco), "predict_flag"] = str(coco)
                    com_ls_copy = copy.copy(com_ls)
                    for delt in com_ls_copy:
                        if len(set(delt).intersection(set(coco))) != 0:
                            com_ls.remove(delt)
                    if len(com_ls) == 0:
                        break
                    coco = com_ls.pop()
                elif y_pred==0:
                    if len(com_ls) != 0:
                        coco = com_ls.pop()
                    else:
                        break
        return x_testD

    def predict_PRcoco_(self,cm_samples):
        x_testD = pd.DataFrame(columns=["predict_flag"])
        for coco in cm_samples:
            statistcsCom = self.queryex.search_and_store(coco)
            if statistcsCom is not None and len(statistcsCom) != 0:
                y_pred = self.clf.predict(statistcsCom).astype(np.int32)[0]
            else:
                print("query return NULL:" + str(coco))
                continue

            if y_pred==0:
                x_testD.loc[len(x_testD)] = {"predict_flag":y_pred}
            else:
                x_testD.loc[len(x_testD)] = {"predict_flag": coco}
        return x_testD

    def performPredictProcess(self, filtered_test_Commits):
        x_testD = pd.DataFrame(filtered_test_Commits["commits"], columns=["commits"])
        x_testD["predict_flag"] = pd.Series([0] * len(filtered_test_Commits["commits"]))
        x_testD = self.predict_PRcoco(x_testD)
        y_pred = pd.DataFrame(x_testD["predict_flag"], columns=["predict_flag"])
        IoU_,predicted_conf_matrix = calculate_accuracy(self.queryex, filtered_test_Commits, y_pred)
        y_pre_T = y_pred.loc[y_pred["predict_flag"] != 0]
        y_pre_T.drop_duplicates(inplace=True)
        random_n = len(filtered_test_Commits["commits"])//2
        n = len(y_pre_T)
        if n == 0:
            n = max(1,random_n)
        random_cm_samples = non_repeated_sampling(filtered_test_Commits["commits"], n)
        result_df = self.generate_random_cm_datasets(random_cm_samples, self.queryex)
        random_testD = self.predict_PRcoco_(random_cm_samples)
        randomcocoPr_IoU, randomcocoPr_predicted_conf_matrix = calculate_accuracy(self.queryex, filtered_test_Commits, random_testD)

        random_y_pred_ = pd.DataFrame({"predict_flag":random_cm_samples})
        _, randomcoco_predicted_conf_matrix = calculate_accuracy(self.queryex, filtered_test_Commits,
                                                                                  random_y_pred_,direct = True)

        return IoU_,randomcocoPr_IoU,predicted_conf_matrix,\
               randomcocoPr_predicted_conf_matrix,randomcoco_predicted_conf_matrix

    def test_performPredictProcess_(self, filtered_test_Commits, clf):
        x_testD = pd.DataFrame(filtered_test_Commits["commits"], columns=["commits"])
        x_testD["predict_flag"] = pd.Series([0] * len(filtered_test_Commits["commits"]))
        IoU_, conf_matrix, random_conf_matrix, len_y_pre, len_y_true =0,0,0,0,0
        len_ = 2
        while True:
            len_ += 1
            com_ls = dp.findNext_coco(x_testD, len_)
            if len_>6:
                break
        return IoU_, conf_matrix, random_conf_matrix, len_y_pre, len_y_true

    def generate_random_cm_datasets(self,random_cm_samples, queryex):
        random_cm_final = {"predict_flag": []}
        for random_cm in random_cm_samples:
            statistcsCom = queryex.search_and_store(random_cm)
            if statistcsCom is not None and len(statistcsCom) != 0:
                random_cm_final["predict_flag"].append(random_cm)
        random_cm_df = pd.DataFrame(random_cm_final)
        return random_cm_df
