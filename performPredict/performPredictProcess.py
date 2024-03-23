import copy

import numpy as np
import pandas as pd
import torch

from utils.MongoDBUtils import RedisStorage
from performPredict.evaluation_process import calculate_accuracy
from utils.predictingUtils import non_repeated_sampling
from performPredict.DataPreproAndModelExecutor.datapreprocessor import datapreprocessor
from performPredict.DataPreproAndModelExecutor.QueryExecutor import queryexecutor
from CONSTANT.Predicting_Constant import db_id_Torepo,extendFeaturesIndex
from performPredict.merged_PR_predict import MergedPRPredictor

dp = datapreprocessor()
class performPredict():

    def __init__(self, clf, csv_values, clf_trained):
        self.clf = clf
        self.repo_owner = {"repo": csv_values["repo_name"], "owner": csv_values["owner_name"],
                           "PR_start_time": csv_values["PR_start_time"]}

        self.db_id = db_id_Torepo[self.repo_owner["repo"]]
        self.redis_st = RedisStorage(db=self.db_id)
        self.queryex = queryexecutor(self.repo_owner, extendFeaturesIndex, suffix="commits")
        self.queryex.redis_st = self.redis_st
        self.mergedPR_predictor = None
        if clf_trained is not None:
            self.mergedPR_predictor = MergedPRPredictor(self.redis_st, extendFeaturesIndex, clf_trained)
        self.queryex.mergedPR_predictor = self.mergedPR_predictor
        self.extendFeaturesIndex = extendFeaturesIndex

    def predict_PRcoco_(self, cm_samples):
        x_testD = pd.DataFrame(columns=["predict_flag"])
        for coco in cm_samples:
            statistcsCom = self.queryex.search_and_store(coco)
            if statistcsCom is not None and len(statistcsCom) != 0:
                staCom = statistcsCom.iloc[:, :-2]
                staCom = torch.tensor(staCom.values.astype(float), dtype=torch.float32)
                MsgAndCodeDiff = statistcsCom.iloc[:, -2:]
                y_pred = self.clf.predict(staCom,MsgAndCodeDiff).tolist()[0]
            else:
                print("查询结果为空" + str(coco))
                continue

            if y_pred[0] < 0.5:
                x_testD.loc[len(x_testD)] = {"predict_flag": 0}
            else:
                x_testD.loc[len(x_testD)] = {"predict_flag": coco}

        return x_testD

    def predict_PRcoco(self, x_testD):
        len_ = 6
        #len_ = 0
        i = 0
        while True:
            len_ = min(len_-1, len(x_testD.loc[x_testD["predict_flag"].isin([0])]))
            #len_ += 1
            if len_< 1:
                break
            com_ls = dp.findNext_coco(x_testD, len_)
            if len(com_ls) == 0 :
                break
            print("len_:", len_)
            # print("com_ls:", len(com_ls))
            coco = com_ls.pop(0)
            while True:
                i += 1
                # print(i)
                statistcsCom = self.queryex.search_and_store(coco)
                if statistcsCom is not None and len(statistcsCom) != 0:
                    staCom = statistcsCom.iloc[:, :-2]
                    staCom = torch.tensor(staCom.values.astype(float), dtype=torch.float32)
                    MsgAndCodeDiff = statistcsCom.iloc[:, -2:]
                    y_pred = self.clf.predict(staCom,MsgAndCodeDiff).tolist()[0]
                else:
                    print("" + str(coco))
                    y_pred = 0
                # print("y_pred：", str(y_pred))
                if y_pred[0] >= 0.5:
                    x_testD.loc[x_testD["commits"].isin(coco), "predict_flag"] = str(coco)
                    com_ls_copy = copy.copy(com_ls)
                    for delt in com_ls_copy:
                        if len(set(delt).intersection(set(coco))) != 0:
                            com_ls.remove(delt)
                    if len(com_ls) == 0:
                        break
                    coco = com_ls.pop()

                else:
                    if len(com_ls) != 0:
                        coco = com_ls.pop()
                    else:
                        break

        return x_testD

    def performPredictProcess(self,filtered_test_Commits, mergedPRFlag):
        x_testD = pd.DataFrame(filtered_test_Commits["commits"], columns=["commits"])
        x_testD["predict_flag"] = pd.Series([0] * len(filtered_test_Commits["commits"]))
        x_testD = self.predict_PRcoco(x_testD)
        y_pred = pd.DataFrame(x_testD["predict_flag"], columns=["predict_flag"])
        predicted_conf_matrix, y_pre_T_earlyPR, y_mergedPR_data = calculate_accuracy(self.queryex, filtered_test_Commits, y_pred,mergedPRFlag)

        randomcocoPr_matrix = predicted_conf_matrix
        y_pre_T_RandomPR = y_pre_T_earlyPR
        random_cm_samples = non_repeated_sampling(filtered_test_Commits["commits"], y_pred)
        random_testD = self.predict_PRcoco_(random_cm_samples)
        randomcocoPr_matrix, y_pre_T_RandomPR, y_mergedPR_data = calculate_accuracy(self.queryex, filtered_test_Commits,
                                                                                  random_testD,mergedPRFlag)

        return  predicted_conf_matrix, randomcocoPr_matrix, y_pre_T_earlyPR, y_pre_T_RandomPR, y_mergedPR_data