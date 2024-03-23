
import copy
import glob
import warnings
import numpy as np
import pandas as pd
from utils.SaveUtils import save_results
from datasets.GenerateDataSetForMPTrain import GenerateDataSetForMPTrainAndTest
from utils.testUtils import generate_owner_test_commits, calculate_metrics, compare_models, \
    generate_owner_commits

# Disable all warnings
warnings.filterwarnings("ignore")
from utils.runUtils import load_opt_sched_from_ckptTest, update_result
from datasets.GenerateDataSetForPRTrain import GenerateDataSetForPRTrainAndTest
from CONSTANT.CONSTANT import *
from CONSTANT.CONSTANT import CB_max_Output_length
from performPredict.performPredictProcess import performPredict
import torch

csv_values = pd.read_csv(repo_owner_name_Filepath)
Test_ownerShas_ls = glob.glob(Test_ownerSha)
def test(data_file, modelPath,MP_modelPath, model, optimizer, scheduler, test_device, mergedPR = False):
    repo_owner = {}
    repo_owner["repo_name"] = data_file
    repo_owner["owner_name"] = csv_values[csv_values["repo_names"] == repo_owner["repo_name"]]["owner_names"].iloc[0]
    repo_owner["PR_start_time"] = csv_values[csv_values["repo_names"] == repo_owner["repo_name"]]["PR_start_time"].iloc[0]

    if test_device == -1:
        test_device = torch.device("cpu")
    else:
        test_device = torch.device(test_device)
    Mp_model = None
    if not mergedPR:
        if modelPath != None:
            model = load_opt_sched_from_ckptTest(modelPath,model, test_device)
        tokenizer = model.CodeBertS.tokenizer
    else:
        if modelPath is None or MP_modelPath is None:
            raise ""
        model_ = copy.deepcopy(model)
        if modelPath != None:
            model = load_opt_sched_from_ckptTest(modelPath, model, test_device)
        tokenizer = model.CodeBertS.tokenizer
        if MP_modelPath != None:
            Mp_model = load_opt_sched_from_ckptTest(MP_modelPath, model_, test_device)

    dataset = GenerateDataSetForPRTrainAndTest(data_file, repo_owner_name_Filepath, Test_ownerSha, tokenizer,
                                               CB_max_Output_length, trainFlag)
    X_data, y_data = dataset.generate_PRtraintest_data(trainFlag)
    X_data["label"] = y_data
    X_data.reset_index(inplace=True)


    if not mergedPR:
        ProdictP = performPredict(model,repo_owner,clf_trained=None)
    else:
        compare_models(model,Mp_model)
        ProdictP = performPredict(model, repo_owner, clf_trained=Mp_model)


    test_owner = X_data["owner"]
    filtered_test_Commits = generate_owner_commits(X_data,False)


    pr_all = []
    random_pr_all = []
    conf_matrix_pr_per = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    conf_matrix_randompr_per = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    EarlyPR_Result = None
    RandomPR_Result = None
    True_mergedPR = None
    for index, row in filtered_test_Commits.iterrows():
        if index<1:
            continue
        predicted_conf_matrix, randomcocoPr_matrix, y_pre_T_earlyPR, y_pre_T_RandomPR, y_mergedPR_data \
            = ProdictP.performPredictProcess(row, mergedPR)
        EarlyPR_Result = update_result(EarlyPR_Result, y_pre_T_earlyPR)
        RandomPR_Result = update_result(RandomPR_Result, y_pre_T_RandomPR)
        True_mergedPR = update_result(True_mergedPR, y_mergedPR_data)

        conf_matrix_pr_per = conf_matrix_pr_per + predicted_conf_matrix
        conf_matrix_randompr_per = conf_matrix_randompr_per + randomcocoPr_matrix
        evaluate_pr = calculate_metrics(conf_matrix_pr_per, i=index)
        evaluate_randompr = calculate_metrics(conf_matrix_randompr_per,random_flag=True,i=index)

        random_pr_all.append(evaluate_randompr)
        pr_all.append(evaluate_pr)

    save_results(EarlyPR_Result, data_file, "EarlyPR")
    save_results(RandomPR_Result, data_file, "RandomPR")
    calculate_metrics(conf_matrix_pr_per)
    calculate_metrics(conf_matrix_randompr_per, random_flag=True)