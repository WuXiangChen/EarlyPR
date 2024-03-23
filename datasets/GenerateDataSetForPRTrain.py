import numpy as np

from datasets.dataset_utils import *
import torch
from datasets.GenerateDataForTrainTest import GenerateDataForTrainTest
from torch.utils.data import Dataset

class GenerateDataSetForPRTrainAndTest(GenerateDataForTrainTest,Dataset):
    def __init__(self, data_file, repo_owner_name_Filepath, Test_ownerSha, tokenizer, max_input_length, trainFlag=True):
        super().__init__(data_file, repo_owner_name_Filepath, Test_ownerSha,max_input_length, tokenizer)
        if callable(super().generate_PRtraintest_data):
            self.PR_x, self.PR_y= super().generate_PRtraintest_data(trainFlag)

    def existPRtrainData(self):
        if isinstance(self.PR_x, pd.DataFrame) or isinstance(self.PR_y, pd.DataFrame):
            return True
        return False

    def __len__(self):
        return len(self.PR_x)


    def __getitem__(self, item):
        # 将数据加入torch中
        temp_Xvalue = self.PR_x.iloc[item].iloc[0:-2].values.astype(np.float32)
        temp_Yvalue = self.PR_y.iloc[item]
        data_tensor = torch.tensor(temp_Xvalue, dtype=torch.float32)
        label_tensor = torch.tensor(temp_Yvalue, dtype=torch.float32)

        msg = self.PR_x.iloc[item].iloc[-2]
        codeDiff = self.PR_x.iloc[item].iloc[-1]

        res_token = super().TokenizeMsgAndCodeDiff(msg, codeDiff)
        return data_tensor,res_token, label_tensor