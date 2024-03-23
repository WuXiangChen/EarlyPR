import numpy as np

from datasets.dataset_utils import *
import torch
from datasets.GenerateDataForTrainTest import GenerateDataForTrainTest
from torch.utils.data import Dataset


class GenerateDataSetForMPTrainAndTest(GenerateDataForTrainTest, Dataset):
    def __init__(self, data_file, repoOwnerNameFilepath, Test_ownerSha, tokenizer, max_input_length, trainFlag=True):
        super().__init__(data_file, repoOwnerNameFilepath, Test_ownerSha, max_input_length, tokenizer)
        if callable(super().generate_MPRtraintest_data):
            self.MP_x, self.MP_y = super().generate_MPRtraintest_data(trainFlag)


    def existPRtrainData(self):
        if isinstance(self.MP_x, pd.DataFrame) or isinstance(self.MP_y, pd.DataFrame):
            return True
        return False

    def __len__(self):
        return len(self.MP_x)


    def __getitem__(self, item):

        temp_Xvalue = self.MP_x.iloc[item].iloc[0:-3].values.astype(np.float32)
        temp_Yvalue = self.MP_y.iloc[item]
        data_tensor = torch.tensor(temp_Xvalue, dtype=torch.float32)
        label_tensor = torch.tensor(temp_Yvalue, dtype=torch.float32)

        msg = self.MP_x.iloc[item].iloc[-2]
        codeDiff = self.MP_x.iloc[item].iloc[-1]

        res_token = super().TokenizeMsgAndCodeDiff(msg, codeDiff)
        return data_tensor, res_token, label_tensor
