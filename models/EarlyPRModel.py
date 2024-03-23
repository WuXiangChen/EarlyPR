### 本节用于定义EarlyPR 训练过程中的全部模型 和 forward过程
'''
    导包区
'''
import os
import sys
import torch
from torch import nn
import pathlib
from datasets.tokenizingMsgAndCodeDiff import tokenizeMsgAndCodeDiff
from CONSTANT.CONSTANT import CB_max_Output_length

class EarlyPRModel(nn.Module):

    def __init__(self, STAFeaExtract: nn.Module = None, CodeBertS  = None, CodeBertHotPath: pathlib = None,
                 Trans_Encoder: nn.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.STAFeaExtract = STAFeaExtract
        if CodeBertHotPath!=None:
            self.InitializeFromPreTrained(CodeBertS,CodeBertHotPath)
        else:
            self.CodeBertS = CodeBertS
        self.Trans_Encoder = Trans_Encoder
        self.convs_code = nn.Conv1d(kernel_size=5, in_channels=1, out_channels=1)
        self.pool_code = nn.AvgPool1d(kernel_size=3)
        self.linear_code = nn.Linear(in_features=254, out_features=16)
        self.linears = nn.Sequential(nn.Linear(CodeBertS.linear_size[0], 1))
        self.sigmod = nn.Sigmoid()

    # 这说明数据集还得改
    def forward(self,STA_data, MsgAndCodeDiff):
        feaCom = self.STAFeaExtract(STA_data)
        if MsgAndCodeDiff!=None:
            msgCode = self.CodeBertS.model(**MsgAndCodeDiff).last_hidden_state
            msgCode_ = msgCode[:, 0, :].squeeze(1)
            msgCode_ = self.convs_code(msgCode_.unsqueeze(1))
            msgCode_ = self.pool_code(msgCode_)
            msgCode_ = self.linear_code(msgCode_.view(-1, 254))
            feaCom = torch.cat([feaCom,msgCode_],dim=-1)

            feaCom = feaCom.unsqueeze(1)

        transFea = self.Trans_Encoder(feaCom)
        transFea = transFea.squeeze(1)
        linearFea = self.linears(transFea)
        output =  self.sigmod(linearFea)

        return output

    def predict(self, STA_data, MsgAndCodeDiff):
        tokenizer = tokenizeMsgAndCodeDiff(CB_max_Output_length, self.CodeBertS.tokenizer)
        res_tokens = tokenizer.TokenizeMsgAndCodeDiff(MsgAndCodeDiff)
        with torch.no_grad():
            self.eval()  # Set the model to evaluation mode
            feaCom = self.STAFeaExtract(STA_data)
            if MsgAndCodeDiff is not None:
                msgCode = self.CodeBertS.model(**res_tokens).last_hidden_state
                msgCode_ = msgCode[:, 0, :].squeeze(1)
                msgCode_ = self.convs_code(msgCode_.unsqueeze(1))
                msgCode_ = self.pool_code(msgCode_)
                msgCode_ = self.linear_code(msgCode_.view(-1, 254))
                try:
                    feaCom = torch.cat([feaCom, msgCode_], dim=-1)
                except:
                    print("feaCom:",feaCom.shape)
                    print("msgCode_:",msgCode_.shape)
                feaCom = feaCom.unsqueeze(1)

            transFea = self.Trans_Encoder(feaCom)
            transFea = transFea.squeeze(1)
            linearFea = self.linears(transFea)
            output = self.sigmod(linearFea)
        return output

