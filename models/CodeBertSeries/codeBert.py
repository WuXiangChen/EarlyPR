### 本节是codeBertSeries的超类文件

import torch
import os
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import pathlib
root_path = "models/CodeBertSeries/"
class codeBert(nn.Module):
    def __init__(self, model_name:str = "codeBert-base"):
        super(codeBert, self).__init__()
        # 这里model_name主动的加载文件路径，而不是输入文件路径
        # 检测是否有model_name的文件名存在
        model_path = root_path+model_name
        if not  self.exist_modelFile(model_path):
            raise "No such file or directory: "+model_path
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.config = RobertaConfig.from_pretrained(model_path)
        self.model = RobertaModel.from_pretrained(model_path, config=self.config)
        self.linear_size = [32]

    def exist_modelFile(self, model_path):

        return os.path.exists(model_path)

    # 将初始化的选项放在 模型建立过程中
    def initialize_para_fromScratch(self):
        pass

    # 这里的加载方式还有点不太一样，这里应该定义为从pt模型中加载指定的模型
    def initialize_para_fromPreTrained(self):
        pass

# model = UniXcoder("./unixcoder-base")
# print()