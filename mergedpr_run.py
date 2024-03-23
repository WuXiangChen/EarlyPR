
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path
from utils.parseArgument import parser
from transformers import get_linear_schedule_with_warmup
from utils.runUtils import save_ckpt
from datasets.GenerateDataSetForMPTrain import GenerateDataSetForMPTrainAndTest
from CONSTANT.CONSTANT import *
from CONSTANT.CONSTANT import CB_max_Output_length
from torch.utils.data import DataLoader
from  train import train
import torch
from models.EarlyPRModel import EarlyPRModel
import torch.nn as nn
from utils.runUtils import buildModel
from test import test

def run(data_file, codeBertS_name, head, device):


    STAFeaExtract,codeBertS,modelHead = buildModel(codeBertS_name, head)

    model = EarlyPRModel(STAFeaExtract=STAFeaExtract,CodeBertS=codeBertS,Trans_Encoder=modelHead)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_epochs,
    )
    loss_fn = nn.BCELoss()

    if args.do_train:
        trainFlag = True
        dataset = GenerateDataSetForMPTrainAndTest(data_file, repo_owner_name_Filepath, Test_ownerSha,
                                                   codeBertS.tokenizer, CB_max_Output_length, trainFlag)
        dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True,pin_memory=True, num_workers=8)
        device = torch.device(device)
        model.to(device)
        output_dir: Path = Path("output/mergedpr_output/"+args.repo_name+"_"+args.output_dir)
        output_dir.mkdir(exist_ok=True)
        for epoch in range(scheduler.last_epoch, args.num_train_epochs):
            train(dataloader, model, device, loss_fn, optimizer, scheduler)
            # TODO: save optimizer and scheduler
            if (epoch + 1) % args.save_steps == 0:
                save_ckpt(output_dir.joinpath(f'epoch_{epoch}d'),model,optimizer,scheduler)


if __name__ == '__main__':
    # 这里应当是一个参数的输入
    args = parser.parse_args()
    run(args.repo_name, args.codeBertS_name, args.head_define, args.device)