
import warnings
from datetime import datetime
from pathlib import Path
# Disable all warnings
warnings.filterwarnings("ignore")
from utils.parseArgument import parser
from transformers import get_linear_schedule_with_warmup
from utils.runUtils import save_ckpt, load_opt_sched_from_ckpt
from datasets.GenerateDataSetForPRTrain import GenerateDataSetForPRTrainAndTest
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
        dataset = GenerateDataSetForPRTrainAndTest(data_file, repo_owner_name_Filepath, Test_ownerSha,
                                                   codeBertS.tokenizer, CB_max_Output_length, trainFlag)
        dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=16)
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(device)


        if args.modelPath != None:
            model, optimizer, scheduler = load_opt_sched_from_ckpt(args.modelPath, optimizer, scheduler, model, device)

        model.to(device)
        output_dir: Path = Path("output/pr_output/"+args.repo_name+"_"+args.output_dir)
        output_dir.mkdir(exist_ok=True)
        for epoch in range(scheduler.last_epoch, args.num_train_epochs):
            train(dataloader, model, device, loss_fn, optimizer, scheduler)
            # TODO: save optimizer and scheduler
            if (epoch + 1) % args.save_steps == 0:
                save_ckpt(output_dir.joinpath(f'epoch_{epoch}d'),model,optimizer,scheduler)

    if args.do_test:
        test(data_file, args.modelPath, args.MPmodelPath, model, optimizer, scheduler, args.test_device, args.mergedPRTest)
if __name__ == '__main__':
    args = parser.parse_args()
    run(args.repo_name, args.codeBertS_name, args.head_define, args.device)