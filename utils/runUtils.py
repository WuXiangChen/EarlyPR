import os
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple
import torch.nn as nn
import torch

def save_ckpt(
    ckpt_output_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    """
    Save checkpoint.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    ckpt_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_output_dir.joinpath('model.pt'))
    # model.config.to_json_file(ckpt_output_dir.joinpath('config.json'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_output_dir.joinpath('optimizer.pt'))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_output_dir.joinpath('scheduler.pt'))

def load_opt_sched_from_ckpt(
    ckpt_output_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    model=None,
    device: Union[torch.device, str] = 'cpu'
):
    """
    Load checkpoint.

    Args:
        ckpt_output_dir (Union[str, Path]): The directory path where the checkpoint is saved.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer to be loaded. Defaults to None.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler], optional): The scheduler to be loaded. Defaults to None.
        device (Union[torch.device, str], optional): The device to be used for loading. Defaults to 'cpu'.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    if model is not None:
        model.load_state_dict(torch.load(ckpt_output_dir.joinpath('model.pt'), map_location=device))
        model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(ckpt_output_dir.joinpath('optimizer.pt'), map_location=device))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(ckpt_output_dir.joinpath('scheduler.pt'), map_location=device))
    return model, optimizer, scheduler

def load_opt_sched_from_ckptTest(
    ckpt_output_dir: Union[str, Path],
    model=None,
    device: Union[torch.device, str] = 'cpu'
):
    """
    Load checkpoint.

    Args:
        ckpt_output_dir (Union[str, Path]): The directory path where the checkpoint is saved.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer to be loaded. Defaults to None.
        scheduler (Optional[torch.optim.lr_scheduler.LRScheduler], optional): The scheduler to be loaded. Defaults to None.
        device (Union[torch.device, str], optional): The device to be used for loading. Defaults to 'cpu'.
    """
    ckpt_output_dir: Path = Path(ckpt_output_dir)
    if model is not None:
        model.load_state_dict(torch.load(ckpt_output_dir.joinpath('model.pt'), map_location=device))
        model.to(device)
    return model

from models.Head.transformerEncode import transHead
from models.CodeBertSeries import codeBert
from models.Embedding import FeatureShrink
def buildModel(codeBertS_name,head):
    STAFeaExtract = FeatureShrink()
    codeBertS = codeBert(codeBertS_name)
    # transformer 比较麻烦的是有一堆的参数需要进行定义
    modelHead = eval(head)()
    return STAFeaExtract,codeBertS,modelHead

import pandas as pd
def update_result(result, new_data):
    if new_data is not None:
        if result is None:
            result = new_data
        else:
            # 按列合并
            result = pd.concat([result, new_data], axis=0)
    return result


