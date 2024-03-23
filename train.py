from utils.trainUtils import *
import time
from torch import nn
import torch


def train(dataloader,
              model,
              device,
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler
          ):
    """
    Train the model.
    """
    start_time: float = time.time()
    model.train()
    total_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    total_loss: float = 0.0
    for batch_idx, (data_tensor, res_token, label_tensor) in enumerate(dataloader):
        data_tensor=data_tensor.to(device)

        for key in res_token:
            res_token[key] = res_token[key].to(device)

        label = label_tensor.unsqueeze(1).to(device)

        optimizer.zero_grad()
        # forward
        pred = model(data_tensor, res_token)
        loss = loss_fn(pred, label)
        total_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()

        # output log
        if (batch_idx + 1) % 10 == 0:
            loss, current = loss.item(), min((batch_idx + 1) * batch_size, total_size)
            print(f'loss: {loss:>7f} [{current:>5d}/{total_size:>5d}]')

    scheduler.step()
    end_time: float = time.time()
    print(f'This epoch training time: {end_time - start_time}s, total loss: {total_loss:.4f}')




