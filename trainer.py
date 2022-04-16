import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from loss import MyLoss
import utils


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = MyLoss().to(self.args.device)

    def train(self, train_loader, valid_loader):
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = np.float32('inf')
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x, target) in train_bar:
                # forward
                x, target = x.float().to(self.args.device), target.float().to(self.args.device)
                out = self.net(x)
                loss = self.criterion(out, target)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # visualization
                self.writer.add_scalar('loss/train_loss', loss.item(), global_step=self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.3f}')
            self.writer.add_scalar('step-epoch', epoch, global_step=self.sum_train_steps)
            # valid
            if epoch % valid_every_epochs == 0:
                metric = self.valid(valid_loader)
                valid_loss = metric['loss']
                if valid_loss <= best_metric:
                    no_better_epoch = 0
                    best_metric = valid_loss
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=self.optimizer)
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save model
            if epoch >= self.args.start_save_model_epochs:
                if (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                    model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                    utils.save_model_state_dict(model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=self.optimizer)

    def valid(self, valid_loader):
        metric = {}
        sum_loss = 0
        num_steps = len(valid_loader)
        self.net.eval()
        valid_bar = tqdm(valid_loader, total=num_steps, desc=f'Valid')
        for (x, target) in valid_bar:
            x, target = x.float().to(self.args.device), target.float().to(self.args.device)
            with torch.no_grad():
                out = self.net(x)
            loss = self.criterion(out, target)
            self.writer.add_scalar('loss/valid_loss', loss.item(), global_step=self.sum_valid_steps)
            sum_loss += loss.item()
            self.sum_valid_steps += 1
        avg_loss = sum_loss / num_steps
        metric['loss'] = avg_loss
        self.logger.info(f'valid loss: {avg_loss:.3f}')
        return metric

    def test(self, test_loader):
        pass
