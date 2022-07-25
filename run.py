import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import MyNet
from trainer import Trainer
from dataset import MyDataset
import utils

sep = os.sep

def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)
    # set device
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    # load data
    train_dataset = MyDataset(args.train_dirs)
    valid_dataset = MyDataset(args.valid_dirs)
    test_dataset = MyDataset(args.test_dirs)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    # set model
    net = MyNet()
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    scheduler = None
    # trainer
    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler)
    # train model
    if not args.load_epoch:
        trainer.train(train_dataloader, valid_dataloader)
    # test model
    load_epoch = args.load_epoch if args.load_epoch else 'best'
    model_path = os.path.join(args.writer.log_dir, 'model', f'{load_epoch}_checkpoint.pth.tar')
    if args.dp:
        trainer.net.module.load_state_dict(torch.load(model_path)['model'])
    else:
        trainer.net.load_state_dict(torch.load(model_path)['model'])
    trainer.test(test_dataloader)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'{time_str}-{args.version}' if not args.load_epoch and args.time_version else args.version
    log_dir = f'runs/{args.version}'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save / load version files
    if args.save_version_files: utils.save_load_version_files(log_dir, args.load_epoch, args.save_version_file_patterns)
    # save config file
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))
    # run
    args.writer, args.logger = writer, logger
    args.logger.info(args)
    main(args)
    # restore latest version files
    if args.save_version_files and args.load_epoch:
        utils.copy_files(f'.{sep}runs{sep}latest_project', f'.{sep}', args.save_version_file_patterns, pass_dirs)


if __name__ == '__main__':
    run()