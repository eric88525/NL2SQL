#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from datetime import datetime, timedelta
import copy
import os
from torch.utils.tensorboard import SummaryWriter
from model.m1v2_model import M1Model
from dataset.m1v2_dataset import M1Dataset
from dataset.utils import *
from torch.utils.data import DataLoader


def get_batch_loss(cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label):
    """Caculate loss of agg, cond_conn_op, conds_ops"""
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    loss += loss_fn(cond_conn_op_pred, cond_conn_op_label)*0.33
    loss += loss_fn(conds_ops_pred.view(-1, 5), conds_ops_label)*0.33
    loss += loss_fn(agg_pred.view(-1, 7), agg_label)*0.33

    return loss


def getTime():
    return (datetime.now()+timedelta(hours=8)).strftime("%m/%d %H:%M")


def train(args):

    # train data
    train_data = M1Dataset(args.train_table_file,
                           args.train_data_file, args.model_type)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, collate_fn=lambda b: M1Dataset.collate_fn(b, train_data.tokenizer))

    # val data
    val_data = M1Dataset(args.val_table_file,
                         args.val_data_file, args.model_type)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, collate_fn=lambda b: M1Dataset.collate_fn(b, val_data.tokenizer))

    # model
    model = M1Model(args.model_type).to(args.device)

    # parameter & opt
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    minloss = 100000
    steps = 0

    # 1 epoch = train model with epoch_samples batchs
    for epoch in range(args.epoch):
        print(f"epoch: {epoch} time: {getTime()}")

        epoch_loss = 0
        model.train()
        tqdm_loader = tqdm(train_loader, ncols=50)
        for batch in tqdm_loader:

            for k in batch.keys():
                batch[k] = batch[k].to(args.device)

            cond_conn_op_label, agg_label, conds_ops_label = batch[
                'cond_conn_op'], batch['agg'], batch['conds_ops']
            # pred
            cond_conn_op_pred, conds_ops_pred, agg_pred = model(**batch)
            # conn , ops , agg
            batch_loss = get_batch_loss(
                cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label)

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tqdm_loader.set_description(f'loss = {batch_loss.item():.3f}')
            writer.add_scalar("Train/batch", batch_loss.item(), steps)
            steps += 1

        writer.add_scalar("Train/epoch", epoch_loss /
                          len(train_loader), epoch)
        val_loss = test(model, val_loader)
        writer.add_scalar("Val/epoch", val_loss, epoch)

        if val_loss < minloss:
            minloss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            print(f"save model at epoch {epoch}, dev loss: {val_loss:.3f}")
            torch.save(best_model, f'saved_models/{args.exp_name}')


def test(model, loader):
    """Test"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:

            for k in batch.keys():
                batch[k] = batch[k].to(args.device)

            # label
            cond_conn_op_label, agg_label, conds_ops_label = batch[
                'cond_conn_op'], batch['agg'], batch['conds_ops']
            # pred
            cond_conn_op_pred, conds_ops_pred, agg_pred = model(**batch)
            # conn , ops , agg
            batch_loss = get_batch_loss(
                cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label)
            total_loss += batch_loss

    return total_loss/ len(loader)


def main(args):

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    mode = ('train', 'test')

    if 'train' in mode:
        train(args)

    if 'test' in mode:
        test_data = M1Dataset(args.test_table_file,
                              args.test_data_file, args.model_type)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 pin_memory=True, collate_fn=lambda b: M1Dataset.collate_fn(b, test_data.tokenizer))
        model = M1Model(args.model_type).to(args.device)
        model.load_state_dict(torch.load(f'saved_models/{args.exp_name}'))
        test_loss = test(model, test_loader)
        writer.add_scalar("Test/epoch", test_loss)


# In[ ]:
if __name__ == '__main__':

    seed_all(2022)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    parser = argparse.ArgumentParser()

    # data files
    parser.add_argument('--train_table_file',
                        default='./data/train/train.tables.json', type=str)
    parser.add_argument('--train_data_file',
                        default='./data/train/train.json', type=str)
    parser.add_argument('--val_table_file',
                        default='./data/val/val.tables.json', type=str)
    parser.add_argument('--val_data_file',
                        default='./data/val/val.json', type=str)
    parser.add_argument('--test_table_file',
                        default='./data/test/test.tables.json', type=str)
    parser.add_argument('--test_data_file',
                        default='./data/test/test.json', type=str)

    # train args
    parser.add_argument(
        '--exp-name', default="M1v2_albert_chinese_large_v1", type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--learning-rate', default=2e-5, type=float)
    parser.add_argument('--weight-decay', default=0.001, type=float)
    parser.add_argument(
        '--model-type', default='voidful/albert_chinese_large', type=str)
    parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")

    writer.add_text('args', str(args))
    main(args)
