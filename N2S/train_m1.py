#!/usr/bin/env python
# coding: utf-8
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
from model.m1_model import M1Model
from dataset.m1_dataset import M1Dataset, BatchSampler
from dataset.utils import *


def get_batch_loss(cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label):
    """Caculate loss of agg, cond_conn_op, conds_ops"""
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    loss += loss_fn(cond_conn_op_pred, cond_conn_op_label)*0.33

    # conds_ops_pred shape = (batch_size, columns_count, 5)
    loss += loss_fn(conds_ops_pred.view(-1, 5), conds_ops_label.view(-1))*0.33
    # agg_pred shape = (batch_size, columns_count, 7)
    loss += loss_fn(agg_pred.view(-1, 7), agg_label.view(-1))*0.33

    return loss


def getTime():
    return (datetime.now()+timedelta(hours=8)).strftime("%m/%d %H:%M")


def train(args):

    # train data
    train_data = M1Dataset(args.train_table_file, args.train_data_file)
    train_batch_sampler = BatchSampler(
        train_data.datas, args.model_type, args.device)

    # val data
    val_data = M1Dataset(args.val_table_file, args.val_data_file)
    val_batch_sampler = BatchSampler(
        val_data.datas, args.model_type, args.device)

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

        for _ in tqdm(range(args.samples_in_epoch), ncols=50):

            batch = train_batch_sampler.get_batch(args.batch_size, encode=True)
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

            writer.add_scalar("Train/batch", batch_loss.item(), steps)
            steps += 1

        writer.add_scalar("Train/epoch", epoch_loss /
                          args.samples_in_epoch, epoch)
        val_loss = test(model, val_batch_sampler, round=1000)
        writer.add_scalar("Val/epoch", val_loss, epoch)

        if val_loss < minloss:
            minloss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            print(f"save model at epoch {epoch}, val loss: {val_loss:.3f}")
            torch.save(best_model, f'saved_models/{args.exp_name}')


def test(model, batch_sampler, round):
    """Test round of samples"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(round):
            batch = batch_sampler.get_batch(args.batch_size, encode=True)
            # label
            cond_conn_op_label, agg_label, conds_ops_label = batch[
                'cond_conn_op'], batch['agg'], batch['conds_ops']
            # pred
            cond_conn_op_pred, conds_ops_pred, agg_pred = model(**batch)
            # conn , ops , agg
            batch_loss = get_batch_loss(
                cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label)
            total_loss += batch_loss

    return total_loss/round


def main(args):

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    mode = 'train'

    if mode == 'train':
        train(args)


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
    parser.add_argument('--exp-name', default="albert_v1", type=str)
    parser.add_argument('--samples-in-epoch', default=10000, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--weight-decay', default=0.001, type=float)
    parser.add_argument(
        '--model-type', default='voidful/albert_chinese_large', type=str)
    parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")

    writer.add_text('args', str(args))

    main(args)
