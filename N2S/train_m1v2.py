#!/usr/bin/env python
# coding: utf-8
from email.policy import default
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
from sklearn.metrics import f1_score


def get_batch_loss(cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label):
    """Caculate loss of agg, cond_conn_op, conds_ops"""
    loss_fn = nn.CrossEntropyLoss()

    # train agg label counts = [41080, 510, 376, 245, 2902, 1007, 282311]
    # weight = 1 - label_counts / sum(label_counts)
    weight = torch.tensor(
        [0.87492046, 0.99844716, 0.99885516, 0.99925403,
            0.99116405, 0.99693391, 0.14042523],
        device=cond_conn_op_pred.device)

    agg_loss_fn = nn.CrossEntropyLoss(weight=weight)

    loss = 0
    loss += loss_fn(cond_conn_op_pred, cond_conn_op_label)*0.1
    loss += loss_fn(conds_ops_pred.view(-1, 5), conds_ops_label)*0.4

    loss += agg_loss_fn(agg_pred.view(-1, 7), agg_label)*0.5

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

    if args.save_by == 'loss':
        bestscore = 10000
    elif args.save_by == 'f1':
        bestscore = 0

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

        if args.save_by == 'loss':  # save model by validation loss
            val_loss = test(model, val_loader)
            writer.add_scalar("Val/epoch", val_loss, epoch)

            if val_loss < bestscore:  # loss smaller is better
                bestscore = val_loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, f'saved_models/{args.exp_name}')
                print(f"save model at epoch {epoch}, dev loss: {val_loss:.3f}")

        elif args.save_by == 'f1':  # save model by validation f1 score
            val_f1 = test_f1(model, val_loader)
            writer.add_scalar("Val/conn_f1", val_f1['conn_f1'], epoch)
            writer.add_scalar("Val/ops_f1", val_f1['ops_f1'], epoch)
            writer.add_scalar("Val/agg_f1", val_f1['agg_f1'], epoch)
            writer.add_scalar("Val/mean_f1", val_f1['mean_f1'],  epoch)

            if val_f1['mean_f1'] > bestscore:  # f1 score bigger is better
                bestscore = val_f1['mean_f1']
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, f'saved_models/{args.exp_name}')

                print(f"save model at epoch {epoch}")
                print(f"agg_f1: {val_f1['agg_f1']:.3f}")
                print(f"conn_f1: {val_f1['conn_f1']:.3f}")
                print(f"ops_f1: {val_f1['ops_f1']:.3f}")
                print(f"mean_f1: {val_f1['mean_f1']:.3f}")

def test_f1(model, loader):
    """Test and return f1 score"""

    model.eval()
    # condition connect operators: ['', 'AND', 'OR']
    conn_f1, conn_pred_list, conn_label_list = 0, [], []

    # condition operator: ['>', '<', '=', '!=', '']
    ops_f1, ops_pred_list, ops_label_list = 0, [], []

    # agg: ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM', 'not select this column']
    agg_f1, agg_pred_list, agg_label_list = 0, [], []

    with torch.no_grad():
        for batch in loader:
            for k in ['input_ids', 'attention_mask', 'token_type_ids', 'header_idx']:
                batch[k] = batch[k].to(next(model.parameters()).device)

            # pred
            conn_pred, ops_pred, agg_pred = model(**batch)

            conn_pred_list.extend(conn_pred.argmax(
                dim=-1).cpu().numpy().tolist())
            ops_pred_list.extend(ops_pred.argmax(
                dim=-1).cpu().numpy().tolist())
            agg_pred_list.extend(agg_pred.argmax(
                dim=-1).cpu().numpy().tolist())

            conn_label_list.extend(batch['cond_conn_op'].tolist())
            ops_label_list.extend(batch['conds_ops'].tolist())
            agg_label_list.extend(batch['agg'].tolist())

    conn_f1 = f1_score(conn_label_list, conn_pred_list, average='macro')
    ops_f1 = f1_score(ops_label_list, ops_pred_list, average='macro')
    agg_f1 = f1_score(agg_label_list, agg_pred_list, average='macro')

    f1_scores = {
        'conn_f1': conn_f1,
        'ops_f1': ops_f1,
        'agg_f1': agg_f1,
        'mean_f1': (conn_f1 + ops_f1 + agg_f1) / 3,
    }

    return f1_scores


def test(model, loader):
    """Test and return loss"""

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:

            for k in batch.keys():
                batch[k] = batch[k].to(next(model.parameters()).device)

            # label
            cond_conn_op_label, agg_label, conds_ops_label = batch[
                'cond_conn_op'], batch['agg'], batch['conds_ops']
            # pred
            cond_conn_op_pred, conds_ops_pred, agg_pred = model(**batch)
            # conn , ops , agg
            batch_loss = get_batch_loss(
                cond_conn_op_pred, cond_conn_op_label, conds_ops_pred, conds_ops_label, agg_pred, agg_label)
            total_loss += batch_loss

    return total_loss / len(loader)


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

        if args.save_by == 'loss':
            test_loss = test(model, test_loader)
            writer.add_scalar("Test/epoch", test_loss)
        elif args.save_by == 'f1':
            test_f1 = test_f1(model, test_loader)
            writer.add_text(f"agg_f1: {test_f1['agg_f1']:.3f}, \
                    conn_f1: {test_f1['conn_f1']:.3f}, \
                    ops_f1: {test_f1['ops_f1']:.3f}, \
                    mean_f1: {test_f1['mean_f1']:.3f}")


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
        '--save_by', choices=['loss', 'f1'], default='f1', type=str)
    parser.add_argument(
        '--exp_name', default="M1v2_chinese-roberta-wwm-ext_byf1", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument(  # voidful/albert_chinese_large
        '--model_type', default='hfl/chinese-roberta-wwm-ext', type=str)
    parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")
    writer.add_text('args', str(args))

    main(args)
