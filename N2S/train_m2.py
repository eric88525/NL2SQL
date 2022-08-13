import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import random
import gc
import copy
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime, timedelta
from dataset.utils import *
from dataset.m2_dataset import M2Dataset
from model.m2_model import M2Model


def get_loader(dataset, args, weighted_sampler: bool):

    if weighted_sampler:
        print("[info] train data with weighted sampler")

        class_count = np.unique(dataset.labels, return_counts=True)[1]
        weights = 1 / class_count
        samples_weight = torch.tensor([weights[t] for t in dataset.labels])
        sampler = WeightedRandomSampler(
            samples_weight, len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)


def getTime():
    return (datetime.now()+timedelta(hours=8)).strftime("%m/%d %H:%M")


def train(args):

    train_data = M2Dataset(args.train_table_file,
                           args.train_data_file, args.model_type)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                              num_workers=8, collate_fn=lambda b: M2Dataset.collate_fn(b, train_data.tokenizer))

    val_data = M2Dataset(args.val_table_file,
                         args.val_data_file, args.model_type)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, collate_fn=lambda b: M2Dataset.collate_fn(b, val_data.tokenizer))

    # model
    model = M2Model(args.model_type).to(args.device)

    # parameter & opt
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    minloss = 100000
    loss_fn = nn.BCELoss()
    steps = 0

    for epoch in range(args.epoch):
        print(f'epoch{epoch} {getTime()}')
        epoch_loss = 0
        model.train()

        for batch in tqdm(train_loader, ncols=50):

            for k in batch.keys():
                batch[k] = batch[k].to(args.device)

            label = batch['label']
            pred = model(**batch)

            batch_loss = loss_fn(pred, label)
            epoch_loss += batch_loss.item()

            writer.add_scalar("Train/batch", batch_loss.item(), steps)
            steps += 1
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        writer.add_scalar("Train/epoch", epoch_loss / len(train_loader), epoch)

        try:
            val_loss = test(model, val_loader)
            writer.add_scalar("Val/epoch", val_loss, epoch)
            if val_loss < minloss:
                minloss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, f'saved_models/{args.exp_name}')
        except Exception as e:
            print(e)
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, f'saved_models/{args.exp_name}')

        print(f'Train/Epoches:{epoch} Loss {epoch_loss/len(train_loader)}')


def test(model, data_loader):

    model.eval()

    loss_fn = nn.BCELoss()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            for k in batch.keys():
                batch[k] = batch[k].to(args.device)

            label = batch['label']
            pred = model(**batch)

            batch_loss = loss_fn(pred, label)
            total_loss += batch_loss.item()

    print(f'Test/LOSS: {total_loss} AVG: {total_loss/len(data_loader)}')

    return total_loss / len(data_loader)


def main(args):
    mode = set(['train', 'test'])

    if 'train' in mode:
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        train(args)

    if 'test' in mode:
        test_data = M2Dataset(args.test_table_file,
                              args.test_data_file, args.model_type)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=8, collate_fn=lambda b: M2Dataset.collate_fn(b, test_data.tokenizer))
        model = M2Model(args.model_type).to(args.device)
        model.load_state_dict(torch.load(f'saved_models/{args.exp_name}'))
        test_loss = test(model, test_loader)
        writer.add_scalar("Test/epoch", test_loss)


if __name__ == "__main__":

    seed_all(2022)

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

    parser.add_argument(
        '--exp-name', default="M2_albert_chinese_large_v1", type=str)
    parser.add_argument('--batch-size', default=8, type=int)
    # check_num
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--learning-rate', default=2e-5, type=float)
    parser.add_argument('--weight-decay', default=0.001, type=float)

    # model_type = 'hfl/chinese-bert-wwm'  'hfl/chinese-roberta-wwm-ext'  'hfl/chinese-roberta-wwm-ext-large'
    parser.add_argument(
        '--model-type', default='voidful/albert_chinese_large', type=str)

    parser.add_argument('--device', default=torch.device('cuda:0'), type=int)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=f"./runs/{args.exp_name}")
    # debug()
    main(args)
