#!/bin/bash

exp_name="m2_model"
batch_size="8"
epoch="20"
learning_rate="2e-5"
model_type="voidful/albert_chinese_large"

python train_m2.py \
--exp_name $exp_name \
--batch_size $batch_size \
--epoch $epoch \
--learning_rate $learning_rate \
--model_type $model_type