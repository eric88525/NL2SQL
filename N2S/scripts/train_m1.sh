#!/bin/bash
save_by="f1"
exp_name="m1_model"
batch_size="64"
epoch="10"
learning_rate="5e-5"
model_type="voidful/albert_chinese_large"

python train_m1v2.py \
--save_by $save_by \
--exp_name $exp_name \
--epoch $epoch \
--batch_size $batch_size \
--learning_rate $learning_rate \
--model_type $model_type \
