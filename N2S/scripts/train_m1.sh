#!/bin/bash
save_by="f1"
exp_name="m1_model"
# batch size 64 takes about 16GB GPU memory
batch_size="64"
epoch="10"
learning_rate="5e-5"
model_type="hfl/chinese-roberta-wwm-ext"

python train_m1v2.py \
--save_by $save_by \
--exp_name $exp_name \
--epoch $epoch \
--batch_size $batch_size \
--learning_rate $learning_rate \
--model_type $model_type \
