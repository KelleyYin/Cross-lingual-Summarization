#!/bin/bash
ID=0
MODEL_PATH=checkpoints/rush-32k-independent/checkpoint_acc_59.94_loss_2.80_ppl_6.95_e27.pt
DATA_RAW=/data/mmyin/data-raw/summ/rush-32k-joint
src_path=checkpoints/ch-en-32k-independent/

CUDA_VISIBLE_DEVICES=$ID python interactive.py -s src -t tgt --buffer-size 1024 -data $DATA_RAW --path $MODEL_PATH --max-len-b 50 --src_path $src_path --beam 12 > tmp.en_

echo "Translation Have Done !"

