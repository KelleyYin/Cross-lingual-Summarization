#!/bin/bash
ID=5
MODEL_PATH=checkpoints/ch-en-32k-joint/checkpoint_acc_62.42_loss_2.82_ppl_7.07_e37.pt
DATA_RAW=/data/mmyin/data-raw/summ/ch-en-32k-joint

DATA_PATH=/data/mmyin/sum_data/rush_data/mt_ldc/32k-bpe-joint/rush.test.bpe.ch
Pred_txt=checkpoints/ch-en-32k-joint/rush.trans

CUDA_VISIBLE_DEVICES=$ID python interactive.py \
        -s ch -t en \
        --buffer-size 1024 \
        -data $DATA_RAW --path $MODEL_PATH \
        --src_path $DATA_PATH \
        --beam 5 > test.en_
grep ^H test.en_ | cut -f3- > $Pred_txt

#model=checkpoint_acc_57.76_loss_3.10_ppl_8.58_e12.pt
#file_model=checkpoints/rush-5w
#data_raw=/data/mmyin/data-raw/summ/rush-5w
#ref_file=/data/mmyin/sum_data/rush_data/train/test.tgt
#CUDA_VISIBLE_DEVICES=$ID python interactive.py \
#        -s src -t tgt \
#        --buffer-size 1024 \
#        -data $data_raw --path $file_model/$model \
#        --max-len-b 50 \
#        --src_path $Pred_txt \
#        --beam 12 > test.summ_
#grep ^H test.summ_ | cut -f3- > $file_model/rush.test.tgt

#files2rouge $ref_file $file_model/rush.test.tgt

echo "Translation Have Done !"

