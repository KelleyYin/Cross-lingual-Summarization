num=5

MODEL_PATH=checkpoints/ch-en-32k-joint-pre/checkpoint_last.pt
data_dir=/data/mmyin/data-raw/summ/ch-en-32k-joint

ref_file=/data/mmyin/sum_data/rush_data/mt_ldc
src_file=$ref_file/32k-bpe-joint

test_set='nist02 nist03 nist04 nist05 nist08'
for test in $test_set;do
    echo $test'.in'
    CUDA_VISIBLE_DEVICES=$num python interactive.py -data $data_dir\
    --buffer-size 1024 --path $MODEL_PATH\
    --beam 5\
    -s ch -t en \
    --remove-bpe \
    --src_path $src_file/$test'.bpe.in' > $test'_'
    grep ^H $test'_' | cut -f3- > $test'.txt'
    perl ~/tools/multi-bleu.perl -lc $ref_file/$test'.#.ref.' < $test'.txt'
done

echo "Translation Have Done !"






