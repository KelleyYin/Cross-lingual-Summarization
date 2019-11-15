# Introduction

## Preprocesing
Binary the dataset. 

```
DATA_PATH=/path/to/data/file
DATA_BIN=/path/to/save/data-bin
SRC=
TGT=

python preprocess.py -s $SRC -t $TGT \
		--trainpref $DATA_PATH/train \
		--validpref $DATA_PATH/valid \
		--destdir $DDATA_BIN \
		--output-format binary \
```

## Training new model


```
DATA_BIN=
SAVE_FILE=
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python trian.py -data $DATA_BIN \
		-s $SRC -t $TGT \
		--lr 0.0005 --min-lr 1e-09 \
		--weight-decay 0 --clip-norm 0.0 \
		--dropout 0.3 \
		--max-tokens 4500 \
		--arch transformer \
		--optimizer adam --adam-betas '(0.9, 0.98)' \
		--warmup-updates 4000 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--save-dir $SAVE_FILE
```

NOTICES:    

* if 




# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

Currently fairseq requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size` as command line
options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

## Training a New Model

The following tutorial is for machine translation.
For an example of how to use Fairseq for other tasks, such as [language modeling](examples/language_model/README.md), please see the `examples/` directory.

### Data Pre-processing

Fairseq contains example pre-processing scripts for several translation datasets: IWSLT 2014 (German-English), WMT 2014 (English-French) and WMT 2014 (English-German).
To pre-process and binarize the IWSLT dataset:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to `data-bin/iwslt14.tokenized.de-en`.

### Training
Use `python train.py` to train a new model.
Here a few example settings that work well for the IWSLT 2014 dataset:
```
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

By default, `python train.py` will use all available GPUs on your machine.
Use the [CUDA_VISIBLE_DEVICES](http://acceleware.com/blog/cudavisibledevices-masking-gpus) environment variable to select specific GPUs and/or to change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum number of tokens per batch (`--max-tokens`).
You may need to use a smaller value depending on the available GPU memory on your system.

### Generation
Once your model is trained, you can generate translations using `python generate.py` **(for binarized data)** or `python interactive.py` **(for raw text)**:
```
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5
  | [de] dictionary: 35475 types
  | [en] dictionary: 24739 types
  | data-bin/iwslt14.tokenized.de-en test 6750 examples
  | model fconv
  | loaded checkpoint trainings/fconv/checkpoint_best.pt
  S-721   danke .
  T-721   thank you .
  ...
```

To generate translations with only a CPU, use the `--cpu` flag.
BPE continuation markers can be removed with the `--remove-bpe` flag.



### Usage

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Scoring with score.py:
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

# Large mini-batch training with delayed updates

The `--update-freq` option can be used to accumulate gradients from multiple mini-batches and delay updating,
creating a larger effective batch size.
Delayed updates can also improve training speed by reducing inter-GPU communication costs and by saving idle time caused by variance in workload across GPUs.
See [Ott et al. (2018)](https://arxiv.org/abs/1806.00187) for more details.

To train on a single GPU with an effective batch size that is equivalent to training on 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0 python train.py --update-freq 8 (...)
```

# Training with half precision floating point (FP16)

> Note: FP16 training requires a Volta GPU and CUDA 9.1 or greater

Recent GPUs enable efficient half precision floating point computation, e.g., using [Nvidia Tensor Cores](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

Fairseq supports FP16 training with the `--fp16` flag:
```
python train.py --fp16 (...)
```

# Distributed training

Distributed training in fairseq is implemented on top of [torch.distributed](http://pytorch.org/docs/master/distributed.html).
Training begins by launching one worker process per GPU.
These workers discover each other via a unique host and port (required) that can be used to establish an initial connection.
Additionally, each worker has a rank, that is a unique number from 0 to n-1 where n is the total number of GPUs.

If you run on a cluster managed by [SLURM](https://slurm.schedmd.com/) you can train a large English-French model on the WMT 2014 dataset on 16 nodes with 8 GPUs each (in total 128 GPUs) using this command:

```
$ DATA=...   # path to the preprocessed dataset, must be visible from all nodes
$ PORT=9218  # any available TCP port that can be used by the trainer to establish initial connection
$ sbatch --job-name fairseq-py --gres gpu:8 --cpus-per-task 10 \
    --nodes 16 --ntasks-per-node 8 \
    --wrap 'srun --output train.log.node%t --error train.stderr.node%t.%j \
    python train.py $DATA \
    --distributed-world-size 128 \
    --distributed-port $PORT \
    --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
    --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
    --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --wd 0.0001'
```

Alternatively you can manually start one process per GPU:
```
$ DATA=...  # path to the preprocessed dataset, must be visible from all nodes
$ HOST_PORT=master.devserver.com:9218  # one of the hosts used by the job
$ RANK=...  # the rank of this process, from 0 to 127 in case of 128 GPUs
$ python train.py $DATA \
    --distributed-world-size 128 \
    --distributed-init-method 'tcp://$HOST_PORT' \
    --distributed-rank $RANK \
    --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
    --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
    --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --wd 0.0001
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# Citation

If you use the code in your paper, then please cite it as:

```
@inproceedings{gehring2017convs2s,
  author    = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title     = "{Convolutional Sequence to Sequence Learning}",
  booktitle = {Proc. of ICML},
  year      = 2017,
}
```

# License
fairseq(-py) is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

# Credits
This is a PyTorch version of [fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence learning toolkit from Facebook AI Research. The original authors of this reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam Gross.
