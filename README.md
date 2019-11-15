# Cross-Lingual Abstractive Sentence Summarization(CL-ASSUM)
## Introduction
We implemented CL-ASSUM on [fairseq](https://github.com/pytorch/fairseq/). 
In this repo, it contains of four parts.

- Transformer
- Teaching-Generation
- Teaching-Attention
- Teaching-Generation-Attention

### Teacher models
Before staring the experiment, you should first use [Transformer](Transformer) to train the teacher model of NMT model and momolingual summarization model.

Please refer to [Transformer](Transformer) for more deatils.

### Teaching-Generation

### Teaching-Attention

### Teaching-Generation-Attention

   
## Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* PyTorch version >= 0.4.0.

## Cross-Lingual Test Set
In our experiments, we manually translate the English sentences into the Chinese sentences for the validation and evaluation sets of Gigaword and DUC2004.    


## License


## Reference
If you find CL-ASSUM useful in your work, you can cite this paper as below:

```
@inproceedings{duan-etal-2019-zero,
    title = "Zero-Shot Cross-Lingual Abstractive Sentence Summarization through Teaching Generation and Attention",
    author = "Duan, Xiangyu  and Yin, Mingming  and Zhang, Min  and Chen, Boxing  and Luo, Weihua",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1305",
    doi = "10.18653/v1/P19-1305",
    pages = "3162--3172",
   }
```
