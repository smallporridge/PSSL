# PSSL
Source code of CIKM2021 Long Paper "PSSL: Self-supervised Learning for Personalized Search with Contrastive Sampling". It consists of the pre-training stage `./pretrain` and the fine-tuning stage `./finetune`.

## Requirements
I test the code with the following packages. Other versions may also work, but I'm not sure. <br>
- Python 3.6 <br>
- Pytorch 1.3.1 (with GPU support)

## Usage
### pre-training
At the pre-training stage, we first extract data pairs from query logs, as shown in `/datasample/***_pair.txt`. Then, we train the parameters with the contrastive learning framework by:
```
python prepare_load.py
```

### fine-tuning
We initial the encoders with pre-trained parameters and train the ranking model by:
```
python dataset_new.py
```

## Citations
If you use the code, please cite the following paper:  
```
@inproceedings{ZhouDYW21,
  author    = {Yujia Zhou and
               Zhicheng Dou and
               Yutao Zhu and
               Ji{-}Rong Wen},
  title     = {PSSL: Self-supervised Learning for Personalized Search with Contrastive Sampling},
  booktitle = {{CIKM}},
  publisher = {{ACM}},
  year      = {2021}
}
