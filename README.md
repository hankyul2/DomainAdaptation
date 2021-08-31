# Domain Adaptation

This repository contains DANN, CDAN, BSP pytorch source code (not official, just my study) Domain Adaptation papers are really difficult for beginners. So I want to give some intuition of how it works. Use it for your start points. Thanks



I will update basic idea about domain adaptation



### Tutorial (use standard dataset: Office-31, Office-home, Digits)

1. Clone repository and install dependency

   ```bash
   git clone https://github.com/hankyul2/DomainAdaptation.git
   pip3 install requirements.txt
   ```

2. Download Dataset 

   1. Use download script

      ```bash
      # cloned repository
      python3 download_dataset.py
      ```

   2. Download Dataset yourself and put them in data directory

      1. Office-31 (save under `data/office_31`)

         Go to [jindongwang/transferlearning](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md), this repository summarizes how to download various datasets, including office-31.

      2. Office-home (save under `data/office_home`)

         Go to [official home page](https://www.hemanthdv.org/officeHomeDataset.html). 

      3. Digits (save under `data/MNIST`, `data/SVHN`, `data/USPS`)

         You can download from torchvision. It will automatically download SVHN, MNIST, USPS.

3. Train 

   ```bash
   python3 main.py --gpu_id --src a --tgt w
   ```

   



### Closed world Benchmark Result

- *italic number* means score lower than official score
- numbers in each block: (official score) / (in this work score)
- All model are based on *resnet50* pretrained on *ILSVRC2012*
- baseline's gpu memory usage is about 5GB, others gpu memory usages is about 8GB
- all scores are 3 times averaged scores
- (score not matching with official one) / (all score) ratio: 16 / 36

*Office-31 Dataset Benchmark reported in original papers*

|                   | A -> D                  | A -> W                  | D -> A                  | D -> W                  | W -> A                  | W -> D                   | Avg           |
| ----------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ------------------------ | ------------- |
| source only*      | 80.8 / *79.2**(-1.6)*** | 76.9 / *76.4**(-0.5)*** | 60.3 / 64.8             | 95.3 / 96.7             | 63.6 / 64.8             | 98.7 / 99.0              | 79.3 / 80.15  |
| DANN (2016) *     | 79.7 / 82.5             | 82.0 / 83.4             | 68.2 / *65.5**(-2.7)*** | 96.9 / 98.3             | 67.4 / *65.5**(-1.9)*** | 99.1 / 100.0             | 82.2 / 82.5   |
| CDAN (2018)       | 89.8 / *87.6**(-1.8)*** | 93.1 / *92.0**(-1.1)*** | 70.1 / *68.8**(-1.3)*** | 98.2 / 98.3             | 68.0 / 69.3             | 100.0 / 100.0            | 86.6 / *86.0* |
| CDAN+E (2018)     | 92.9 / *87.3**(-5.6)*** | 94.1 / *90.5**(-3.6)*** | 71.0 / 72.4             | 98.6 / 98.6             | 69.3 / 71.0             | 100.0 / 100.0            | 87.7 / *86.6* |
| DANN+BSP (2019)   | 90.0 / *82.8**(-7.2)*** | 93.0 / *82.8**(-4.2)*** | 71.9 / *64.6**(-7.3)*** | 98.0 / *97.3**(-0.7)*** | 73.0 / *67.4**(-6.6)*** | *100.0 /* 99.7**(-0.3)** | 87.7 / 82.4   |
| CDAN+BSP (2019)   | / 89.3                  | / 91.6                  | / 74.1                  | / 97.9                  | / 74.0                  | / 99.8                   | / 87.8        |
| CDAN+E+BSP (2019) | 93.0 / *89.0**(-4.0)*** | 93.3 / *91.3**(-2.0)*** | 73.6 / 74.1             | 98.2 / 98.5             | 72.6 / 73.9             | 100.0 / 100.0            | 88.5 / *87.8* |
| SHOT (2021)       | 94.0                    | 90.1                    | 74.7                    | 98.4                    | 74.3                    | 99.9                     | 88.6          |

*Note*

1. Source only results score are from SHOT paper, not original source only score.
2. Original DANN(2016) scores are not used in here, because backbone are updated since 2016. The scores used here are reported from some papers (BSP, SHOT)





### Todo List

- [x] Prepare Dataset
  - [x] Office-31
  - [x] Office-Home : [공식 홈페이지](https://www.hemanthdv.org/officeHomeDataset.html) 에서 다운로드 가능
  - [x] Digits (모두 torchvision datasets에서 다운로드 가능)
    - [x] SVHN
    - [x] MNIST
    - [x] USPS
- [x] Source Only
- [x] DANN 
- [x] CDAN
- [x] DANN + BSP
- [x] CDAN + BSP
- [x] +Entropy
- [x] +10-crop ensemble
- [ ] Fixbi
- [ ] SHOT
- [ ] TransDA



### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)
- [thuml/CDAN](https://github.com/thuml/CDAN)
- [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

- [thuml/Batch-Spectral-Penalization](https://github.com/thuml/Batch-Spectral-Penalization)
- [Transferability vs Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)