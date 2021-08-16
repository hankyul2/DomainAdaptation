# Domain Adaptation

This repository contains DANN, CDAN, BSP pytorch source code (not official, just my study) Domain Adaptation papers are really difficult for beginners. So I want to give some intuition of how it works. Use it for your start points. Thanks



We will update basic idea about domain adaptation



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

   



### Benchmark Result (Closed world)

*Office-31 Dataset Benchmark done in this work(2021)*

|                                           | A -> D | A -> W | D -> A | D -> W | W -> A | W -> D | Avg  |
| ----------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ---- |
| source only (pretrained on ImageNet 2012) |        |        |        |        |        |        |      |
| DANN (2016)                               |        |        |        |        |        |        |      |
| CDAN (2018)                               |        |        |        |        |        |        |      |
| DANN + BSP (2019)                         |        |        |        |        |        |        |      |
| CDAN + BSP (2019)                         |        |        |        |        |        |        |      |
| SHOT (2021)                               |        |        |        |        |        |        |      |



*Office-31 Dataset Benchmark reported in original papers*

|                                        | A -> D | A -> W | D -> A | D -> W | W -> A | W -> D | Avg  |
| -------------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ---- |
| source only (use reported from SHOT) * | 80.8   | 76.9   | 60.3   | 95.3   | 63.6   | 98.7   | 79.3 |
| DANN (2016) *                          | 79.7   | 82.0   | 68.2   | 96.9   | 67.4   | 99.1   | 82.2 |
| CDAN (2018)                            |        |        |        |        |        |        |      |
| CDAN+E (2018)                          |        |        |        |        |        |        |      |
| CDAN+BSP (2018)                        |        |        |        |        |        |        |      |
| DANN + BSP (2019)                      | 90.0   | 93.0   | 71.9   | 98.0   | 73.0   | 100.0  | 87.7 |
| CDAN + BSP (2019)                      | 93.0   | 93.3   | 73.6   | 98.2   | 72.6   | 100.0  | 88.5 |
| SHOT (2021)                            | 94.0   | 90.1   | 74.7   | 98.4   | 74.3   | 99.9   | 88.6 |

Things you should know

1. Source only results score are from SHOT paper, not original source only score.
2. Original DANN(2016) scores are not used in here, because backbone are updated since 2016. The scores used here are reported from some papers (BSP, SHOT)





### Further work

- [ ] Prepare Dataset
  - [x] Office-31
  - [x] Office-Home : [공식 홈페이지](https://www.hemanthdv.org/officeHomeDataset.html) 에서 다운로드 가능
  - [x] Digits (모두 torchvision datasets에서 다운로드 가능)
    - [x] SVHN
    - [x] MNIST
    - [x] USPS
- [ ] Source Only
- [ ] DANN 
- [ ] CDAN
- [ ] DANN + BSP
- [ ] CDAN + BSP
- [ ] +Entropy
- [ ] +10-crop ensemble

