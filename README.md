# Domain Adaptation

This repository contains pytorch version source code introduced by domain adaptation paper:

1. DANN (2015) [[paper](http://proceedings.mlr.press/v37/ganin15.html), [repo](https://github.com/fungtion/DANN)] 
2. CDAN (2017) [[paper](https://arxiv.org/abs/1705.10667), [repo](https://github.com/thuml/CDAN)]
3. MSTN (2018) [[paper](https://proceedings.mlr.press/v80/xie18c.html), [repo](https://github.com/wgchang/DSBN)]
4. BSP (2019) [[paper](http://proceedings.mlr.press/v97/chen19i.html), [repo](https://github.com/thuml/Batch-Spectral-Penalization)]
5. DSBN (2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html), [repo](https://github.com/wgchang/DSBN)]
6. RSDA-MSTN (2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Spherical_Space_Domain_Adaptation_With_Robust_Pseudo-Label_Loss_CVPR_2020_paper.html), [repo](https://github.com/XJTU-XGU/RSDA)]
7. SHOT (2020) [[paper](http://proceedings.mlr.press/v119/liang20a.html), [repo](https://github.com/tim-learn/SHOT)]
8. TransDA (2021) [[paper](https://arxiv.org/abs/2105.14138), [repo](https://github.com/ygjwd12345/TransDA)]
9. FixBi (2021) [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Na_FixBi_Bridging_Domain_Spaces_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.html) ,[repo](https://github.com/NaJaeMin92/FixBi)]



### Tutorial (use standard dataset: Office-31)

1. Clone repository and install dependency

   ```bash
   git clone https://github.com/hankyul2/DomainAdaptation.git
   pip3 install -r requirements.txt
   ```

2. Train Model (If you don't use neptune, just remove it from [here](https://github.com/hankyul2/DomainAdaptation/blob/3af6a9ee3848ef3757c63fcf3f0083757e1a4564/src/cli.py#L82)). Check configuration in [configs](https://github.com/hankyul2/DomainAdaptation/tree/main/configs)

   ```bash
   python3 main.py fit --config=configs/cdan_e.yaml -d 'amazon_webcam' -g '0,'
   ```
   
   

### Closed world Benchmark Result

*Office-31 Dataset Benchmark reported in original papers*

|                   | A>D      | A>W      | D>A      | D>W      | W>A      | W>D       | Avg      |
| ----------------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| source only       | 80.8     | 76.9     | 60.3     | 95.3     | 63.6     | 98.7      | 79.3     |
| DANN (2015)       | 79.7     | 82.0     | 68.2     | 96.9     | 67.4     | 99.1      | 82.2     |
| CDAN (2017)       | 89.8     | 93.1     | 70.1     | 98.2     | 68.0     | 100.0     | 86.6     |
| CDAN+E (2017)     | 92.9     | 94.1     | 71.0     | 98.6     | 69.3     | 100.0     | 87.7     |
| MSTN (2018)       | 90.4     | 91.3     | 72.7     | 98.9     | 65.6     | 100.0     | 86.5     |
| BSP+DANN (2019)   | 90.0     | 93.0     | 71.9     | 98.0     | 73.0     | 100.0     | 87.7     |
| BSP+CDAN+E (2019) | 93.0     | 93.3     | 73.6     | 98.2     | 72.6     | 100.0     | 88.5     |
| DSBN+MSTN (2019)  | 90.8     | 93.3     | 72.7     | 99.1     | 73.9     | 100.0     | 88.3     |
| RSDA+MSTN (2020)  | 95.8     | 96.1     | 77.4     | 99.3     | 78.9     | 100.0     | 91.1     |
| SHOT (2020)       | 94.0     | 90.1     | 74.7     | 98.4     | 74.3     | 99.9      | 88.6     |
| TransDA (2021)    | **97.2** | 95.0     | 73.7     | 99.3     | 79.3     | 99.6      | 90.7     |
| FixBi (2021)      | 95.0     | **96.1** | **78.7** | **99.3** | **79.4** | **100.0** | **91.4** |

*In this work*

|                                                              | A>D<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | A>W<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | D>A<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | D>W<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | W>A<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | W>D<br />[[tf.dev](https://tensorboard.dev/experiment/TAqpaeiAQVKoP85cQ9ogSw/#scalars)] | Avg      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| source only<br />[[code](src/system/source_only.py), [config](configs/source_only.yaml)] | 82.3                                                         | 77.9                                                         | 63.0                                                         | 94.5                                                         | 64.7                                                         | 98.3                                                         | 80.1     |
| source only ViT<br />[[code](src/system/source_only.py), [config](configs/source_only.yaml)] | 88.0                                                         | 87.9                                                         | 76.7                                                         | 97.7                                                         | 77.1                                                         | 99.7                                                         | 87.8     |
| DANN (2015)<br />[[code](src/system/dann.py), [config](configs/dann.yaml)] | 87.2                                                         | 90.4                                                         | 70.6                                                         | 97.8                                                         | 73.7                                                         | 99.7                                                         | 86.6     |
| CDAN (2017)<br />[[code](src/system/cdan.py), [config](configs/dan.yaml)] | 92.4                                                         | 95.1                                                         | 75.8                                                         | 98.6                                                         | 74.4                                                         | 99.9                                                         | 89.4     |
| CDAN+E (2017)<br />[[code](src/system/cdan.py), [config](configs/cdan_e.yaml)] | 93.2                                                         | 95.6                                                         | 75.1                                                         | 98.7                                                         | 75.0                                                         | **100.0**                                                    | 89.6     |
| MSTN (2018)<br />[[code](src/system/mstn.py), [config](configs/mstn.yaml)] | 89.0                                                         | 92.7                                                         | 71.4                                                         | 97.9                                                         | 74.1                                                         | 99.9                                                         | 87.5     |
| BSP+DANN (2019)<br />[[code](src/system/bsp.py), [config](configs/bsp_dann.yaml)] | 86.3                                                         | 89.1                                                         | 71.4                                                         | 97.7                                                         | 73.4                                                         | **100.0**                                                    | 86.3     |
| BSP+CDAN+E (2019)<br />[[code](src/system/bsp.py), [config](configs/bsp_cdan_e.yaml)] | 92.6                                                         | 94.7                                                         | 73.8                                                         | 98.7                                                         | 74.7                                                         | **100.0**                                                    | 89.1     |
| DSBN+MSTN (2019)<br />[[code](src/system/dsbn.py), [config](configs/dsbn_mstn_stage1.yaml)] | 87.3                                                         | 91.9                                                         | 71.0                                                         | 97.8                                                         | 73.4                                                         | 100.0                                                        | 86.9     |
| RSDA+MSTN (2020)<br />[Not Implemented]                      | -                                                            | -                                                            | -                                                            | -                                                            | -                                                            | -                                                            | -        |
| SHOT (2020)<br />[[code](src/system/shot.py), [config](configs/shot.yaml)] | 93.2                                                         | 92.5                                                         | 74.3                                                         | 98.2                                                         | 75.9                                                         | **100.0**                                                    | 89.0     |
| SHOT (CDAN+E) (2020)<br />[[code](src/system/exp.py), [config](configs/exp/shot_cdan.yaml)] | 93.2                                                         | 95.7                                                         | 77.7                                                         | **98.9**                                                     | 76.0                                                         | **100.0**                                                    | 90.2     |
| MIXUP (CDAN+E) (2021)<br />[[code](src/system/exp.py), [config](configs/exp/pseudo_mixup_ratio_cdan.yaml)] | 92.9                                                         | **96.1**                                                     | 76.2                                                         | 98.9                                                         | 77.7                                                         | **100.0**                                                    | 90.3     |
| TransDA (2021)<br />[[code](src/system/trans_da.py), [config](configs/transDA.yaml)] | 94.4                                                         | 95.8                                                         | **82.3**                                                     | 99.2                                                         | **82.0**                                                     | 99.8                                                         | **92.3** |
| FixBi (2021)<br />[[code](src/system/fixbi.py), [config](configs/fixbi.yaml)] | 90.8                                                         | 95.7                                                         | 72.6                                                         | 98.7                                                         | 74.8                                                         | **100.0**                                                    | 88.8     |

*Note*

1. Reported scores are from SHOT, FixBi paper
2. Evaluation datasets are:  `valid` = `test` = `target`. For me, this looks weird, but there are no other way to reproduce results in paper. But, source only model's evaluation is a bit different: `valid=source`, `test=target`
3. In this works, scores are 3 times averaged scores.
4. Optimizer and learning rate scheduler are same to all model(SGD) except `mstn`, `dsbn+mstn` (Adam)
5. `SHOT` can results lower accuracy than reported scores. To reproduce reported score, I recommend you to use provided source only model weights. (download [here]()) I don't know why...
6. `BSP`, `DSBN+MSTN`, `FixBi`: Fails to reproduce scores reported in paper
7. `SHOT (CDAN+E)`, `MIXUP (CDAN+E)` is just experimental method, so no tf.dev is provided.



### Future Updates

- [ ] Add weight parameter
- [ ] Add ViT results
- [ ] Add office-home dataset results
- [ ] Add digits results

