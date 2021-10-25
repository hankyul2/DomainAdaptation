# Domain Adaptation

This repository contains pytorch version source code introduced by domain adaptation paper:

1. DANN (2015)
2. CDAN (2017)
3. MSTN (2018)
4. BSP (2019)
5. DSBN (2019)
6. RSDA-MSTN (2020)
7. SHOT (2020)
8. TransDA (2021)
9. FixBi (2021)



### Tutorial (use standard dataset: Office-31, Office-home, Digits)

1. Clone repository and install dependency

   ```bash
   git clone https://github.com/hankyul2/DomainAdaptation.git
   pip3 install requirements.txt
   ```

2. Train Model (current version is too difficult to use, wait a moment)

   *pass*



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

|                                         | A>D      | A>W      | D>A      | D>W      | W>A      | W>D       | Avg      |
| --------------------------------------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| source only<br />[tf.dev, summary]      | 82.3     | 77.9     | 63.0     | 94.5     | 64.7     | 98.3      | 80.1     |
| DANN (2015)<br />[tf.dev, summary]      | 87.2     | 90.4     | 70.6     | 97.8     | 73.7     | 99.7      | 86.6     |
| CDAN (2017)<br />[tf.dev, summary]      | 92.4     | 95.1     | **75.8** | 98.6     | 74.4     | 99.9      | 89.4     |
| CDAN+E (2017)<br />[tf.dev, summary]    | 93.2     | 95.6     | 75.1     | **98.7** | 75.0     | **100.0** | **89.6** |
| MSTN (2018)<br />[tf.dev, summary]      | 89.0     | 92.7     | 71.4     | 97.9     | 74.1     | 99.9      | 87.5     |
| BSP+DANN(2019)<br />[tf.dev, summary]   | 86.3     | 89.1     | 71.4     | 97.7     | 73.4     | **100.0** | 86.3     |
| BSP+CDAN+E(2019)<br />[tf.dev, summary] | 92.6     | 94.7     | 73.8     | **98.7** | 74.7     | **100.0** | 89.1     |
| DSBN+MSTN (2019)<br />[tf.dev, summary] | 87.3     | 91.9     | 71.0     | 97.8     | 73.4     | 100.0     | 86.9     |
| RSDA+MSTN (2020)<br />[tf.dev, summary] | -        | -        | -        | -        | -        | -         | -        |
| SHOT (2020)<br />[tf.dev, summary]      | 93.2     | 92.5     | 74.3     | 98.2     | **75.9** | **100.0** | 89.0     |
| TransDA (2021)<br />[tf.dev, summary]   | **97.5** | **96.4** | 71.2     | 97.7     | 67.7     | 99.3      | 88.3     |
| FixBi (2021)<br />[tf.dev, summary]     | 90.8     | 95.7     | 72.6     | **98.7** | 74.8     | **100.0** | 88.8     |

*Note*

1. Reported scores are from SHOT, FixBi paper
2. Backbone models are two types:
   1. *resnet50* pretrained on *ILSVRC2012*
   2. resnet50 + vit_base_patch16_224 (for transDA)
3. Evaluation datasets are:  `valid` = `test` = `target`. For me, this looks weird, but there are no other way to reproduce results in paper. But, source only model's evaluation is a bit different: `valid=source`, `test=target`
4. In this works, scores are 3 times averaged scores.
6. Optimizer and learning rate scheduler are same to all model(SGD) except `mstn`, `dsbn+mstn` (Adam)
7. Fails to reproduce results: `BSP`, `DSBN+MSTN`, `FixBi`, `TransDA`



### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)
- [thuml/CDAN](https://github.com/thuml/CDAN)
- [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

- [thuml/Batch-Spectral-Penalization](https://github.com/thuml/Batch-Spectral-Penalization)
- [Transferability vs Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)