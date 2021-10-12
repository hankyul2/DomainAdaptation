# Domain Adaptation

This repository contains pytorch version source code introduced by domain adaptation paper:

1. DANN (2016)
2. CDAN (2018)
3. BSP (2019)
4. SHOT (2020)
5. RSDA-MSTN (2020)
6. TransDA (2021)
7. FixBi (2021)



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
| DANN (2016)       | 79.7     | 82.0     | 68.2     | 96.9     | 67.4     | 99.1      | 82.2     |
| CDAN (2018)       | 89.8     | 93.1     | 70.1     | 98.2     | 68.0     | 100.0     | 86.6     |
| CDAN+E (2018)     | 92.9     | 94.1     | 71.0     | 98.6     | 69.3     | 100.0     | 87.7     |
| DANN+BSP (2019)   | 90.0     | 93.0     | 71.9     | 98.0     | 73.0     | 100.0     | 87.7     |
| CDAN+E+BSP (2019) | 93.0     | 93.3     | 73.6     | 98.2     | 72.6     | 100.0     | 88.5     |
| SHOT (2020)       | 94.0     | 90.1     | 74.7     | 98.4     | 74.3     | 99.9      | 88.6     |
| TransDA (2021)    | **97.2** | 95.0     | 73.7     | 99.3     | 79.3     | 99.6      | 90.7     |
| RSDA-MSTN (2020)  | 95.8     | 96.1     | 77.4     | 99.3     | 78.9     | 100.0     | 91.1     |
| FixBi (2021)      | 95.0     | **96.1** | **78.7** | **99.3** | **79.4** | **100.0** | **91.4** |

*In this work*

|                                          | A>D  | A>W  | D>A  | D>W  | W>A  | W>D  | Avg  |
| ---------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| source only                              |      |      |      |      |      |      |      |
| DANN (2016)<br />[tf.dev, summary]       |      |      |      |      |      |      |      |
| CDAN (2018)<br />[tf.dev, summary]       |      |      |      |      |      |      |      |
| CDAN+E (2018)<br />[tf.dev, summary]     |      |      |      |      |      |      |      |
| DANN+BSP (2019)<br />[tf.dev, summary]   |      |      |      |      |      |      |      |
| CDAN+E+BSP (2019)<br />[tf.dev, summary] |      |      |      |      |      |      |      |
| SHOT (2020)<br />[tf.dev, summary]       |      |      |      |      |      |      |      |
| TransDA (2021)<br />[tf.dev, summary]    |      |      |      |      |      |      |      |
| RSDA-MSTN (2020)<br />[tf.dev, summary]  |      |      |      |      |      |      |      |
| FixBi (2021)<br />[tf.dev, summary]      |      |      |      |      |      |      |      |

*Note*

1. Reported scores are from SHOT, FixBi paper
2. Backbone models are two types:
   1. *resnet50* pretrained on *ILSVRC2012*
   2. resnet50 + vit_base_patch16_224 (for transDA)
3. Evaluation datasets are:  `valid` = `test` = `target`. For me, this looks weird, but there are no other way to reproduce paper results.



### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)
- [thuml/CDAN](https://github.com/thuml/CDAN)
- [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

- [thuml/Batch-Spectral-Penalization](https://github.com/thuml/Batch-Spectral-Penalization)
- [Transferability vs Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)