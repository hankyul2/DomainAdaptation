### Base (source only)

This is baseline model, which have pretrained resnet50 with bottleneck layer and fc layer



#### Training Result

amazon to dslr (82.5)

| no   | method | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | amazon | dslr | 2021-08-29/16-26-44 | 82.50001 | 32    | 50     | 0.0003 | 32         |
| 2    | DANN   | amazon | dslr | 2021-08-29/16-51-42 | 82.70834 | 34    | 50     | 0.0003 | 32         |
| 3    | DANN   | amazon | dslr | 2021-08-29/17-17-04 | 82.29167 | 44    | 50     | 0.0003 | 32         |



amazon to webcam (83.4)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | amazon | webcam | 2021-08-29/16-26-45 | 83.46355 | 38    | 50     | 0.0003 | 32         |
| 2    | DANN   | amazon | webcam | 2021-08-29/16-53-26 | 83.59375 | 47    | 50     | 0.0003 | 32         |
| 3    | DANN   | amazon | webcam | 2021-08-29/17-20-45 | 83.20313 | 46    | 50     | 0.0003 | 32         |



dslr to amazon (65.5)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | dslr | amazon | 2021-08-29/16-26-45 | 65.44745 | 21    | 50     | 0.0003 | 32         |
| 2    | DANN   | dslr | amazon | 2021-08-29/16-50-51 | 65.98012 | 23    | 50     | 0.0003 | 32         |
| 3    | DANN   | dslr | amazon | 2021-08-29/17-15-20 | 65.09233 | 15    | 50     | 0.0003 | 32         |



dslr to webcam (98.3)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | dslr | webcam | 2021-08-29/16-26-45 | 98.69792 | 37    | 50     | 0.0003 | 32         |
| 2    | DANN   | dslr | webcam | 2021-08-29/16-37-05 | 97.91667 | 50    | 50     | 0.0003 | 32         |
| 3    | DANN   | dslr | webcam | 2021-08-29/16-47-29 | 98.17709 | 34    | 50     | 0.0003 | 32         |



webcam to amazon (65.5)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | webcam | amazon | 2021-08-29/16-26-46 | 65.05682 | 40    | 50     | 0.0003 | 32         |
| 2    | DANN   | webcam | amazon | 2021-08-29/16-56-23 | 65.87358 | 15    | 50     | 0.0003 | 32         |
| 3    | DANN   | webcam | amazon | 2021-08-29/17-26-25 | 65.58949 | 18    | 50     | 0.0003 | 32         |



webcam to dslr (100.0)

| no   | method | src    | tgt  | start_time          | acc  | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | ---- | ----- | ------ | ------ | ---------- |
| 1    | DANN   | webcam | dslr | 2021-08-29/16-26-46 | 100  | 21    | 50     | 0.0003 | 32         |
| 2    | DANN   | webcam | dslr | 2021-08-29/16-38-49 | 100  | 20    | 50     | 0.0003 | 32         |
| 3    | DANN   | webcam | dslr | 2021-08-29/16-50-57 | 100  | 13    | 50     | 0.0003 | 32         |



#### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)