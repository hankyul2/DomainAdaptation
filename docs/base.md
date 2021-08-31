### Base (source only)

This is baseline model, which have pretrained resnet50 with bottleneck layer and fc layer



#### Training Result

amazon to dslr (79.2)

| no   | method | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | amazon | dslr | 2021-08-29/15-11-08 | 80.62501 | 40    | 50     | 0.0003 | 32         |
| 2    | BASE   | amazon | dslr | 2021-08-29/15-26-56 | 78.33334 | 50    | 50     | 0.0003 | 32         |
| 3    | BASE   | amazon | dslr | 2021-08-29/15-41-54 | 78.54167 | 41    | 50     | 0.0003 | 32         |



amazon to webcam (76.4)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | amazon | webcam | 2021-08-29/15-11-08 | 76.69271 | 34    | 50     | 0.0003 | 32         |
| 2    | BASE   | amazon | webcam | 2021-08-29/15-24-43 | 76.30209 | 30    | 50     | 0.0003 | 32         |
| 3    | BASE   | amazon | webcam | 2021-08-29/15-38-05 | 76.30209 | 32    | 50     | 0.0003 | 32         |



dslr to amazon (64.8)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | dslr | amazon | 2021-08-29/15-11-08 | 64.20454 | 9     | 50     | 0.0003 | 32         |
| 2    | BASE   | dslr | amazon | 2021-08-29/15-31-35 | 65.76704 | 16    | 50     | 0.0003 | 32         |
| 3    | BASE   | dslr | amazon | 2021-08-29/15-48-08 | 64.41762 | 37    | 50     | 0.0003 | 32         |



dslr to webcam (96.7)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | dslr | webcam | 2021-08-29/15-11-09 | 96.875   | 40    | 50     | 0.0003 | 32         |
| 2    | BASE   | dslr | webcam | 2021-08-29/15-18-21 | 97.52605 | 17    | 50     | 0.0003 | 32         |
| 3    | BASE   | dslr | webcam | 2021-08-29/15-26-02 | 95.83334 | 13    | 50     | 0.0003 | 32         |



webcam to amazon (64.8)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | webcam | amazon | 2021-08-29/15-11-09 | 65.83807 | 35    | 50     | 0.0003 | 32         |
| 2    | BASE   | webcam | amazon | 2021-08-29/15-24-42 | 63.95597 | 31    | 50     | 0.0003 | 32         |
| 3    | BASE   | webcam | amazon | 2021-08-29/15-36-58 | 64.55966 | 32    | 50     | 0.0003 | 32         |



webcam to dslr (99.0)

| no   | method | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | BASE   | webcam | dslr | 2021-08-29/15-11-09 | 98.54167 | 19    | 50     | 0.0003 | 32         |
| 2    | BASE   | webcam | dslr | 2021-08-29/15-18-29 | 98.95834 | 21    | 50     | 0.0003 | 32         |
| 3    | BASE   | webcam | dslr | 2021-08-29/15-26-18 | 99.37501 | 39    | 50     | 0.0003 | 32         |



#### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)