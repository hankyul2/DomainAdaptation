### CDAN_E_BSP



#### Training Result

amazon to dslr (89.0)

| no   | method     | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | amazon | dslr | 2021-08-29/22-31-43 | 90.62501 | 11    | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | amazon | dslr | 2021-08-29/22-49-18 | 87.70834 | 9     | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | amazon | dslr | 2021-08-29/23-08-25 | 88.54167 | 12    | 50     | 0.0003 | 32         |



amazon to webcam (91.3)

| no   | method     | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | amazon | webcam | 2021-08-29/23-34-00 | 91.79688 | 9     | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | amazon | webcam | 2021-08-30/00-00-20 | 91.14584 | 20    | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | amazon | webcam | 2021-08-30/00-26-34 | 90.88542 | 8     | 50     | 0.0003 | 32         |



dslr to amazon (74.1)

| no   | method     | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | dslr | amazon | 2021-08-29/22-26-35 | 75.53267 | 31    | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | dslr | amazon | 2021-08-29/22-44-20 | 74.0412  | 30    | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | dslr | amazon | 2021-08-29/23-02-50 | 72.79829 | 19    | 50     | 0.0003 | 32         |



dslr to webcam (98.5)

| no   | method     | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | dslr | webcam | 2021-08-29/19-17-35 | 98.56771 | 8     | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | dslr | webcam | 2021-08-29/19-31-26 | 98.56771 | 20    | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | dslr | webcam | 2021-08-29/19-45-12 | 98.4375  | 16    | 50     | 0.0003 | 32         |



webcam to amazon (73.9)

| no   | method     | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | webcam | amazon | 2021-08-29/23-57-31 | 73.79262 | 18    | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | webcam | amazon | 2021-08-30/00-26-56 | 74.57387 | 40    | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | webcam | amazon | 2021-08-30/00-54-48 | 73.40199 | 38    | 50     | 0.0003 | 32         |



webcam to dslr (100.0)

| no   | method     | src    | tgt  | start_time          | acc  | epoch | nepoch | lr     | batch_size |
| ---- | ---------- | ------ | ---- | ------------------- | ---- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E_BSP | webcam | dslr | 2021-08-29/19-52-47 | 100  | 10    | 50     | 0.0003 | 32         |
| 2    | CDAN_E_BSP | webcam | dslr | 2021-08-29/20-05-34 | 100  | 12    | 50     | 0.0003 | 32         |
| 3    | CDAN_E_BSP | webcam | dslr | 2021-08-29/20-18-08 | 100  | 12    | 50     | 0.0003 | 32         |



#### References

- [fungtion/DANN](https://github.com/fungtion/DANN)
- [Domain Adversarial Training of Neural Network](https://arxiv.org/abs/1505.07818)