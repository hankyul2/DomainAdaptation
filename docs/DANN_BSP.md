### DANN_BSP





#### Training Result

amazon to dslr (82.8)

| no   | method   | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | amazon | dslr | 2021-08-29/20-35-29 | 84.16667 | 9     | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | amazon | dslr | 2021-08-29/20-55-57 | 81.87501 | 8     | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | amazon | dslr | 2021-08-29/21-15-07 | 82.50001 | 11    | 50     | 0.0003 | 32         |



amazon to webcam (82.8)

| no   | method   | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | amazon | webcam | 2021-08-29/20-53-15 | 82.8125  | 10    | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | amazon | webcam | 2021-08-29/21-20-03 | 83.20313 | 12    | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | amazon | webcam | 2021-08-29/21-47-00 | 82.42188 | 11    | 50     | 0.0003 | 32         |



dslr to amazon (64.6)

| no   | method   | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | dslr | amazon | 2021-08-29/20-30-01 | 65.55398 | 4     | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | dslr | amazon | 2021-08-29/20-50-58 | 64.3821  | 7     | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | dslr | amazon | 2021-08-29/21-10-06 | 63.7429  | 2     | 50     | 0.0003 | 32         |



dslr to webcam (97.3)

| no   | method   | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | dslr | webcam | 2021-08-29/18-08-48 | 97.00521 | 18    | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | dslr | webcam | 2021-08-29/18-19-28 | 96.875   | 6     | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | dslr | webcam | 2021-08-29/18-30-28 | 97.91667 | 18    | 50     | 0.0003 | 32         |



webcam to amazon (67.4)

| no   | method   | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | webcam | amazon | 2021-08-29/20-59-27 | 66.33523 | 47    | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | webcam | amazon | 2021-08-29/21-29-17 | 67.72017 | 48    | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | webcam | amazon | 2021-08-29/21-59-11 | 68.14631 | 41    | 50     | 0.0003 | 32         |



webcam to dslr (99.7)

| no   | method   | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | DANN_BSP | webcam | dslr | 2021-08-29/18-25-43 | 99.79167 | 25    | 50     | 0.0003 | 32         |
| 2    | DANN_BSP | webcam | dslr | 2021-08-29/18-39-04 | 99.37501 | 7     | 50     | 0.0003 | 32         |
| 3    | DANN_BSP | webcam | dslr | 2021-08-29/18-53-06 | 100      | 7     | 50     | 0.0003 | 32         |



#### References

- [thuml/Batch-Spectral-Penalization](https://github.com/thuml/Batch-Spectral-Penalization)
- [Transferability vs Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)

