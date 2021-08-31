### CDAN_BSP





#### Training Result

amazon to dslr (89.3)

| no   | method   | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | amazon | dslr | 2021-08-29/21-35-35 | 89.37501 | 30    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | amazon | dslr | 2021-08-29/21-54-17 | 89.37501 | 18    | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | amazon | dslr | 2021-08-29/22-13-18 | 89.16667 | 35    | 50     | 0.0003 | 32         |



amazon to webcam (91.6)

| no   | method   | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | amazon | webcam | 2021-08-29/22-13-57 | 91.79688 | 48    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | amazon | webcam | 2021-08-29/22-40-23 | 92.44792 | 41    | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | amazon | webcam | 2021-08-29/23-07-09 | 90.4948  | 36    | 50     | 0.0003 | 32         |



dslr to amazon (74.1)

| no   | method   | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | dslr | amazon | 2021-08-29/21-30-09 | 73.47301 | 16    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | dslr | amazon | 2021-08-29/21-48-59 | 74.18324 | 14    | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | dslr | amazon | 2021-08-29/22-07-47 | 74.71591 | 17    | 50     | 0.0003 | 32         |



dslr to webcam (97.9)

| no   | method   | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | dslr | webcam | 2021-08-29/18-42-01 | 98.17709 | 14    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | dslr | webcam | 2021-08-29/18-53-03 | 97.91667 | 16    | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | dslr | webcam | 2021-08-29/19-03-55 | 97.65625 | 22    | 50     | 0.0003 | 32         |



webcam to amazon (74.0)

| no   | method   | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | webcam | amazon | 2021-08-29/22-29-04 | 73.65057 | 13    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | webcam | amazon | 2021-08-29/22-58-47 | 73.89915 | 39    | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | webcam | amazon | 2021-08-29/23-28-32 | 74.39631 | 18    | 50     | 0.0003 | 32         |



webcam to dslr (99.8)

| no   | method   | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | -------- | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_BSP | webcam | dslr | 2021-08-29/19-07-33 | 100      | 15    | 50     | 0.0003 | 32         |
| 2    | CDAN_BSP | webcam | dslr | 2021-08-29/19-22-47 | 100      | 8     | 50     | 0.0003 | 32         |
| 3    | CDAN_BSP | webcam | dslr | 2021-08-29/19-38-08 | 99.37501 | 11    | 50     | 0.0003 | 32         |



#### References

- [thuml/Batch-Spectral-Penalization](https://github.com/thuml/Batch-Spectral-Penalization)
- [Transferability vs Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i.html)