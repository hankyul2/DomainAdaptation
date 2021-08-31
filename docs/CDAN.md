### CDAN





#### Training Result

amazon to dslr (87.6)

| no   | method | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | amazon | dslr | 2021-08-29/17-42-38 | 86.87501 | 18    | 50     | 0.0003 | 32         |
| 2    | CDAN   | amazon | dslr | 2021-08-29/18-10-51 | 89.58334 | 48    | 50     | 0.0003 | 32         |
| 3    | CDAN   | amazon | dslr | 2021-08-29/18-41-51 | 86.25001 | 36    | 50     | 0.0003 | 32         |



amazon to webcam (92.0)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | amazon | webcam | 2021-08-29/17-48-30 | 92.31771 | 47    | 50     | 0.0003 | 32         |
| 2    | CDAN   | amazon | webcam | 2021-08-29/18-18-27 | 91.66667 | 46    | 50     | 0.0003 | 32         |
| 3    | CDAN   | amazon | webcam | 2021-08-29/18-49-49 | 92.0573  | 49    | 50     | 0.0003 | 32         |



dslr to amazon (68.8)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | dslr | amazon | 2021-08-29/17-39-52 | 68.71449 | 50    | 50     | 0.0003 | 32         |
| 2    | CDAN   | dslr | amazon | 2021-08-29/18-04-59 | 68.32387 | 15    | 50     | 0.0003 | 32         |
| 3    | CDAN   | dslr | amazon | 2021-08-29/18-32-26 | 69.46023 | 23    | 50     | 0.0003 | 32         |



dslr to webcam (98.3)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | dslr | webcam | 2021-08-29/16-58-03 | 98.17709 | 11    | 50     | 0.0003 | 32         |
| 2    | CDAN   | dslr | webcam | 2021-08-29/17-08-54 | 98.4375  | 15    | 50     | 0.0003 | 32         |
| 3    | CDAN   | dslr | webcam | 2021-08-29/17-20-14 | 98.3073  | 34    | 50     | 0.0003 | 32         |



webcam to amazon (69.3)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | webcam | amazon | 2021-08-29/17-57-52 | 67.96875 | 24    | 50     | 0.0003 | 32         |
| 2    | CDAN   | webcam | amazon | 2021-08-29/18-28-57 | 70.02841 | 37    | 50     | 0.0003 | 32         |
| 3    | CDAN   | webcam | amazon | 2021-08-29/18-59-26 | 69.85085 | 30    | 50     | 0.0003 | 32         |



webcam to dslr (100.0)

| no   | method | src    | tgt  | start_time          | acc  | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | ---- | ----- | ------ | ------ | ---------- |
| 1    | CDAN   | webcam | dslr | 2021-08-29/17-03-22 | 100  | 27    | 50     | 0.0003 | 32         |
| 2    | CDAN   | webcam | dslr | 2021-08-29/17-15-54 | 100  | 11    | 50     | 0.0003 | 32         |
| 3    | CDAN   | webcam | dslr | 2021-08-29/17-29-08 | 100  | 36    | 50     | 0.0003 | 32         |



#### References

- [thuml/CDAN](https://github.com/thuml/CDAN)
- [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)