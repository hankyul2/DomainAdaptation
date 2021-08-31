### CDAN 



#### Training Result

amazon to dslr (87.3)

| no   | method | src    | tgt  | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | amazon | dslr | 2021-08-29/19-13-09 | 88.12501 | 38    | 50     | 0.0003 | 32         |
| 2    | CDAN_E | amazon | dslr | 2021-08-29/19-44-07 | 86.45834 | 21    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | amazon | dslr | 2021-08-29/20-11-11 | 87.29167 | 16    | 50     | 0.0003 | 32         |



amazon to webcam (90.5)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | amazon | webcam | 2021-08-29/19-22-09 | 91.14584 | 23    | 50     | 0.0003 | 32         |
| 2    | CDAN_E | amazon | webcam | 2021-08-29/19-54-11 | 89.0625  | 19    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | amazon | webcam | 2021-08-29/20-24-44 | 91.27605 | 40    | 50     | 0.0003 | 32         |



dslr to amazon (72.4)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | dslr | amazon | 2021-08-29/19-01-40 | 72.65625 | 42    | 50     | 0.0003 | 32         |
| 2    | CDAN_E | dslr | amazon | 2021-08-29/19-31-41 | 73.01137 | 41    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | dslr | amazon | 2021-08-29/20-02-01 | 71.62642 | 15    | 50     | 0.0003 | 32         |



dslr to webcam (98.6)

| no   | method | src  | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ---- | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | dslr | webcam | 2021-08-29/17-31-44 | 98.69792 | 7     | 50     | 0.0003 | 32         |
| 2    | CDAN_E | dslr | webcam | 2021-08-29/17-43-25 | 98.04688 | 28    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | dslr | webcam | 2021-08-29/17-55-19 | 98.95834 | 5     | 50     | 0.0003 | 32         |



webcam to amazon (71.0)

| no   | method | src    | tgt    | start_time          | acc      | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ------ | ------------------- | -------- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | webcam | amazon | 2021-08-29/19-31-10 | 69.88637 | 21    | 50     | 0.0003 | 32         |
| 2    | CDAN_E | webcam | amazon | 2021-08-29/20-00-08 | 72.54972 | 39    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | webcam | amazon | 2021-08-29/20-29-47 | 70.66762 | 30    | 50     | 0.0003 | 32         |



webcam to dslr (100.0)

| no   | method | src    | tgt  | start_time          | acc  | epoch | nepoch | lr     | batch_size |
| ---- | ------ | ------ | ---- | ------------------- | ---- | ----- | ------ | ------ | ---------- |
| 1    | CDAN_E | webcam | dslr | 2021-08-29/17-42-45 | 100  | 7     | 50     | 0.0003 | 32         |
| 2    | CDAN_E | webcam | dslr | 2021-08-29/17-56-13 | 100  | 13    | 50     | 0.0003 | 32         |
| 3    | CDAN_E | webcam | dslr | 2021-08-29/18-10-44 | 100  | 7     | 50     | 0.0003 | 32         |



#### References

- [thuml/CDAN](https://github.com/thuml/CDAN)
- [Conditional Adversarial Domain Adaptation](https://arxiv.org/abs/1705.10667)

