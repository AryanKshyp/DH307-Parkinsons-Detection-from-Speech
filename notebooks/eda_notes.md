# Stage 2 - exploratory analysis

Generated from `data\manifests\manifest_processed.csv`, 3871 files / 216 speakers / 12.26 hours.

### Speakers per dataset x label

| dataset   |   HC |   PD |   total |
|:----------|-----:|-----:|--------:|
| NeuroVoz  |   58 |   56 |     114 |
| ItalianPD |   37 |   28 |      65 |
| MDVR-KCL  |   21 |   16 |      37 |

### Files, hours and median utterance length

| dataset   |   files |   hours |   median_s |
|:----------|--------:|--------:|-----------:|
| NeuroVoz  |    2968 |    3.86 |       3.47 |
| ItalianPD |     830 |    5.54 |      11.56 |
| MDVR-KCL  |      73 |    2.87 |     141.29 |

### Task availability (files per dataset)

| task         |   NeuroVoz |   ItalianPD |   MDVR-KCL |
|:-------------|-----------:|------------:|-----------:|
| ddk_pa       |          0 |          50 |          0 |
| ddk_pataka   |        101 |           0 |          0 |
| ddk_ta       |          0 |          50 |          0 |
| monologue    |         76 |           0 |          0 |
| read_phrases |          0 |          47 |          0 |
| read_text    |          0 |         123 |         37 |
| read_words   |          0 |          65 |          0 |
| sentence     |       1726 |           0 |          0 |
| spontaneous  |          0 |           0 |         36 |
| vowel_a      |        249 |          99 |          0 |
| vowel_e      |        204 |          99 |          0 |
| vowel_i      |        237 |          99 |          0 |
| vowel_o      |        240 |          99 |          0 |
| vowel_u      |        135 |          99 |          0 |

Raw task codes present in all three corpora: none

### Assigned task families (files per dataset)

| family          |   NeuroVoz |   ItalianPD |   MDVR-KCL |
|:----------------|-----------:|------------:|-----------:|
| phrases         |       1726 |          47 |          0 |
| reading passage |          0 |         123 |         37 |
| spontaneous     |         76 |           0 |         36 |
| words           |          0 |          65 |          0 |

2110 of 3871 files belong to an assigned task family; the remaining 1761 are vowels and DDK.

### Age metadata coverage (speakers)

| dataset   |   speakers |   with_age |
|:----------|-----------:|-----------:|
| NeuroVoz  |        114 |        106 |
| ItalianPD |         65 |         58 |
| MDVR-KCL  |         37 |          0 |

### Age by dataset x label

|                     |   n |   mean |   min |   max |
|:--------------------|----:|-------:|------:|------:|
| ('ItalianPD', 'HC') |  32 |   45.9 |    19 |    77 |
| ('ItalianPD', 'PD') |  26 |   67   |    40 |    80 |
| ('NeuroVoz', 'HC')  |  53 |   64   |    31 |    86 |
| ('NeuroVoz', 'PD')  |  53 |   71.1 |    41 |    88 |

### ItalianPD age by cohort

| cohort          |   n |   mean |   min |   max |
|:----------------|----:|-------:|------:|------:|
| elderly_control |  17 |   68   |    60 |    77 |
| patient         |  26 |   67   |    40 |    80 |
| young_control   |  15 |   20.8 |    19 |    29 |

### Controls younger than the youngest patient

| dataset   |   controls |   under_50 |   patient_min_age |
|:----------|-----------:|-----------:|------------------:|
| ItalianPD |         32 |         15 |                40 |
| NeuroVoz  |         53 |          2 |                41 |


## Corpus separability from mean MFCCs

Logistic regression on the 20 mean MFCCs, 5-fold speaker-disjoint:

- predicting corpus: 0.892 +/- 0.009 (majority class 0.767)
- predicting PD/HC: 0.683 +/- 0.035 (majority class 0.500)

Figures written to `notebooks\figures`.
