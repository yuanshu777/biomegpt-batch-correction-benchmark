# MMUPHin-Style CRC389 GRL Comparison

## Scope

This report re-evaluates Raw abundance, MMUPHin adjusted abundance, Full-data tuned GRL, and Cross-fitted tuned GRL on the same 389 overlap samples with MMUPHin-style metrics.

Important: this is not the original 551-sample MMUPHin controlled benchmark from the professor-facing screenshot. CRC389 excludes some samples/studies, and FengQ has only CRC samples in this overlap subset, so its held-out disease AUC is skipped.

## CRC389 Same-Evaluator Primary Table

| method                     |   condition_R2_study_controlled |   disease_LOSO_mean_within_study_AUC |   study_R2_condition_controlled |   study_prediction_balanced_accuracy |
|:---------------------------|--------------------------------:|-------------------------------------:|--------------------------------:|-------------------------------------:|
| Raw abundance              |                      0.00710953 |                             0.733621 |                       0.0272079 |                             0.795247 |
| MMUPHin adjusted abundance |                      0.00728463 |                             0.776142 |                       0.0160286 |                             0.569473 |
| Full-data tuned GRL        |                      0.0220151  |                             0.675858 |                       0.0831716 |                             0.52781  |
| Cross-fitted tuned GRL     |                      0.0143522  |                             0.609569 |                       0.0447281 |                             0.395419 |

## Interpretation

- Raw study BA is 0.795; MMUPHin is 0.569; cross-fitted GRL is 0.395.
- Raw disease LOSO AUC is 0.734; MMUPHin is 0.776; cross-fitted GRL is 0.610.
- Main reading: Cross-fitted GRL has the lowest study-classifier balanced accuracy on this CRC389 same-evaluator table, but its condition-controlled study R2 is still higher than MMUPHin and its disease LOSO AUC is lower than both raw and MMUPHin.
- Therefore your intuition is partly right: cross-fitted GRL is not bad on the study-classifier metric. The unresolved issues are disease-signal retention under LOSO and the higher study R2 compared with MMUPHin.

## Metric Caveats

- Original MMUPHin abundance PERMANOVA uses Bray-Curtis distance. GRL z has negative embedding dimensions, so this CRC389 same-evaluator table uses standardized Euclidean/linear partial R2 for all four methods.
- Classifier metrics are logistic probes with deterministic repeated folds, not the R/glmnet implementation from the original controlled benchmark.
- Use this as a fair CRC389 diagnostic, not as a replacement for the original 551-sample MMUPHin table.

## Official Full Benchmark Reference

These are the current local frozen raw/MMUPHin reference metrics from `crc_controlled_benchmark`; they are included to explain why the screenshot numbers are close but not identical to CRC389.

| method   | metric                             |   estimate | detail                                                                                                                                                                                                                              |
|:---------|:-----------------------------------|-----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| raw      | study_R2_condition_controlled      | 0.0785634  | Bray-Curtis; marginal term in studyID + study_condition                                                                                                                                                                             |
| raw      | condition_R2_study_controlled      | 0.00787524 | Bray-Curtis; marginal term in studyID + study_condition                                                                                                                                                                             |
| raw      | study_prediction_balanced_accuracy | 0.75637    | 3 fixed repeats; multinomial glmnet; chance 0.200                                                                                                                                                                                   |
| raw      | disease_LOSO_mean_within_study_AUC | 0.709835   | FengQ_2015.metaphlan_bugs_list.stool=0.749; HanniganGD_2017.metaphlan_bugs_list.stool=0.593; VogtmannE_2016.metaphlan_bugs_list.stool=0.669; YuJ_2015.metaphlan_bugs_list.stool=0.790; ZellerG_2014.metaphlan_bugs_list.stool=0.749 |
| mmuphin  | study_R2_condition_controlled      | 0.0300456  | Bray-Curtis; marginal term in studyID + study_condition                                                                                                                                                                             |
| mmuphin  | condition_R2_study_controlled      | 0.00884871 | Bray-Curtis; marginal term in studyID + study_condition                                                                                                                                                                             |
| mmuphin  | study_prediction_balanced_accuracy | 0.673665   | 3 fixed repeats; multinomial glmnet; chance 0.200                                                                                                                                                                                   |
| mmuphin  | disease_LOSO_mean_within_study_AUC | 0.687841   | FengQ_2015.metaphlan_bugs_list.stool=0.732; HanniganGD_2017.metaphlan_bugs_list.stool=0.601; VogtmannE_2016.metaphlan_bugs_list.stool=0.683; YuJ_2015.metaphlan_bugs_list.stool=0.752; ZellerG_2014.metaphlan_bugs_list.stool=0.672 |

## Output Files

- `metrics_long`: `outputs\metrics\mmuphin_style_crc389_metrics_long.csv`
- `primary_table`: `outputs\metrics\mmuphin_style_crc389_primary_table.csv`
- `official_full_reference`: `outputs\metrics\mmuphin_style_official_full_reference_raw_vs_mmuphin.csv`
- `figure_dir`: `outputs\figures\mmuphin_style_crc389`
