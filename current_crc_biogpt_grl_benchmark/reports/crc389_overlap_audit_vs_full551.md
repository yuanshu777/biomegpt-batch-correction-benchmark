# CRC389 Overlap Audit vs Full551

CRC389 is now treated as an exploratory BiomeGPT-overlap subset unless exact sample identity and study composition can be justified.

## Dataset Counts

| dataset   |   n_samples |   n_studies | studies                                                                                                                                                                                           |
|:----------|------------:|------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| full551   |         551 |           5 | FengQ_2015.metaphlan_bugs_list.stool;HanniganGD_2017.metaphlan_bugs_list.stool;VogtmannE_2016.metaphlan_bugs_list.stool;YuJ_2015.metaphlan_bugs_list.stool;ZellerG_2014.metaphlan_bugs_list.stool |
| crc389    |         389 |           4 | FengQ_2015.metaphlan_bugs_list.stool;VogtmannE_2016.metaphlan_bugs_list.stool;YuJ_2015.metaphlan_bugs_list.stool;ZellerG_2014.metaphlan_bugs_list.stool                                           |

## Interpretation

- Full551 remains the canonical MMUPHin CRC benchmark.
- CRC389 changes sample and study composition, including fewer studies than the original 551 benchmark.
- PCA and metrics from CRC389 should not be used as primary MMUPHin benchmark claims.
