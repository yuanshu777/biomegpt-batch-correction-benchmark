# MMUPHin Original CRC Reproduction Check

## What Was Re-run

Re-ran the original local MMUPHin CRC scripts from:

- `C:\Users\Yuanshu\Documents\new_attemp_batch\mmuphin_crc_scouting.R`
- `C:\Users\Yuanshu\Documents\new_attemp_batch\prepare_crc_controlled_benchmark.R`

using:

- `C:\Program Files\R\R-4.5.1\bin\Rscript.exe`
- MMUPHin package dataset: `CRC_abd`, `CRC_meta`

## Scouting Output

Source:

- `C:\Users\Yuanshu\Documents\new_attemp_batch\outputs_mmuphin_dataset_scouting\mmuphin_crc_raw_vs_adjusted_metrics.csv`

This reproduces the professor-facing screenshot values:

| Metric | Raw | Adjusted |
|---|---:|---:|
| Study R2, controlling for condition | 0.078563 | 0.030046 |
| Study classifier balanced accuracy | 0.765422 | 0.647386 |
| Disease LOSO mean within-study AUC | 0.713335 | 0.705801 |
| Condition R2, controlling for study | 0.007875 | 0.008849 |

## Controlled Benchmark Output

Source:

- `C:\Users\Yuanshu\Documents\new_attemp_batch\crc_controlled_benchmark\reports\crc_raw_vs_mmuphin_metrics.csv`

The PERMANOVA R2 values match the scouting output exactly; classifier values differ slightly because this controlled benchmark uses its own frozen split/evaluator setup.

| Metric | Raw | MMUPHin |
|---|---:|---:|
| Study R2, controlling for condition | 0.078563 | 0.030046 |
| Study prediction balanced accuracy | 0.756370 | 0.673665 |
| Disease LOSO mean within-study AUC | 0.709835 | 0.687841 |
| Condition R2, controlling for study | 0.007875 | 0.008849 |

## Important Correction

The earlier CRC389 Python value around `0.027` is not directly comparable with the MMUPHin `7.86%` value. It came from a 389-sample overlap subset and a standardized Euclidean/linear partial-R2 approximation, while the MMUPHin result is a Bray-Curtis PERMANOVA R2 on the original full 551-sample CRC dataset.

Use the MMUPHin full-data R outputs above as the reference when comparing to the professor-facing screenshot.
