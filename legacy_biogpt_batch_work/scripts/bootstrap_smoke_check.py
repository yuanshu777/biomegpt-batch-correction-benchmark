from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
data_dir = root / 'dataset_v3'
required = [
    data_dir / 'abund_pretraining_phase1_gut_and_nongut.csv.zip',
    data_dir / 'abund_pretraining_phase2_gut.csv.zip',
    data_dir / 'abund_finetuning_gut_prev3.csv.zip',
    data_dir / 'meta_pretraining_phase1_gut_and_nongut.csv',
    data_dir / 'meta_pretraining_phase2_gut.csv',
    data_dir / 'meta_finetuning_gut_prev3.csv',
    data_dir / 'species_taxonomy_filled_validated_Serena.xlsx',
    data_dir / 'ExVal' / 'df_validation_data.csv',
    data_dir / 'ExVal' / 'df_validation_data_metadata.csv',
    data_dir / 'meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))

meta = pd.read_csv(data_dir / 'meta_pretraining_phase2_gut.csv')
batch = pd.read_csv(data_dir / 'meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv', low_memory=False)
summary = {
    'data_dir': str(data_dir),
    'phase2_meta_shape': list(meta.shape),
    'phase2_unique_samples': int(meta.iloc[:, 0].nunique()),
    'batch_annotation_shape': list(batch.shape),
    'batch_confidence_counts': batch['external_confidence'].value_counts(dropna=False).to_dict(),
    'safe_for_final_batch_correction_conservative_counts': batch['safe_for_final_batch_correction_conservative'].value_counts(dropna=False).astype(int).to_dict(),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
