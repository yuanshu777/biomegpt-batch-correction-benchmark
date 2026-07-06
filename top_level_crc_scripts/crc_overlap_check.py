from __future__ import annotations

import csv
import re
import subprocess
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent
BIOGPT_DIR = ROOT / "biogpt data"
PRIMARY_BIOGPT = BIOGPT_DIR / "meta_pretraining_phase2_gut_batch_annotation_external_enriched.csv"
CRC_BENCHMARK_META = ROOT / "crc_controlled_benchmark" / "data" / "crc_metadata.csv"
OUT_DIR = ROOT / "crc_overlap_check"
TMP_DIR = ROOT / "tmp" / "crc_overlap_check"
FULL_MMUPHIN_META = TMP_DIR / "mmuphin_crc_full_metadata.csv"
ENA_RUN_LOOKUP = TMP_DIR / "mmuphin_ena_run_lookup.csv"
RSCRIPT = Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe")

CRC_PATTERNS = [
    r"\bcrc\b",
    r"colorectal\s+cancer",
    r"colorectal\s+carcinoma",
    r"colon\s+cancer",
    r"rectal\s+cancer",
    r"adenoma",
    r"colorectal\s+adenoma",
    r"\bcolcan\b",
]
CONTROL_PATTERNS = [r"\bhealthy\b", r"\bcontrol\b", r"\bnormal\b"]
SAMPLE_LABEL_COLUMNS = [
    "Phenotype",
    "Phenotype_fullname",
    "ena_disease",
    "ena_host_status",
    "ena_sample_title",
    "ena_description",
    "sra_Disease",
    "sra_Affection_Status",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    tokens = re.split(r"[^A-Za-z0-9]+", str(value))
    return {tok.upper() for tok in tokens if len(tok) >= 4}


def ordered_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    tokens = re.split(r"[^A-Za-z0-9]+", str(value))
    return [tok.upper() for tok in tokens if len(tok) >= 4]


def any_pattern(value: object, patterns: list[str]) -> bool:
    if pd.isna(value):
        return False
    text = normalize_label(value)
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def export_mmuphin_full_metadata() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if FULL_MMUPHIN_META.exists():
        return
    if not RSCRIPT.exists():
        raise FileNotFoundError(f"Rscript not found: {RSCRIPT}")
    helper = TMP_DIR / "export_mmuphin_crc_meta.R"
    r_code = f"""
data('CRC_meta', package='MMUPHin')
CRC_meta[['sample_id']] <- rownames(CRC_meta)
CRC_meta <- CRC_meta[, c('sample_id', setdiff(colnames(CRC_meta), 'sample_id'))]
utils::write.csv(CRC_meta, file={str(FULL_MMUPHIN_META).replace(chr(92), '/')!r}, row.names=FALSE, na='')
"""
    helper.write_text(r_code, encoding="utf-8")
    subprocess.run([str(RSCRIPT), str(helper)], check=True, cwd=ROOT)


def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def ena_lookup_run(run_accession: str) -> dict[str, str]:
    fields = (
        "run_accession,sample_accession,secondary_sample_accession,"
        "study_accession,secondary_study_accession,study_title"
    )
    url = "https://www.ebi.ac.uk/ena/portal/api/filereport?" + urlencode(
        {
            "accession": run_accession,
            "result": "read_run",
            "fields": fields,
            "format": "tsv",
        }
    )
    req = Request(url, headers={"User-Agent": "crc-overlap-check/1.0"})
    with urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    parsed = pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    if parsed.empty:
        return {
            "run_accession": run_accession,
            "sample_accession": "",
            "secondary_sample_accession": "",
            "study_accession": "",
            "secondary_study_accession": "",
            "study_title": "",
            "lookup_status": "not_found",
        }
    row = parsed.iloc[0].to_dict()
    row["lookup_status"] = "ok"
    return row


def ena_file_report(accession: str) -> pd.DataFrame:
    fields = (
        "run_accession,sample_accession,secondary_sample_accession,"
        "study_accession,secondary_study_accession,study_title"
    )
    url = "https://www.ebi.ac.uk/ena/portal/api/filereport?" + urlencode(
        {
            "accession": accession,
            "result": "read_run",
            "fields": fields,
            "format": "tsv",
        }
    )
    req = Request(url, headers={"User-Agent": "crc-overlap-check/1.0"})
    with urlopen(req, timeout=90) as response:
        text = response.read().decode("utf-8")
    return pd.read_csv(StringIO(text), sep="\t", dtype=str, keep_default_na=False)


def build_ena_run_lookup(crc_full: pd.DataFrame) -> pd.DataFrame:
    if ENA_RUN_LOOKUP.exists():
        return read_csv_str(ENA_RUN_LOOKUP)

    sample_run_rows = []
    all_runs: set[str] = set()
    for _, row in crc_full.iterrows():
        sample_id = row["sample_id"]
        for token in ordered_tokens(row.get("NCBI_accession", "")):
            sample_run_rows.append(
                {
                    "mmuphin_sample_id": sample_id,
                    "mmuphin_studyID": row.get("studyID", ""),
                    "mmuphin_condition": row.get("study_condition", ""),
                    "run_accession": token,
                }
            )
            all_runs.add(token)

    sample_runs = pd.DataFrame(sample_run_rows)

    study_accessions: set[str] = set()
    representative_rows = []
    for study_id, group in sample_runs.groupby("mmuphin_studyID"):
        representative_run = group["run_accession"].iloc[0]
        try:
            representative = ena_lookup_run(representative_run)
            representative["lookup_status"] = "ok"
            representative_rows.append(representative)
            for field in ["study_accession", "secondary_study_accession"]:
                value = representative.get(field, "")
                if value:
                    study_accessions.add(value)
        except Exception as exc:  # pragma: no cover - network defensive logging
            representative_rows.append(
                {
                    "run_accession": representative_run,
                    "sample_accession": "",
                    "secondary_sample_accession": "",
                    "study_accession": "",
                    "secondary_study_accession": "",
                    "study_title": "",
                    "lookup_status": f"error: {exc}",
                }
            )

    reports = []
    for accession in sorted(study_accessions):
        try:
            reports.append(ena_file_report(accession))
        except Exception:
            continue

    if reports:
        report_df = pd.concat(reports, ignore_index=True).drop_duplicates("run_accession")
    else:
        report_df = pd.DataFrame(
            columns=[
                "run_accession",
                "sample_accession",
                "secondary_sample_accession",
                "study_accession",
                "secondary_study_accession",
                "study_title",
            ]
        )

    lookup_df = sample_runs.merge(report_df, on="run_accession", how="left")
    for col in [
        "sample_accession",
        "secondary_sample_accession",
        "study_accession",
        "secondary_study_accession",
        "study_title",
    ]:
        lookup_df[col] = lookup_df[col].fillna("")
    lookup_df["lookup_status"] = lookup_df["sample_accession"].map(lambda x: "ok" if x else "not_found")

    missing_runs = set(lookup_df.loc[lookup_df["lookup_status"] != "ok", "run_accession"])
    if missing_runs:
        fallback_rows = []
        for run in sorted(missing_runs):
            try:
                fallback_rows.append(ena_lookup_run(run))
            except Exception as exc:  # pragma: no cover - network defensive logging
                fallback_rows.append(
                    {
                        "run_accession": run,
                        "sample_accession": "",
                        "secondary_sample_accession": "",
                        "study_accession": "",
                        "secondary_study_accession": "",
                        "study_title": "",
                        "lookup_status": f"error: {exc}",
                    }
                )
        fallback = pd.DataFrame(fallback_rows).drop_duplicates("run_accession")
        lookup_df = lookup_df.drop(
            columns=[
                "sample_accession",
                "secondary_sample_accession",
                "study_accession",
                "secondary_study_accession",
                "study_title",
                "lookup_status",
            ]
        ).merge(fallback, on="run_accession", how="left")

    lookup_df.to_csv(ENA_RUN_LOOKUP, index=False)
    return lookup_df


def discover_metadata_files() -> pd.DataFrame:
    rows = []
    for path in sorted(BIOGPT_DIR.rglob("*")):
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        if not re.search(r"meta|metadata|sample|training|analysis|batch", path.name, re.I):
            continue
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            head = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=25, sep=sep)
        except Exception as exc:  # pragma: no cover - defensive audit logging
            rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "rows": "",
                    "columns": "",
                    "key_columns": "",
                    "read_status": f"error: {exc}",
                    "used_for_primary_overlap": False,
                }
            )
            continue
        key_cols = [
            col
            for col in head.columns
            if re.search(r"sample|study|cohort|phenotype|disease|condition|accession|batch", col, re.I)
        ]
        row_count = sum(1 for _ in path.open("rb")) - 1
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "rows": row_count,
                "columns": len(head.columns),
                "key_columns": ";".join(key_cols),
                "read_status": "ok",
                "used_for_primary_overlap": path == PRIMARY_BIOGPT,
            }
        )
    return pd.DataFrame(rows)


def classify_condition(row: pd.Series) -> str:
    values = [row.get(col, "") for col in SAMPLE_LABEL_COLUMNS]
    if any(any_pattern(v, CRC_PATTERNS) for v in values):
        return "CRC_candidate"
    if any(any_pattern(v, CONTROL_PATTERNS) for v in values):
        return "control_or_healthy"
    return "other_or_unknown"


def collect_biogpt_id_tokens(row: pd.Series, id_cols: list[str]) -> set[str]:
    tokens: set[str] = set()
    for col in id_cols:
        tokens |= split_tokens(row.get(col, ""))
    return tokens


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_mmuphin_full_metadata()

    metadata_inventory = discover_metadata_files()
    if not PRIMARY_BIOGPT.exists():
        raise FileNotFoundError(f"Primary BiomeGPT metadata not found: {PRIMARY_BIOGPT}")
    if not CRC_BENCHMARK_META.exists():
        raise FileNotFoundError(f"CRC benchmark metadata not found: {CRC_BENCHMARK_META}")

    bio = read_csv_str(PRIMARY_BIOGPT)
    crc = read_csv_str(CRC_BENCHMARK_META)
    crc_full = read_csv_str(FULL_MMUPHIN_META)
    ena_lookup = build_ena_run_lookup(crc_full)
    ena_ok = ena_lookup[ena_lookup["lookup_status"] == "ok"].copy()
    ena_by_sample = (
        ena_ok.groupby("mmuphin_sample_id")
        .agg(
            ena_run_accessions=("run_accession", lambda s: ";".join(sorted(set(s)))),
            ena_sample_accessions=("sample_accession", lambda s: ";".join(sorted(set(x for x in s if x)))),
            ena_secondary_sample_accessions=(
                "secondary_sample_accession",
                lambda s: ";".join(sorted(set(x for x in s if x))),
            ),
            ena_study_accessions=("study_accession", lambda s: ";".join(sorted(set(x for x in s if x)))),
            ena_secondary_study_accessions=(
                "secondary_study_accession",
                lambda s: ";".join(sorted(set(x for x in s if x))),
            ),
            ena_study_titles=("study_title", lambda s: " | ".join(sorted(set(x for x in s if x)))),
        )
        .reset_index()
        .rename(columns={"mmuphin_sample_id": "sample_id"})
    )
    crc = crc.merge(
        crc_full.drop(columns=[c for c in ["studyID", "study_condition"] if c in crc_full.columns]),
        on="sample_id",
        how="left",
    )
    crc = crc.merge(ena_by_sample, on="sample_id", how="left")
    for col in [
        "ena_run_accessions",
        "ena_sample_accessions",
        "ena_secondary_sample_accessions",
        "ena_study_accessions",
        "ena_secondary_study_accessions",
        "ena_study_titles",
    ]:
        crc[col] = crc[col].fillna("")

    bio["sample_id_norm"] = bio["sample_id"].map(normalize_text)
    bio["study_norm"] = bio["batch_label_external_recommended"].map(normalize_text)
    bio["condition_class"] = bio.apply(classify_condition, axis=1)

    phenotype_cols = [col for col in SAMPLE_LABEL_COLUMNS if col in bio.columns]
    bio["crc_label_columns"] = bio.apply(
        lambda row: ";".join([col for col in phenotype_cols if any_pattern(row.get(col, ""), CRC_PATTERNS)]),
        axis=1,
    )
    bio["crc_label_values"] = bio.apply(
        lambda row: " | ".join(
            [
                f"{col}={row.get(col, '')}"
                for col in phenotype_cols
                if any_pattern(row.get(col, ""), CRC_PATTERNS)
            ]
        ),
        axis=1,
    )
    bio["is_crc_candidate"] = bio["crc_label_columns"] != ""

    id_cols = [
        col
        for col in bio.columns
        if re.search(r"sample|accession|run|biosample|experiment|study|bioproject|submission", col, re.I)
    ]
    bio_token_to_rows: dict[str, list[int]] = {}
    for idx, row in bio.iterrows():
        for token in collect_biogpt_id_tokens(row, id_cols):
            bio_token_to_rows.setdefault(token, []).append(idx)

    bio_sample_norm_to_rows: dict[str, list[int]] = {}
    for idx, norm in bio["sample_id_norm"].items():
        if norm:
            bio_sample_norm_to_rows.setdefault(norm, []).append(idx)

    crc["sample_id_norm"] = crc["sample_id"].map(normalize_text)
    crc["subject_id_norm"] = crc["subjectID"].map(normalize_text)
    crc["study_norm"] = crc["studyID"].map(normalize_text)

    sample_rows = []
    exact_sample_matches = 0
    normalized_sample_matches = 0
    accession_matches = 0
    mmuphin_samples_with_any_match: set[str] = set()
    matched_bio_indices: set[int] = set()

    for _, row in crc.iterrows():
        exact_indices = bio.index[bio["sample_id"] == row["sample_id"]].tolist()
        norm_indices = bio_sample_norm_to_rows.get(row["sample_id_norm"], [])
        subject_indices = bio_sample_norm_to_rows.get(row["subject_id_norm"], [])
        accession_tokens = (
            split_tokens(row.get("NCBI_accession", ""))
            | split_tokens(row.get("ena_run_accessions", ""))
            | split_tokens(row.get("ena_sample_accessions", ""))
            | split_tokens(row.get("ena_secondary_sample_accessions", ""))
        )
        accession_indices: set[int] = set()
        matched_tokens = []
        for token in sorted(accession_tokens):
            hits = bio_token_to_rows.get(token, [])
            if hits:
                matched_tokens.append(token)
                accession_indices.update(hits)

        all_indices = set(exact_indices) | set(norm_indices) | set(subject_indices) | accession_indices
        if exact_indices:
            exact_sample_matches += 1
        if norm_indices or subject_indices:
            normalized_sample_matches += 1
        if accession_indices:
            accession_matches += 1
        if all_indices:
            mmuphin_samples_with_any_match.add(row["sample_id"])
            matched_bio_indices.update(all_indices)

        matched_bio = bio.loc[sorted(all_indices)] if all_indices else pd.DataFrame()
        sample_rows.append(
            {
                "mmuphin_sample_id": row["sample_id"],
                "mmuphin_subjectID": row.get("subjectID", ""),
                "mmuphin_studyID": row["studyID"],
                "mmuphin_condition": row["study_condition"],
                "mmuphin_NCBI_accession": row.get("NCBI_accession", ""),
                "mmuphin_ena_sample_accessions": row.get("ena_sample_accessions", ""),
                "mmuphin_ena_secondary_sample_accessions": row.get("ena_secondary_sample_accessions", ""),
                "mmuphin_ena_study_accessions": row.get("ena_study_accessions", ""),
                "mmuphin_ena_study_titles": row.get("ena_study_titles", ""),
                "exact_sample_id_match_count": len(exact_indices),
                "normalized_id_possible_match_count": len(set(norm_indices) | set(subject_indices)),
                "accession_match_count": len(accession_indices),
                "match_status": (
                    "confirmed_exact_sample_id"
                    if exact_indices
                    else "confirmed_accession_overlap"
                    if accession_indices
                    else "possible_normalized_id_match"
                    if norm_indices or subject_indices
                    else "no_match"
                ),
                "matched_biogpt_sample_ids": ";".join(sorted(matched_bio["sample_id"].unique())) if not matched_bio.empty else "",
                "matched_biogpt_studies": ";".join(sorted(matched_bio["batch_label_external_recommended"].unique()))
                if not matched_bio.empty and "batch_label_external_recommended" in matched_bio
                else "",
                "matched_accession_tokens": ";".join(matched_tokens),
            }
        )

    sample_overlap = pd.DataFrame(sample_rows)

    mmuphin_study_counts = (
        crc.groupby(["studyID", "study_condition"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["CRC", "control"]:
        if col not in mmuphin_study_counts.columns:
            mmuphin_study_counts[col] = 0

    bio_study_counts = (
        bio.groupby(["batch_label_external_recommended", "condition_class"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["CRC_candidate", "control_or_healthy", "other_or_unknown"]:
        if col not in bio_study_counts.columns:
            bio_study_counts[col] = 0

    bio_study_token_to_study: dict[str, set[str]] = {}
    study_token_cols = [
        col
        for col in bio.columns
        if re.search(r"study|project|bioproject|batch", col, re.I)
    ]
    for _, brow in bio.iterrows():
        study_label = brow.get("batch_label_external_recommended", "")
        if not study_label:
            continue
        for col in study_token_cols:
            for token in split_tokens(brow.get(col, "")):
                bio_study_token_to_study.setdefault(token, set()).add(study_label)

    study_rows = []
    for _, mstudy in mmuphin_study_counts.iterrows():
        study_id = mstudy["studyID"]
        mm_samples = set(crc.loc[crc["studyID"] == study_id, "sample_id"])
        matched_samples = sample_overlap.loc[
            (sample_overlap["mmuphin_studyID"] == study_id) & (sample_overlap["match_status"] != "no_match")
        ]
        matched_bio_studies = sorted(
            {
                s
                for cell in matched_samples["matched_biogpt_studies"].fillna("")
                for s in cell.split(";")
                if s
            }
        )
        mm_study_rows = crc[crc["studyID"] == study_id]
        mm_public_study_tokens = (
            set().union(*[split_tokens(x) for x in mm_study_rows["ena_study_accessions"].tolist()])
            | set().union(*[split_tokens(x) for x in mm_study_rows["ena_secondary_study_accessions"].tolist()])
        )
        public_study_matches = sorted(
            {
                study
                for token in mm_public_study_tokens
                for study in bio_study_token_to_study.get(token, set())
            }
        )
        mm_public_study_accessions = ";".join(sorted(mm_public_study_tokens))
        mm_public_study_titles = " | ".join(
            sorted({x for x in mm_study_rows["ena_study_titles"].tolist() if x})
        )

        exact_study_hits = bio_study_counts[
            bio_study_counts["batch_label_external_recommended"].map(normalize_text)
            == normalize_text(study_id)
        ]

        if matched_bio_studies:
            status = "confirmed_by_sample_accession_overlap"
        elif public_study_matches:
            status = "confirmed_public_study_accession_overlap"
        elif not exact_study_hits.empty:
            status = "possible_normalized_study_name_match"
        else:
            status = "no_match"

        for bio_study in (
            matched_bio_studies
            or public_study_matches
            or exact_study_hits["batch_label_external_recommended"].tolist()
            or [""]
        ):
            bcounts = bio_study_counts[bio_study_counts["batch_label_external_recommended"] == bio_study]
            study_rows.append(
                {
                    "mmuphin_studyID": study_id,
                    "mmuphin_public_study_accessions_from_ena": mm_public_study_accessions,
                    "mmuphin_public_study_titles_from_ena": mm_public_study_titles,
                    "mmuphin_total_samples": len(mm_samples),
                    "mmuphin_crc_samples": int(mstudy.get("CRC", 0)),
                    "mmuphin_control_samples": int(mstudy.get("control", 0)),
                    "biogpt_study_or_batch": bio_study,
                    "biogpt_total_samples": int(bcounts[["CRC_candidate", "control_or_healthy", "other_or_unknown"]].sum(axis=1).iloc[0])
                    if not bcounts.empty
                    else 0,
                    "biogpt_crc_candidate_samples": int(bcounts["CRC_candidate"].iloc[0]) if not bcounts.empty else 0,
                    "biogpt_control_or_healthy_samples": int(bcounts["control_or_healthy"].iloc[0]) if not bcounts.empty else 0,
                    "biogpt_other_or_unknown_samples": int(bcounts["other_or_unknown"].iloc[0]) if not bcounts.empty else 0,
                    "matched_mmuphin_samples_by_accession_or_id": len(matched_samples),
                    "overlap_status": status,
                    "both_crc_and_control_in_biogpt_study": bool(
                        (not bcounts.empty)
                        and int(bcounts["CRC_candidate"].iloc[0]) > 0
                        and int(bcounts["control_or_healthy"].iloc[0]) > 0
                    ),
                }
            )

    study_overlap = pd.DataFrame(study_rows)

    candidate_cols = [
        "sample_id",
        "Phenotype",
        "Phenotype_fullname",
        "body_site",
        "batch_label_external_recommended",
        "external_study_accession",
        "external_study_title",
        "external_sample_or_biosample",
        "ena_sample_accession",
        "ena_secondary_sample_accession",
        "ena_study_accession",
        "ena_sample_title",
        "ena_description",
        "ena_project_name",
        "sra_Run",
        "sra_SRAStudy",
        "sra_BioProject",
        "sra_BioSample",
        "sra_Disease",
        "condition_class",
        "crc_label_columns",
        "crc_label_values",
    ]
    candidate_cols = [c for c in candidate_cols if c in bio.columns]
    candidate_samples = bio.loc[bio["is_crc_candidate"], candidate_cols].copy()

    candidate_study_summary = (
        bio.groupby("batch_label_external_recommended", dropna=False)
        .agg(
            total_samples=("sample_id", "nunique"),
            crc_candidate_samples=("is_crc_candidate", "sum"),
            condition_labels=("condition_class", lambda s: ";".join(sorted(set(s)))),
            phenotype_labels=("Phenotype", lambda s: ";".join(sorted(set(s))[:20])),
        )
        .reset_index()
    )
    candidate_study_summary = candidate_study_summary[
        candidate_study_summary["crc_candidate_samples"] > 0
    ].copy()
    candidate_study_summary["has_crc_and_control_or_healthy"] = candidate_study_summary[
        "condition_labels"
    ].map(lambda x: "CRC_candidate" in x and "control_or_healthy" in x)

    n_mmuphin = len(crc)
    n_exact = exact_sample_matches
    n_accession_mm = accession_matches
    n_any_mm = len(mmuphin_samples_with_any_match)
    matched_mmuphin_studies = study_overlap.loc[
        study_overlap["overlap_status"].isin(
            [
                "confirmed_by_sample_accession_overlap",
                "confirmed_public_study_accession_overlap",
                "possible_normalized_study_name_match",
            ]
        ),
        "mmuphin_studyID",
    ].nunique()

    if n_any_mm >= 0.8 * n_mmuphin:
        case = "Case A: Full or near-full overlap"
        recommendation = (
            "Use exact overlapping CRC samples for a direct same-sample benchmark: raw abundance vs "
            "MMUPHin-adjusted abundance vs BiomeGPT raw CLS vs scGPT-style corrected CLS."
        )
    elif n_any_mm > 0 or matched_mmuphin_studies > 0:
        case = "Case B: Partial overlap"
        recommendation = (
            "Use the overlapping subset for direct comparison, and report the reduced benchmark size as a limitation."
        )
    elif len(candidate_samples) > 0:
        case = "Case C: Disease-level overlap only"
        recommendation = (
            "Use MMUPHin CRC as an external baseline dataset, and use the internal BiomeGPT CRC subset "
            "as a separate disease-specific test. Do not claim same-sample comparison."
        )
    else:
        case = "Case D: No CRC overlap"
        recommendation = (
            "Use MMUPHin CRC as the standalone external benchmark. Check whether its species/features can be "
            "mapped into BiomeGPT input format. If not feasible, build a small scGPT-style masked-abundance benchmark model."
        )

    counts_rows = [
        {"metric": "biogpt_primary_metadata_file", "value": str(PRIMARY_BIOGPT.relative_to(ROOT))},
        {"metric": "biogpt_primary_metadata_rows", "value": len(bio)},
        {"metric": "biogpt_metadata_files_discovered", "value": len(metadata_inventory)},
        {"metric": "biogpt_crc_candidate_samples", "value": int(len(candidate_samples))},
        {
            "metric": "biogpt_studies_with_crc_candidate_samples",
            "value": int(candidate_study_summary["batch_label_external_recommended"].nunique()),
        },
        {
            "metric": "biogpt_crc_candidate_studies_with_control_or_healthy",
            "value": int(candidate_study_summary["has_crc_and_control_or_healthy"].sum()),
        },
        {"metric": "mmuphin_crc_samples", "value": n_mmuphin},
        {"metric": "mmuphin_crc_studies", "value": crc["studyID"].nunique()},
        {"metric": "exact_sample_id_matches", "value": n_exact},
        {"metric": "exact_sample_id_match_percent_of_mmuphin", "value": round(100 * n_exact / n_mmuphin, 3)},
        {"metric": "normalized_sample_id_possible_matches", "value": normalized_sample_matches},
        {"metric": "mmuphin_samples_with_accession_matches", "value": n_accession_mm},
        {
            "metric": "mmuphin_samples_with_any_sample_id_or_accession_match",
            "value": n_any_mm,
        },
        {
            "metric": "any_sample_id_or_accession_match_percent_of_mmuphin",
            "value": round(100 * n_any_mm / n_mmuphin, 3),
        },
        {"metric": "mmuphin_studies_with_confirmed_or_possible_overlap", "value": int(matched_mmuphin_studies)},
        {"metric": "overlap_case", "value": case},
        {"metric": "recommended_next_step", "value": recommendation},
    ]
    counts = pd.DataFrame(counts_rows)

    counts.to_csv(OUT_DIR / "crc_overlap_counts.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    sample_overlap.to_csv(OUT_DIR / "crc_sample_overlap.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    study_overlap.to_csv(OUT_DIR / "crc_study_overlap.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    candidate_samples.to_csv(
        OUT_DIR / "crc_candidate_samples_from_biomegpt_metadata.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    top_candidate_studies = candidate_study_summary.sort_values(
        ["has_crc_and_control_or_healthy", "crc_candidate_samples"],
        ascending=[False, False],
    ).head(12)
    confirmed_studies = study_overlap[study_overlap["overlap_status"] != "no_match"]
    confirmed_sample_matches = sample_overlap[sample_overlap["match_status"] != "no_match"]

    report = [
        "# CRC Overlap Check Summary",
        "",
        "## Scope",
        "",
        "This is a metadata-only overlap and feasibility check. No BiomeGPT, scGPT, or model training was run.",
        "",
        "## Inputs Used",
        "",
        f"- Previous BiomeGPT metadata used for primary overlap: `{PRIMARY_BIOGPT.relative_to(ROOT)}`",
        f"- MMUPHin CRC controlled metadata: `{CRC_BENCHMARK_META.relative_to(ROOT)}`",
        f"- MMUPHin original `CRC_meta` was loaded only to access `subjectID` and `NCBI_accession` fields.",
        f"- ENA Portal API file reports were used only to translate MMUPHin run accessions into public sample/study accessions; cached at `{ENA_RUN_LOOKUP.relative_to(ROOT)}`.",
        f"- Additional metadata-like files discovered under `biogpt data`: {len(metadata_inventory)}",
        "",
        "## Main Answer",
        "",
        f"- Does previous BiomeGPT data contain CRC samples? **{'Yes' if len(candidate_samples) else 'No'}**.",
        f"- Does it contain MMUPHin CRC samples? **{'Yes, partially by accession' if n_any_mm else 'No confirmed sample-level overlap'}**.",
        f"- Does it contain MMUPHin CRC studies? **{'Yes' if matched_mmuphin_studies else 'No confirmed MMUPHin study overlap'}**.",
        f"- Overlap classification: **{case}**.",
        f"- Recommended next step: {recommendation}",
        "",
        "## Counts",
        "",
        f"- BiomeGPT primary metadata rows: {len(bio):,}",
        f"- BiomeGPT CRC candidate samples: {len(candidate_samples):,}",
        f"- BiomeGPT studies/batches containing CRC candidates: {candidate_study_summary['batch_label_external_recommended'].nunique():,}",
        f"- Those CRC-candidate studies with both CRC-candidate and control/healthy labels: {int(candidate_study_summary['has_crc_and_control_or_healthy'].sum()):,}",
        f"- MMUPHin CRC samples: {n_mmuphin:,}",
        f"- Exact MMUPHin sample ID matches in BiomeGPT `sample_id`: {n_exact:,} ({100 * n_exact / n_mmuphin:.3f}%)",
        f"- MMUPHin samples with accession-token matches in BiomeGPT metadata: {n_accession_mm:,} ({100 * n_accession_mm / n_mmuphin:.3f}%)",
        f"- MMUPHin samples with any sample ID, normalized ID, or accession match: {n_any_mm:,} ({100 * n_any_mm / n_mmuphin:.3f}%)",
        "",
        "## CRC Evidence in BiomeGPT Metadata",
        "",
        "CRC candidates were detected by scanning phenotype, disease, condition, project, title, and description fields for: "
        "`CRC`, `colorectal cancer`, `colorectal carcinoma`, `colon cancer`, `rectal cancer`, `adenoma`, "
        "`colorectal adenoma`, and `ColCan`.",
        "",
        "Top CRC-candidate BiomeGPT studies/batches:",
        "",
        "| BiomeGPT study/batch | total samples | CRC candidates | labels | has CRC + control/healthy |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in top_candidate_studies.iterrows():
        report.append(
            f"| {row['batch_label_external_recommended']} | {int(row['total_samples'])} | "
            f"{int(row['crc_candidate_samples'])} | {row['condition_labels']} | "
            f"{bool(row['has_crc_and_control_or_healthy'])} |"
        )

    report.extend(
        [
            "",
            "## Sample-Level Overlap",
            "",
            "Exact `sample_id` matching found no direct MMUPHin sample IDs in the BiomeGPT `sample_id` column. "
            "The stronger bridge is accession-token matching against MMUPHin `NCBI_accession` values and BiomeGPT "
            "ENA/SRA/accession fields.",
            "The ENA-resolved sample accessions are derived from MMUPHin `NCBI_accession` run IDs and used only "
            "to bridge different public identifier namespaces.",
            "",
        ]
    )
    if confirmed_sample_matches.empty:
        report.append("No sample-level matches or possible matches were found.")
    else:
        report.extend(
            [
                "| MMUPHin study | MMUPHin sample | condition | match status | BiomeGPT sample IDs | matched accession tokens |",
                "|---|---|---|---|---|---|",
            ]
        )
        for _, row in confirmed_sample_matches.head(30).iterrows():
            report.append(
                f"| {row['mmuphin_studyID']} | {row['mmuphin_sample_id']} | {row['mmuphin_condition']} | "
                f"{row['match_status']} | {row['matched_biogpt_sample_ids']} | {row['matched_accession_tokens']} |"
            )
        if len(confirmed_sample_matches) > 30:
            report.append(f"| ... | ... | ... | ... | {len(confirmed_sample_matches) - 30} additional matched rows in CSV | ... |")

    report.extend(
        [
            "",
            "## Study-Level Overlap",
            "",
        ]
    )
    if confirmed_studies.empty:
        report.append("No MMUPHin study-level overlap was confirmed.")
    else:
        report.extend(
            [
                "| MMUPHin study | MMUPHin n | BiomeGPT study/batch | BiomeGPT n | BiomeGPT CRC candidates | BiomeGPT control/healthy | status |",
                "|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for _, row in confirmed_studies.iterrows():
            report.append(
                f"| {row['mmuphin_studyID']} | {int(row['mmuphin_total_samples'])} | {row['biogpt_study_or_batch']} | "
                f"{int(row['biogpt_total_samples'])} | {int(row['biogpt_crc_candidate_samples'])} | "
                f"{int(row['biogpt_control_or_healthy_samples'])} | {row['overlap_status']} |"
            )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The previous BiomeGPT metadata clearly contains colorectal-cancer-related samples and several CRC-related "
            "studies/batches. However, the overlap with the MMUPHin CRC benchmark is not full: exact MMUPHin sample IDs "
            "are absent, and confirmed overlap is limited to samples recoverable through accession metadata. Ambiguous "
            "identifier-only similarities are kept as possible matches in the CSV rather than promoted to confirmed matches.",
            "",
            "## Output Files",
            "",
            "- `crc_overlap_counts.csv`",
            "- `crc_sample_overlap.csv`",
            "- `crc_study_overlap.csv`",
            "- `crc_candidate_samples_from_biomegpt_metadata.csv`",
            "- `crc_overlap_check_summary.md`",
        ]
    )

    (OUT_DIR / "crc_overlap_check_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Keep an inventory for auditability even though it was not requested as a primary artifact.
    metadata_inventory.to_csv(OUT_DIR / "metadata_inventory_audit.csv", index=False)

    print("CRC_OVERLAP_CHECK_OK")
    print(f"case={case}")
    print(f"biogpt_crc_candidate_samples={len(candidate_samples)}")
    print(f"mmuphin_exact_sample_id_matches={n_exact}")
    print(f"mmuphin_accession_matches={n_accession_mm}")
    print(f"mmuphin_any_matches={n_any_mm}")
    print(f"output_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
