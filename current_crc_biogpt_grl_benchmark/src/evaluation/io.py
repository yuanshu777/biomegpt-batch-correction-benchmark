from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | None, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base or package_root()).joinpath(path).resolve()


def read_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Read the small config subset used in this package without requiring PyYAML."""
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, []).append(parse_scalar(line[4:].strip()))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            result[key] = []
            current_key = key
        else:
            result[key] = parse_scalar(value)
            current_key = key
    return result


def parse_scalar(value: str) -> Any:
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value.strip("'\"")


def read_matrix(path: str | Path, sample_id_column: str = "sample_id") -> pd.DataFrame:
    """Read sample x feature/embedding matrices.

    Also accepts the CRC abundance export format, where the first column is
    feature and the remaining columns are samples; that form is transposed.
    """
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col == sample_id_column:
        return df
    if first_col.lower() in {"feature", "taxon", "species"}:
        df = df.set_index(first_col).T.reset_index().rename(columns={"index": sample_id_column})
        return df
    if sample_id_column not in df.columns:
        df = df.rename(columns={first_col: sample_id_column})
    return df


def write_report_table(rows: list[dict[str, Any]], csv_path: str | Path, md_path: str | Path, title: str) -> None:
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    lines = [f"# {title}", ""]
    if not df.empty:
        lines.extend(df.to_markdown(index=False).splitlines())
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

