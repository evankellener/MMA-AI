from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PRECOMP_METADATA = {
    "age",
    "date",
    "days_since_last_comp",
    "division",
    "dob",
    "event_url",
    "fight_url",
    "fighter",
    "fighter_url",
    "height",
    "opponent",
    "opponent_url",
    "reach",
    "stance",
    "time_format",
    "title_fight",
    "weight",
}


def load_canonical_features(
    master_csv_path: str = "masterMLpublic100.csv.csv",
    registry_path: str = "feature_registry.json",
) -> List[str]:
    master_path = Path(master_csv_path)
    if master_path.exists():
        with master_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return next(reader)

    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Could not find master CSV ({master_csv_path}) or registry JSON ({registry_path})."
        )
    with registry_path.open(encoding="utf-8") as handle:
        registry = json.load(handle)
    required = registry.get("required_features", [])
    if not isinstance(required, list):
        raise ValueError("Feature registry required_features must be a list.")
    return required


def classify_feature(feature_name: str) -> str:
    if feature_name.startswith("precomp_"):
        return "precomp"
    if feature_name in PRECOMP_METADATA:
        return "precomp"
    return "postcomp"


def build_feature_schema(features: Iterable[str]) -> List[Tuple[str, str]]:
    return [(feature, classify_feature(feature)) for feature in features]


def write_feature_schema(
    schema_path: str = "feature_schema.csv",
    master_csv_path: str = "masterMLpublic100.csv.csv",
    registry_path: str = "feature_registry.json",
) -> List[Tuple[str, str]]:
    features = load_canonical_features(master_csv_path, registry_path)
    schema = build_feature_schema(features)
    with Path(schema_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "timing"])
        writer.writerows(schema)
    return schema


def load_feature_schema(schema_path: str = "feature_schema.csv") -> Dict[str, str]:
    schema_file = Path(schema_path)
    if not schema_file.exists():
        schema = write_feature_schema(schema_path=schema_path)
        return {feature: timing for feature, timing in schema}
    with schema_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["feature"]: row["timing"] for row in reader}


def split_dataframe_by_schema(df, schema_map: Dict[str, str]):
    precomp_cols = [col for col in df.columns if schema_map.get(col) == "precomp"]
    postcomp_cols = [col for col in df.columns if schema_map.get(col) == "postcomp"]
    precomp_df = df[precomp_cols].copy()
    postcomp_df = df[postcomp_cols].copy()
    return precomp_df, postcomp_df


def validate_precomp_export(precomp_columns: Iterable[str], schema_map: Dict[str, str]):
    invalid = [col for col in precomp_columns if schema_map.get(col) != "precomp"]
    if invalid:
        raise ValueError(
            "Precomp export includes postcomp-only features: "
            + ", ".join(sorted(invalid))
        )


def export_feature_sets(
    df,
    schema_path: str = "feature_schema.csv",
    precomp_output: str = "features_precomp.csv",
    postcomp_output: str = "features_postcomp.csv",
) -> Tuple[str, str]:
    schema_map = load_feature_schema(schema_path)
    precomp_df, postcomp_df = split_dataframe_by_schema(df, schema_map)
    validate_precomp_export(precomp_df.columns, schema_map)
    precomp_df.to_csv(precomp_output, index=False)
    postcomp_df.to_csv(postcomp_output, index=False)
    return precomp_output, postcomp_output
