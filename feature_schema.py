from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PRECOMP_METADATA = {
    "age",
    "date",
    "bout",
    "days_since_last_comp",
    "division",
    "dob",
    "event",
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
    "weightclass",
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
    """
    Classify a feature as precomp (pre-competition) or postcomp (post-competition).
    
    Precomp features are available before a fight and can be used for prediction.
    Postcomp features contain information about the fight outcome and should only
    be used for analysis or inference after the fight.
    
    Args:
        feature_name: Name of the feature to classify
        
    Returns:
        'precomp' or 'postcomp'
    """
    feature_name_lower = feature_name.lower()
    
    # Explicit precomp prefix
    if feature_name_lower.startswith("precomp_"):
        return "precomp"
    
    # Metadata that's known before the fight
    if feature_name in PRECOMP_METADATA:
        return "precomp"
    
    # Postcomp indicators - features that reveal fight outcomes
    postcomp_keywords = [
        'win', 'loss', 'result', 'outcome', 
        'ko', 'submission', 'decision',
        'win_streak', 'lose_streak', 'win_loss_ratio'
    ]
    
    # Check if feature name contains postcomp keywords
    for keyword in postcomp_keywords:
        if keyword in feature_name_lower:
            return "postcomp"
    
    # Statistical features from past fights are precomp
    # These include: strikes, takedowns, control, adjperf, dec_avg, etc.
    precomp_indicators = [
        'sig_str', 'total_str', 'head', 'body', 'leg',
        'distance', 'clinch', 'ground', 'takedown', 'td_',
        'sub_att', 'submission', 'reversal', 'rev_',
        'control', 'ctrl', 'knockdown', 'kd',
        'adjperf', 'dec_avg', '_per_min', '_acc', '_def',
        'absorbed', 'landed', 'attempted', 'att',
        'age', 'reach', 'height', 'weight', 'stance',
        'days_since', 'num_fights', 'fight_count'
    ]
    
    for indicator in precomp_indicators:
        if indicator in feature_name_lower:
            return "precomp"
    
    # Default to postcomp for safety (avoids leaking info)
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


def resolve_feature_timing(feature_name: str, schema_map: Dict[str, str]) -> str:
    if feature_name in schema_map:
        return schema_map[feature_name]
    lower_name = feature_name.lower()
    if lower_name in schema_map:
        return schema_map[lower_name]
    if lower_name.startswith("opp_"):
        base_name = lower_name.removeprefix("opp_")
        if base_name in schema_map:
            return schema_map[base_name]
    return classify_feature(lower_name)


def split_dataframe_by_schema(df, schema_map: Dict[str, str]):
    timings = {col: resolve_feature_timing(col, schema_map) for col in df.columns}
    precomp_cols = [col for col, timing in timings.items() if timing == "precomp"]
    postcomp_cols = [col for col, timing in timings.items() if timing == "postcomp"]
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
