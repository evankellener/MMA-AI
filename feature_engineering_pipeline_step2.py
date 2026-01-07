"""
Feature Engineering Pipeline - Step 2
Time-decayed averages, opponent-adjusted performance, and decayed differences.

This module computes:
1. Time-decayed averages using an exponential decay with a 1.5-year half-life.
2. Opponent-adjusted performance (AdjPerf) using opponent "allowed" history and
   robust spread (MAD) for z-scoring.
3. Decayed average difference features between fighter and opponent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Dict, List, Tuple
import json
from typing import Callable, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd

from feature_schema import export_feature_sets, load_feature_schema


HALF_LIFE_YEARS = 1.5
DECAY_LAMBDA = np.log(2.0) / HALF_LIFE_YEARS
MIN_ROBUST_SPREAD = 1e-6
ADJPERF_CLIP = 7.0

@dataclass(frozen=True)
class FeatureBuilder:
    pattern: str
    description: str
    builder: Callable[..., pd.Series]


FEATURE_BUILDERS: Dict[str, FeatureBuilder] = {
    "avg_*": FeatureBuilder(
        pattern=r"^avg_",
        description="Expanding mean of the base stat across all prior fights.",
        builder=lambda series: series.expanding().mean(),
    ),
    "recent_avg_*": FeatureBuilder(
        pattern=r"^recent_avg_",
        description="Rolling mean of the base stat over the most recent 3 fights.",
        builder=lambda series: series.rolling(window=3, min_periods=1).mean(),
    ),
    "*_differential": FeatureBuilder(
        pattern=r"_differential$",
        description="Difference between fighter and opponent values.",
        builder=lambda series, opponent: series - opponent,
    ),
    "*_per_min": FeatureBuilder(
        pattern=r"_per_min$",
        description="Normalize a cumulative stat by fight duration (minutes).",
        builder=lambda series, minutes: series / minutes.replace(0, np.nan),
    ),
    "*_acc": FeatureBuilder(
        pattern=r"_acc$",
        description="Accuracy ratio of landed over attempted values.",
        builder=lambda landed, attempted: landed / attempted.replace(0, np.nan),
    ),
    "*_def": FeatureBuilder(
        pattern=r"_def$",
        description="Defense metric based on opponent landing ratio.",
        builder=lambda opp_landed, opp_attempted: 1.0
        - (opp_landed / opp_attempted.replace(0, np.nan)),
    ),
    "*_peak": FeatureBuilder(
        pattern=r"_peak$",
        description="Peak (max) value across a fighter's history.",
        builder=lambda series: series.expanding().max(),
    ),
    "precomp_change_*": FeatureBuilder(
        pattern=r"^precomp_change_",
        description="Change between pre-competition baseline and current value.",
        builder=lambda series, baseline: series - baseline,
    ),
    "*_dec_avg": FeatureBuilder(
        pattern=r"_dec_avg$",
        description="Time-decayed average with exponential half-life.",
        builder=lambda values, weights: weighted_average(values, weights),
    ),
    "*_adjperf": FeatureBuilder(
        pattern=r"_adjperf$",
        description="Opponent-adjusted performance z-score with clipping.",
        builder=lambda observed, mean_allowed, spread: np.clip(
            (observed - mean_allowed) / spread, -ADJPERF_CLIP, ADJPERF_CLIP
        ),
    ),
}


ADDITIONAL_EXPECTED_FEATURES = [
    "Significant Strike Landing Ratio Decayed Adjusted Performance Decayed Average Difference",
]


def load_feature_registry(registry_path: str) -> List[str]:
    with open(registry_path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    required = registry.get("required_features", [])
    if not isinstance(required, list):
        raise ValueError("Feature registry required_features must be a list.")
    return required


def validate_feature_registry(
    df: pd.DataFrame,
    registry_features: Iterable[str],
    additional_expected: Iterable[str] | None = None,
) -> Tuple[List[str], List[str], List[str]]:
    df_columns = list(df.columns)
    registry_list = list(registry_features)
    missing = [feature for feature in registry_list if feature not in df_columns]
    extra = [feature for feature in df_columns if feature not in registry_list]
    additional_expected = list(additional_expected or [])
    missing_additional = [feature for feature in additional_expected if feature not in df_columns]

    print("\n=== Feature Registry Validation ===")
    print(f"Registry features: {len(registry_list)}")
    print(f"Output features: {len(df_columns)}")
    print(f"Missing features: {len(missing)}")
    print(f"Extra features: {len(extra)}")
    
    # Save each list to separate files
    with open("registry_features.txt", "w") as f:
        f.write("\n".join(registry_list))
    print(f"✓ Saved registry features to registry_features.txt")
    
    with open("output_features.txt", "w") as f:
        f.write("\n".join(df_columns))
    print(f"✓ Saved output features to output_features.txt")
    
    with open("missing_features.txt", "w") as f:
        f.write("\n".join(missing))
    print(f"✓ Saved missing features to missing_features.txt")
    
    with open("extra_features.txt", "w") as f:
        f.write("\n".join(extra))
    print(f"✓ Saved extra features to extra_features.txt")
    
    if missing:
        print("Missing feature names:")
        for feature in missing:
            print(f"- {feature}")
    if extra:
        print("Extra feature names:")
        for feature in extra:
            print(f"+ {feature}")
    print("\n=== Additional Feature Coverage ===")
    if missing_additional:
        print("Missing additional features:")
        for feature in missing_additional:
            print(f"- {feature}")
    else:
        print("All additional expected features are present.")

    return missing, extra, missing_additional



def _ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series)


def decay_weights(current_date: np.datetime64, past_dates: np.ndarray) -> np.ndarray:
    days_diff = (current_date - past_dates) / np.timedelta64(1, "D")
    years_diff = days_diff / 365.25
    return np.exp(-DECAY_LAMBDA * years_diff)


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return np.nan
    values = values[mask]
    weights = weights[mask]
    weight_sum = weights.sum()
    if weight_sum == 0:
        return np.nan
    return float(np.sum(values * weights) / weight_sum)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(cumulative, cutoff)])


def weighted_mad(values: np.ndarray, weights: np.ndarray) -> float:
    median = weighted_median(values, weights)
    if np.isnan(median):
        return np.nan
    abs_dev = np.abs(values - median)
    mad = weighted_median(abs_dev, weights)
    if np.isnan(mad):
        return np.nan
    return max(float(mad), MIN_ROBUST_SPREAD)


@dataclass
class FeatureConfig:
    stats_for_decay: List[str]
    stats_for_adjperf: List[str]
    diff_feature_map: Dict[str, str]


DEFAULT_CONFIG = FeatureConfig(
    stats_for_decay=[
        "age_at_fight",
        "sig_str_per_min",
        "total_str_per_min",
        "td_per_min",
        "sub_att_per_min",
        "kd_per_min",
        "ctrl_per_min",
        "sig_str_acc",
        "td_acc",
        "head_acc",
        "head_def",
        "sig_str_def",
        "reach_ratio",
    ],
    stats_for_adjperf=[
        "sig_str_per_min",
        "total_str_per_min",
        "td_per_min",
        "sig_str_acc",
        "head_acc",
        "head_def",
    ],
    diff_feature_map={
        "age_at_fight": "Age Decayed Average Difference",
        "reach_ratio": "Reach Ratio Decayed Average Difference",
        "head_def": "Head Defense Decayed Average Difference",
        "sig_str_per_min": "Significant Strikes Per Minute Decayed Average Difference",
        "total_str_per_min": "Total Strikes Per Minute Decayed Average Difference",
        "td_per_min": "Takedowns Per Minute Decayed Average Difference",
        "sig_str_acc": "Significant Strike Accuracy Decayed Average Difference",
        "head_acc": "Head Accuracy Decayed Average Difference",
    },
)


class TimeDecayFeatureEngineer:
    def __init__(self, half_life_years: float = HALF_LIFE_YEARS) -> None:
        self.half_life_years = half_life_years
        self.decay_lambda = np.log(2.0) / half_life_years

    def add_opponent_columns(self, df: pd.DataFrame, stats: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        df["OPPONENT"] = pd.Series(dtype=object)
        for stat in stats:
            df[f"{stat}_allowed"] = np.nan

        for (_, _), group in df.groupby(["EVENT", "BOUT"]):
            if len(group) != 2:
                continue
            idx1, idx2 = group.index
            fighter1 = group.iloc[0]["FIGHTER"]
            fighter2 = group.iloc[1]["FIGHTER"]
            df.at[idx1, "OPPONENT"] = fighter2
            df.at[idx2, "OPPONENT"] = fighter1
            for stat in stats:
                df.at[idx1, f"{stat}_allowed"] = group.iloc[1].get(stat)
                df.at[idx2, f"{stat}_allowed"] = group.iloc[0].get(stat)
        return df

    def add_defense_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["head_def"] = np.nan
        df["sig_str_def"] = np.nan

        for (_, _), group in df.groupby(["EVENT", "BOUT"]):
            if len(group) != 2:
                continue
            idx1, idx2 = group.index
            opp1_head_att = group.iloc[1].get("head_att")
            opp1_head_landed = group.iloc[1].get("head_landed")
            opp2_head_att = group.iloc[0].get("head_att")
            opp2_head_landed = group.iloc[0].get("head_landed")
            opp1_sig_att = group.iloc[1].get("sig_str_att")
            opp1_sig_landed = group.iloc[1].get("sig_str_landed")
            opp2_sig_att = group.iloc[0].get("sig_str_att")
            opp2_sig_landed = group.iloc[0].get("sig_str_landed")

            df.at[idx1, "head_def"] = (
                1.0 - (opp1_head_landed / opp1_head_att)
                if pd.notna(opp1_head_att) and opp1_head_att > 0
                else np.nan
            )
            df.at[idx2, "head_def"] = (
                1.0 - (opp2_head_landed / opp2_head_att)
                if pd.notna(opp2_head_att) and opp2_head_att > 0
                else np.nan
            )
            df.at[idx1, "sig_str_def"] = (
                1.0 - (opp1_sig_landed / opp1_sig_att)
                if pd.notna(opp1_sig_att) and opp1_sig_att > 0
                else np.nan
            )
            df.at[idx2, "sig_str_def"] = (
                1.0 - (opp2_sig_landed / opp2_sig_att)
                if pd.notna(opp2_sig_att) and opp2_sig_att > 0
                else np.nan
            )
        return df

    def add_reach_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["reach_ratio"] = np.where(
            df["HEIGHT"].notna() & (df["HEIGHT"] != 0),
            df["REACH"] / df["HEIGHT"],
            np.nan,
        )
        return df

    def add_decayed_averages(self, df: pd.DataFrame, stats: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        df["DATE"] = _ensure_datetime(df["DATE"])
        df = df.sort_values(["FIGHTER", "DATE"]).reset_index(drop=True)

        for stat in stats:
            df[f"{stat}_dec_avg"] = np.nan

        for fighter, group in df.groupby("FIGHTER"):
            indices = group.index.to_numpy()
            dates = group["DATE"].to_numpy()
            for i in range(1, len(indices)):
                current_idx = indices[i]
                current_date = dates[i]
                past_indices = indices[:i]
                past_dates = dates[:i]
                weights = decay_weights(current_date, past_dates)
                for stat in stats:
                    values = df.loc[past_indices, stat].to_numpy(dtype=float)
                    df.at[current_idx, f"{stat}_dec_avg"] = weighted_average(values, weights)
        return df

    def add_adjperf(self, df: pd.DataFrame, stats: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        df["DATE"] = _ensure_datetime(df["DATE"])
        df = df.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)
        for stat in stats:
            df[f"{stat}_adjperf"] = np.nan

        fighter_groups = {fighter: group for fighter, group in df.groupby("FIGHTER")}
        for idx, row in df.iterrows():
            opponent = row.get("OPPONENT")
            if not opponent or opponent not in fighter_groups:
                continue
            opponent_history = fighter_groups[opponent]
            current_date = np.datetime64(pd.to_datetime(row["DATE"]))
            opponent_history = opponent_history[opponent_history["DATE"] < current_date]
            if opponent_history.empty:
                continue
            past_dates = opponent_history["DATE"].to_numpy()
            weights = decay_weights(current_date, past_dates)
            for stat in stats:
                allowed_values = opponent_history[f"{stat}_allowed"].to_numpy(dtype=float)
                mean_allowed = weighted_average(allowed_values, weights)
                spread_allowed = weighted_mad(allowed_values, weights)
                if np.isnan(mean_allowed) or np.isnan(spread_allowed):
                    continue
                observed = row.get(stat)
                if pd.isna(observed):
                    continue
                adjperf = (observed - mean_allowed) / spread_allowed
                df.at[idx, f"{stat}_adjperf"] = float(
                    np.clip(adjperf, -ADJPERF_CLIP, ADJPERF_CLIP)
                )
        return df

    def add_decayed_average_differences(
        self, df: pd.DataFrame, feature_map: Dict[str, str]
    ) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)
        opponent_lookup = df.set_index(["EVENT", "BOUT", "FIGHTER"]).sort_index()

        for stat, feature_name in feature_map.items():
            df[feature_name] = np.nan
            for idx, row in df.iterrows():
                opponent = row.get("OPPONENT")
                if not opponent:
                    continue
                try:
                    opp_row = opponent_lookup.loc[(row["EVENT"], row["BOUT"], opponent)]
                    # If loc returns a DataFrame (multiple matches), take first row
                    if isinstance(opp_row, pd.DataFrame):
                        opp_row = opp_row.iloc[0]
                except KeyError:
                    continue
                fighter_dec = row.get(f"{stat}_dec_avg")
                # Access opponent value - handle both Series and scalar
                if isinstance(opp_row, pd.Series):
                    opponent_dec = opp_row.get(f"{stat}_dec_avg")
                else:
                    opponent_dec = opp_row[f"{stat}_dec_avg"]
                # Convert to scalar if Series
                if isinstance(fighter_dec, pd.Series):
                    fighter_dec = fighter_dec.iloc[0] if len(fighter_dec) > 0 else np.nan
                if isinstance(opponent_dec, pd.Series):
                    opponent_dec = opponent_dec.iloc[0] if len(opponent_dec) > 0 else np.nan
                if pd.isna(fighter_dec) or pd.isna(opponent_dec):
                    continue
                df.at[idx, feature_name] = float(fighter_dec) - float(opponent_dec)
        return df


def run_spot_checks() -> None:
    """Spot checks for decay weights, opponent baselines, and diff signs."""
    data = pd.DataFrame(
        [
            {
                "EVENT": "E1",
                "BOUT": "B1",
                "FIGHTER": "A",
                "DATE": "2020-01-01",
                "sig_str_per_min": 2.0,
                "head_landed": 10,
                "head_att": 20,
                "sig_str_landed": 30,
                "sig_str_att": 60,
                "HEIGHT": 70,
                "REACH": 72,
            },
            {
                "EVENT": "E1",
                "BOUT": "B1",
                "FIGHTER": "B",
                "DATE": "2020-01-01",
                "sig_str_per_min": 3.0,
                "head_landed": 12,
                "head_att": 24,
                "sig_str_landed": 40,
                "sig_str_att": 80,
                "HEIGHT": 72,
                "REACH": 74,
            },
            {
                "EVENT": "E1.5",
                "BOUT": "B1.5",
                "FIGHTER": "C",
                "DATE": "2020-06-01",
                "sig_str_per_min": 1.5,
                "head_landed": 4,
                "head_att": 8,
                "sig_str_landed": 15,
                "sig_str_att": 30,
                "HEIGHT": 71,
                "REACH": 73,
            },
            {
                "EVENT": "E1.5",
                "BOUT": "B1.5",
                "FIGHTER": "D",
                "DATE": "2020-06-01",
                "sig_str_per_min": 2.5,
                "head_landed": 8,
                "head_att": 10,
                "sig_str_landed": 25,
                "sig_str_att": 50,
                "HEIGHT": 69,
                "REACH": 71,
            },
            {
                "EVENT": "E2",
                "BOUT": "B2",
                "FIGHTER": "A",
                "DATE": "2021-01-01",
                "sig_str_per_min": 4.0,
                "head_landed": 15,
                "head_att": 30,
                "sig_str_landed": 50,
                "sig_str_att": 100,
                "HEIGHT": 70,
                "REACH": 72,
            },
            {
                "EVENT": "E2",
                "BOUT": "B2",
                "FIGHTER": "C",
                "DATE": "2021-01-01",
                "sig_str_per_min": 1.0,
                "head_landed": 5,
                "head_att": 10,
                "sig_str_landed": 20,
                "sig_str_att": 40,
                "HEIGHT": 71,
                "REACH": 73,
            },
        ]
    )

    engineer = TimeDecayFeatureEngineer()
    data = engineer.add_reach_ratio(data)
    data = engineer.add_defense_metrics(data)
    data = engineer.add_opponent_columns(data, ["sig_str_per_min", "head_def"])
    data = engineer.add_decayed_averages(data, ["sig_str_per_min", "head_def", "reach_ratio"])
    data = engineer.add_adjperf(data, ["sig_str_per_min", "head_def"])
    data = engineer.add_decayed_average_differences(
        data,
        {
            "sig_str_per_min": "Significant Strikes Per Minute Decayed Average Difference",
            "head_def": "Head Defense Decayed Average Difference",
        },
    )

    first_fight = data[(data["EVENT"] == "E2") & (data["FIGHTER"] == "A")].iloc[0]
    expected_weight = np.exp(-DECAY_LAMBDA * (366 / 365.25))
    expected_decay = weighted_average(np.array([2.0]), np.array([expected_weight]))
    assert np.isclose(first_fight["sig_str_per_min_dec_avg"], expected_decay)
    assert first_fight["Head Defense Decayed Average Difference"] > 0

    adjperf_value = first_fight["sig_str_per_min_adjperf"]
    assert adjperf_value <= ADJPERF_CLIP


def run_pipeline(
    input_csv: str = "fighter_aggregated_stats_with_advanced_features.csv",
    output_csv: str = "fighter_aggregated_stats_with_decayed_diffs.csv",
    config: FeatureConfig = DEFAULT_CONFIG,
    registry_path: str = "feature_registry.json",
    additional_expected_features: Iterable[str] = ADDITIONAL_EXPECTED_FEATURES,
    schema_path: str = "feature_schema.csv",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    registry_features = load_feature_registry(registry_path)
    engineer = TimeDecayFeatureEngineer()

    df = engineer.add_reach_ratio(df)
    df = engineer.add_defense_metrics(df)
    df = engineer.add_opponent_columns(df, config.stats_for_adjperf)
    df = engineer.add_decayed_averages(df, config.stats_for_decay)
    df = engineer.add_adjperf(df, config.stats_for_adjperf)
    df = engineer.add_decayed_average_differences(df, config.diff_feature_map)

    validate_feature_registry(df, registry_features, additional_expected_features)
    df.to_csv(output_csv, index=False)
    load_feature_schema(schema_path)
    export_feature_sets(df, schema_path=schema_path)
    return df


if __name__ == "__main__":
    run_spot_checks()
    run_pipeline()
