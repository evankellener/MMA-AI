# MMA AI Feature Engineering Pipeline

This repository contains the implementation of a comprehensive feature engineering pipeline for UFC fight prediction, designed to recreate the 71% accuracy model described in mma_ai.md.

## Pipeline Overview

The feature engineering process transforms 22 base statistics from the UFC database into 20,000+ features, then selects the ~30 most influential features for prediction.

### Target Performance Metrics
- **Accuracy**: 71%
- **Log Loss**: 0.602
- **Brier Score**: 0.207

## Implementation Status

### ✅ Step 1: Data Pipeline - Fight-Level to Fighter-Level Aggregation

**Status**: Complete

**File**: `feature_engineering_pipeline.py`

**What it does**:
- Extracts 22 base statistics from SQLite database tables
- Parses and normalizes fight statistics (strikes, takedowns, control time, etc.)
- Aggregates round-level data to fight-level totals
- Creates fighter-level chronological history
- Calculates per-minute rates and accuracy metrics
- Merges with fighter physical attributes (height, weight, reach, age)

**Base Statistics Extracted** (22 categories):
1. Significant strikes (landed/attempted)
2. Total strikes (landed/attempted)
3. Head strikes (landed/attempted)
4. Body strikes (landed/attempted)
5. Leg strikes (landed/attempted)
6. Distance strikes (landed/attempted)
7. Clinch strikes (landed/attempted)
8. Ground strikes (landed/attempted)
9. Takedowns (landed/attempted)
10. Takedown accuracy
11. Submission attempts
12. Reversals
13. Control time
14. Knockdowns
15. Height
16. Weight
17. Reach
18. Stance
19. Age at fight
20. Fight duration
21. Days since last fight
22. Total fights

**Output**: `fighter_aggregated_stats.csv`
- 16,944 fight records
- 2,627 unique fighters
- 59 features per record
- Date range: 1994-2011 to 2025-12-13

**Derived Features** (16 initial):
- Per-minute rates (7): strikes, takedowns, submissions, reversals, control, knockdowns
- Accuracy rates (9): striking accuracy across all target zones

**Usage**:
```bash
python feature_engineering_pipeline.py
```

### 🔄 Step 2: Time-Decayed Averages (Planned)

**Status**: Not yet implemented

**What it will do**:
- Calculate exponentially time-weighted averages with 1.5-year half-life
- Formula: `weight = EXP(-λ × ((T - t) / 365.25))`
- Give more weight to recent fights vs. historical performance
- Apply to all base statistics and derived features

### 🔄 Step 3: Opponent-Adjusted Performance (AdjPerf) (Planned)

**Status**: Not yet implemented

**What it will do**:
- Calculate z-score normalized performance: `(fighter_stat - opponent_allowed_avg) / opponent_allowed_stddev`
- Build historical baselines for what each opponent typically allows
- Apply Bayesian shrinkage for small sample sizes
- Create opponent-aware performance metrics

### 🔄 Step 4: Comparative Features (Planned)

**Status**: Not yet implemented

**What it will do**:
- Generate fighter1 vs fighter2 difference features
- Generate fighter1 vs fighter2 ratio features
- Apply to both standard and opponent-adjusted statistics
- Create the full feature space (~20,000 features)

### 🔄 Step 5: Feature Selection (Planned)

**Status**: Not yet implemented

**What it will do**:
- Use mutual information, SHAP values, or recursive feature elimination
- Select ~30 most predictive features
- Target features identified from analysis:
  - Age Decayed Average Difference
  - Significant Strike Landing Ratio Decayed Adjusted Performance Decayed Average Difference
  - Reach Ratio Decayed Average Difference
  - (and 28 more...)

### 🔄 Step 6: Model Training (Planned)

**Status**: Not yet implemented

**What it will do**:
- Train using AutoGluon (as mentioned in mma_ai.md)
- Optimize for log loss to achieve proper calibration
- Target 71% accuracy, 0.602 log loss, 0.207 Brier score

## Data Source

All data is extracted from `sqlite_scrapper.db` containing:
- 757 UFC events
- 2,632 fighters
- 8,494 fights
- 39,964 round-by-round statistics

## Requirements

```bash
pip install -r requirements.txt
```

## Next Steps

Continue with Step 2: Implement time-decayed averaging to capture recent fighter form.