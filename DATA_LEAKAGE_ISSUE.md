# Data Leakage Issue - IMPORTANT

## Problem

The current feature set in `matchup_selected_features.csv` appears to have **data leakage**. The model achieves 100% accuracy on the test set, which is unrealistic for UFC fight prediction.

## Root Cause

The features include aggregated statistics (e.g., `f1_win_avg`, `f2_win_avg`, `diff_win_avg`) that may include information from the current fight being predicted. Specifically:

- Features like `win_avg`, `win_peak`, `win_valley` are calculated using expanding windows
- These aggregations might include the outcome of the current fight
- This allows the model to "see" the answer during training and prediction

## Evidence

1. Training a logistic regression model achieves 100% accuracy
2. Test set predictions are 100% confident (0% or 100% probabilities)
3. No prediction errors on random samples

## Solution

The feature engineering pipeline needs to be modified to ensure all features are calculated using **only** data from fights *before* the current fight:

### In `feature_engineering_pipeline.py`:

1. **In `_add_change_features()` and `_add_peak_valley_features()`:**
   - Use `.shift(1)` to exclude the current fight from aggregations
   - Example: `df.groupby('FIGHTER')[stat].shift(1).expanding().mean()`

2. **In `_add_recent_vs_career_features()`:**
   - Ensure rolling and expanding windows don't include current fight
   - Example: `df.groupby('FIGHTER')[stat].shift(1).rolling(window=N).mean()`

3. **In `add_decayed_averages()`:**
   - Already properly excludes current fight (starts from index 1)
   - ✓ This is correct

4. **In `add_opponent_adjusted_performance()`:**
   - Should use historical opponent data only
   - Verify baselines don't include current fight outcome

### Example Fix

**Before (leaking):**
```python
df['win_avg'] = df.groupby('FIGHTER')['win'].transform(
    lambda x: x.expanding().mean()
)
```

**After (correct):**
```python
df['win_avg'] = df.groupby('FIGHTER')['win'].transform(
    lambda x: x.shift(1).expanding().mean()
)
```

## Verification

After fixing, the model should achieve:
- Accuracy: ~70% (not 100%)
- Log Loss: ~0.60
- Brier Score: ~0.20

These are realistic metrics for UFC fight prediction as documented in `mma_ai.md`.

## Current Status

⚠️ **The feature engineering pipeline needs to be fixed before the models can be trusted for real predictions.**

The trained models (`models/`) should NOT be used for actual predictions until the data leakage is fixed.

## Next Steps

1. Fix the feature engineering pipeline to eliminate data leakage
2. Re-run feature engineering: `python feature_engineering_pipeline.py`
3. Re-train models: `python train_model_simple.py`
4. Verify realistic performance metrics (70% accuracy, not 100%)

## References

- See `AdvancedAggregationsCalculator` class in `feature_engineering_pipeline.py`
- Line 821-889 contains the aggregation functions that need fixing
- Compare with `TimeDecayCalculator.add_decayed_averages()` (lines 622-729) which correctly excludes current fight
