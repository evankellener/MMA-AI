# MMA Fight Prediction - Model Training Guide

## Quick Start

### 1. Install AutoGluon (if not already installed)
```bash
pip install autogluon
```

### 2. Run the Training Script
```bash
python train_model.py
```

This will:
- Load `matchup_selected_features.csv` (30 precomp features, no data leakage)
- Split data temporally (72% train, 8% validation, 20% test)
- Train AutoGluon model (5 minute time limit by default)
- Evaluate on test set
- Generate feature importance analysis
- Save model and report

## Training Pipeline Details

### Data Split Strategy

**Temporal Split (No Leakage):**
```
├── Train Set (72%): Oldest fights
├── Validation Set (8%): Middle period  
└── Test Set (20%): Most recent fights
```

This ensures:
- No future information leaks into training
- Model is evaluated on truly unseen future fights
- Realistic assessment of prediction performance

### Model Training

**AutoGluon Configuration:**
- **Metric:** Log loss (for well-calibrated probabilities)
- **Preset:** `best_quality` (explores multiple model types)
- **Time Limit:** 300 seconds (5 minutes) - adjustable
- **Models Tried:** LightGBM, CatBoost, Random Forest, Neural Networks, etc.

**To adjust training time:**
```python
# In train_model.py, modify:
predictor = train_autogluon_model(
    train_df, val_df, feature_cols, target_col, 
    time_limit=600  # 10 minutes instead of 5
)
```

### Evaluation Metrics

The script reports:
1. **Accuracy:** Correct prediction rate
2. **AUC-ROC:** Area under ROC curve (discrimination ability)
3. **Log Loss:** Prediction confidence quality
4. **Brier Score:** Probability calibration quality
5. **Baseline Comparison:** Improvement over always predicting majority class

### Feature Importance

Analyzes which features contribute most to predictions:
- Saves to `feature_importance.csv`
- Shows top 20 in console output
- Helps understand model behavior

## Output Files

After training, you'll have:

```
./
├── autogluon_models/          # Trained model directory
├── feature_importance.csv     # Feature importance scores
└── model_report.json          # Training summary and metrics
```

## Advanced Usage

### Custom Train/Test Split

Modify in `train_model.py`:
```python
train_df, val_df, test_df = temporal_train_test_split(
    df, 
    test_size=0.15,   # 15% test instead of 20%
    val_size=0.15     # 15% validation instead of 10%
)
```

### Using All Features (1,019 precomp)

Instead of selected 30 features:
```python
# In train_model.py, change:
df = load_data('matchup_all_features_precomp.csv')
```

**Note:** More features = longer training time and potential overfitting. Start with the selected 30.

### Loading Trained Model

To use the trained model later:
```python
from autogluon.tabular import TabularPredictor

# Load model
predictor = TabularPredictor.load('./autogluon_models')

# Make predictions
import pandas as pd
new_data = pd.read_csv('new_fights.csv')
predictions = predictor.predict_proba(new_data)
```

## Cross-Validation

For more robust evaluation, use time-series cross-validation:

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

def cross_validate_temporal(df, feature_cols, target_col, n_splits=5):
    """Perform temporal cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    df_sorted = df.sort_values('DATE')
    X = df_sorted[feature_cols]
    y = df_sorted[target_col]
    
    cv_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        train_data = df_sorted.iloc[train_idx]
        test_data = df_sorted.iloc[test_idx]
        
        # Train and evaluate (using AutoGluon or other)
        # ... your training code here
        
    return cv_scores
```

## Performance Expectations

Based on the repository analysis:

**Target Performance:**
- Accuracy: ~71% (better than Vegas 70%)
- Log Loss: ~0.60
- Brier Score: ~0.21

**Baseline:**
- Always predicting majority class: ~50-53%
- Random guessing: 50%

**Good Model:**
- Accuracy > 65%
- Log Loss < 0.65
- Better than baseline by 10+ percentage points

## Troubleshooting

### "AutoGluon not installed"
```bash
pip install autogluon
```

### "Out of memory"
- Reduce `time_limit` in training
- Use fewer features (selected 30 instead of all 1,019)
- Reduce `n_splits` in cross-validation

### "Training takes too long"
- Reduce `time_limit` parameter
- Use `presets='medium_quality'` instead of `'best_quality'`
- Ensure you're using `matchup_selected_features.csv` (30 features) not all features

### "Poor performance"
- Check for data leakage (run `python3 test_feature_selection.py`)
- Ensure temporal split is used (not random)
- Try different features or feature engineering
- Increase training time

## Next Steps

1. **Run Initial Training:**
   ```bash
   python train_model.py
   ```

2. **Review Results:**
   - Check console output for metrics
   - Review `model_report.json`
   - Analyze `feature_importance.csv`

3. **Iterate:**
   - Adjust training time if needed
   - Try different feature sets
   - Experiment with model configurations

4. **Deploy:**
   - Use trained model for predictions
   - Set up prediction pipeline
   - Monitor performance on new fights

## References

- **AutoGluon Documentation:** https://auto.gluon.ai/stable/index.html
- **Repository Analysis:** See `REPOSITORY_ANALYSIS.md`
- **Feature Selection Test:** Run `python3 test_feature_selection.py`
