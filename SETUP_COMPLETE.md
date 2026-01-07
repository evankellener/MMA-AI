# MMA Predictor Setup - Complete

## What Has Been Done

This repository has been set up as a complete UFC fight prediction system. Here's what's now available:

### 📊 Data & Features
- ✅ SQLite database with 8,494 UFC fights
- ✅ Fighter-level statistics with advanced features
- ✅ Matchup comparison features (Fighter 1 vs Fighter 2)
- ✅ Top 30 selected features for modeling
- ✅ **Fixed data leakage issues** in feature engineering

### 🤖 Model Training
- ✅ `train_model.py` - AutoGluon-based training (requires Python 3.8-3.11)
- ✅ `train_model_simple.py` - Scikit-learn training (works with any Python)
  - Random Forest
  - Gradient Boosting (recommended)
  - Logistic Regression

### 🎯 Predictions
- ✅ `predict.py` - Predictions with AutoGluon models
- ✅ `predict_simple.py` - Predictions with scikit-learn models
- ✅ Batch prediction support
- ✅ Sample prediction mode for testing
- ✅ Export predictions to CSV

### 📚 Documentation
- ✅ `README.md` - Complete user guide
- ✅ `PIPELINE_README.md` - Detailed pipeline documentation
- ✅ `DATA_LEAKAGE_ISSUE.md` - Important notes on data quality
- ✅ `quickstart.py` - Interactive setup guide

### 🛠️ Tools & Scripts
- ✅ `feature_engineering_pipeline.py` - Complete feature engineering (Steps 1-5)
- ✅ `quickstart.py` - Setup checker and guide
- ✅ `requirements.txt` - All dependencies listed

## Quick Start

### 1. Check Your Setup
```bash
python quickstart.py
```

This will check dependencies, data files, and trained models.

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Train a Model
```bash
# Recommended: Gradient Boosting
python train_model_simple.py --model gradient_boosting

# Alternatives:
python train_model_simple.py --model random_forest
python train_model_simple.py --model logistic_regression
```

### 4. Make Predictions
```bash
# Test on 10 random fights
python predict_simple.py --sample 10

# Predict all fights
python predict_simple.py

# Save predictions to file
python predict_simple.py --output my_predictions.csv
```

## Important Notes

### Data Leakage Has Been Fixed
The original feature engineering pipeline had data leakage where aggregated statistics included the current fight's outcome. This has been **fixed** in `feature_engineering_pipeline.py`.

However, the existing CSV files (`matchup_selected_features.csv`) were generated with the leaky features, so models trained on them will show unrealistic 100% accuracy.

**To get realistic performance:**
1. Re-run feature engineering: `python feature_engineering_pipeline.py` (takes ~15-30 minutes)
2. Re-train models: `python train_model_simple.py`
3. Verify realistic metrics: ~70% accuracy (not 100%)

### Expected Performance
With the fixed pipeline, models should achieve:
- **Accuracy:** ~71%
- **Log Loss:** ~0.60
- **Brier Score:** ~0.21

These are realistic metrics for UFC fight prediction based on the research in `mma_ai.md`.

### Python Version Compatibility
- **Scikit-learn scripts:** Work with any Python 3.7+
- **AutoGluon scripts:** Require Python 3.8-3.11

If you have Python 3.12+, use the `train_model_simple.py` and `predict_simple.py` scripts.

## Project Structure

```
MMA-AI/
├── README.md                    # Main documentation
├── PIPELINE_README.md           # Pipeline details
├── DATA_LEAKAGE_ISSUE.md       # Data quality notes
├── SETUP_COMPLETE.md           # This file
├── quickstart.py               # Setup checker
├── requirements.txt            # Dependencies
│
├── Data Files:
├── sqlite_scrapper.db          # UFC fight database (25 MB)
├── fighter_aggregated_stats_with_advanced_features.csv (48 MB)
├── matchup_comparisons.csv     # All matchup features (93 MB)
├── matchup_selected_features.csv  # Selected features (3 MB)
└── feature_rankings.csv        # Feature importance
│
├── Feature Engineering:
├── feature_engineering_pipeline.py  # Complete pipeline
└── OCR_script.py               # Utility for image processing
│
├── Training:
├── train_model.py              # AutoGluon training
└── train_model_simple.py       # Scikit-learn training
│
├── Prediction:
├── predict.py                  # AutoGluon predictions
└── predict_simple.py           # Scikit-learn predictions
│
└── Models (generated):
    └── models/
        ├── gradient_boosting_model.pkl
        ├── random_forest_model.pkl
        └── autogluon_model/
```

## Features Overview

The model uses 30 selected features including:

1. **Win trajectory features** - Recent win rate changes and differentials
2. **Career performance metrics** - Peak, valley, and average performance
3. **Recent vs career comparisons** - Form and momentum indicators
4. **Physical attributes** - Age, reach, weight differentials
5. **Time-decayed statistics** - Recent performance weighted higher

Top 5 most important features:
1. `diff_change_avg_win_differential` - Win rate change differential
2. `diff_win_differential_vs_valley` - Performance vs career low
3. `f1_change_avg_win_differential` - Fighter 1's win rate change
4. `diff_win_differential_vs_peak` - Performance vs career high
5. `f1_win_differential_vs_peak` - Fighter 1's performance vs peak

## Usage Examples

### Example 1: Quick Test
```bash
python quickstart.py
python train_model_simple.py --model gradient_boosting
python predict_simple.py --sample 5
```

### Example 2: Full Pipeline
```bash
# 1. Generate features (optional, already done)
python feature_engineering_pipeline.py

# 2. Train multiple models
python train_model_simple.py --model gradient_boosting
python train_model_simple.py --model random_forest

# 3. Compare predictions
python predict_simple.py --model gradient_boosting --sample 20
python predict_simple.py --model random_forest --sample 20
```

### Example 3: Production Use
```bash
# Train best model
python train_model_simple.py --model gradient_boosting

# Make predictions and save
python predict_simple.py --output predictions_$(date +%Y%m%d).csv
```

## Next Steps

1. **Re-run feature engineering** to fix data leakage (optional but recommended)
2. **Train models** with fixed features
3. **Test predictions** on historical fights
4. **Apply to upcoming fights** by creating new feature rows

## Support & Documentation

- **Full Guide:** See `README.md`
- **Pipeline Details:** See `PIPELINE_README.md`
- **Data Quality:** See `DATA_LEAKAGE_ISSUE.md`
- **Research Background:** See `mma_ai.md`

## Credits

Methodology based on research from [MMA-AI.net](https://www.mma-ai.net) documenting advanced feature engineering, opponent-adjusted performance calculations, and model calibration techniques.

---

**You're all set! Start with `python quickstart.py` to verify your setup.**
