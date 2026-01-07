# MMA-AI

AI-powered UFC fight prediction system using machine learning to predict fight outcomes with 71% accuracy.

## Description

This project implements a comprehensive UFC fight prediction pipeline that:
- Extracts and aggregates fight statistics from a UFC database
- Engineers advanced features using time-decay, opponent-adjustment, and comparative analysis
- Trains machine learning models using AutoGluon to predict fight outcomes
- Achieves target performance of 71% accuracy with proper probability calibration

## Project Structure

### Data Files
- `sqlite_scrapper.db` - UFC fight database with events, fighters, and statistics
- `fighter_aggregated_stats_with_advanced_features.csv` - Fighter-level statistics with advanced features
- `matchup_comparisons.csv` - All matchup comparison features
- `matchup_selected_features.csv` - Top ~30 selected features for modeling
- `feature_rankings.csv` - Feature importance rankings

### Scripts
- `feature_engineering_pipeline.py` - Complete feature engineering pipeline (Steps 1-5)
- `train_model.py` - Model training script using AutoGluon (Step 6)
- `predict.py` - Prediction script for new fights
- `OCR_script.py` - Utility script for OCR processing of PNG images

### Documentation
- `PIPELINE_README.md` - Detailed pipeline documentation
- `mma_ai.md` - Comprehensive analysis and methodology from MMA-AI.net

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** AutoGluon requires Python 3.8-3.11. If you have Python 3.12+, you may need to use a virtual environment with Python 3.10 or 3.11.

### 2. Feature Engineering (Optional - Already Done)

The feature engineering pipeline has already been run and generated the CSV files. If you need to regenerate features:

```bash
python feature_engineering_pipeline.py
```

This will:
- Extract base statistics from the SQLite database
- Calculate time-decayed averages (1.5 year half-life)
- Compute opponent-adjusted performance metrics
- Generate advanced aggregations (peaks, valleys, trends)
- Create matchup comparisons with difference and ratio features
- Select top ~30 features using mutual information and correlation

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load the matchup features dataset
- Split data temporally (train on older fights, test on recent fights)
- Train an ensemble model using AutoGluon
- Evaluate performance against target metrics:
  - **Accuracy:** 71%
  - **Log Loss:** 0.602
  - **Brier Score:** 0.207
- Save the trained model to `models/autogluon_model/`

Training takes about 10 minutes by default (configurable via `time_limit` parameter).

### 4. Make Predictions

```bash
# Predict on a sample of fights (for testing)
python predict.py --sample 10

# Predict on all fights in the dataset
python predict.py --input matchup_selected_features.csv

# Save predictions to a file
python predict.py --input matchup_selected_features.csv --output predictions.csv
```

## Pipeline Overview

The prediction system follows a 6-step pipeline:

1. **Data Aggregation** - Extract fight-level statistics from database
2. **Time-Decayed Averages** - Apply exponential time decay with 1.5-year half-life
3. **Opponent-Adjusted Performance** - Calculate z-scores normalized against opponent baselines
4. **Advanced Aggregations** - Add peak/valley tracking, change metrics, and trends
5. **Matchup Comparisons** - Create Fighter1 vs Fighter2 comparative features
6. **Model Training** - Train using AutoGluon with log loss optimization

See `PIPELINE_README.md` for detailed pipeline documentation.

## Performance Targets

Based on the research documented in `mma_ai.md`, the model targets:

- **Accuracy:** 71% (correctly predicting fight winners)
- **Log Loss:** 0.602 (probability calibration quality)
- **Brier Score:** 0.207 (prediction accuracy across all probability ranges)

## Key Features

The model uses approximately 30 selected features including:

- **Time-decayed statistics** - Recent performance weighted more heavily
- **Opponent-adjusted metrics** - Performance relative to opponent's typical allowances
- **Comparative features** - Differences and ratios between fighters
- **Win trajectory features** - Recent form vs career averages
- **Physical attributes** - Age, reach, weight, height differentials

Top features (by importance):
1. `diff_change_avg_win_differential` - Recent win rate change differential
2. `diff_win_differential_vs_valley` - Win performance vs career low differential  
3. `f1_change_avg_win_differential` - Fighter 1's recent win rate change
4. `diff_win_differential_vs_peak` - Win performance vs career high differential
5. `f1_win_differential_vs_peak` - Fighter 1's win performance vs their peak

## Requirements

### Core Dependencies
- Python 3.8-3.11 (for AutoGluon compatibility)
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- autogluon.tabular >= 1.0.0

### Optional Dependencies (for OCR)
- pytesseract >= 0.3.10
- Pillow >= 10.0.0

See `requirements.txt` for complete list.

## Data Sources

All data extracted from `sqlite_scrapper.db` containing:
- 757 UFC events
- 2,632 fighters
- 8,494 fights
- 39,964 round-by-round statistics
- Date range: 1994-2011 to 2025-12-13

## Methodology

The prediction methodology is based on advanced feature engineering and ensemble learning:

### Feature Engineering Layers

1. **Base Statistics** (22 categories)
   - Strikes: significant, total, head, body, leg, distance, clinch, ground
   - Grappling: takedowns, submissions, reversals, control time
   - Outcomes: knockdowns, fight duration
   - Attributes: height, weight, reach, age

2. **Time-Decayed Averages**
   - Formula: `weight = EXP(-λ × ((T - t) / 365.25))`
   - Captures recent form vs historical performance

3. **Opponent-Adjusted Performance (AdjPerf)**
   - Formula: `(fighter_stat - opponent_allowed_avg) / opponent_allowed_stddev`
   - Normalized performance against opponent's defensive capabilities
   - Includes Bayesian shrinkage for small sample sizes

4. **Advanced Aggregations**
   - Peak/valley tracking (career highs and lows)
   - Change metrics (trends and momentum)
   - Recent vs career comparisons

5. **Matchup Features**
   - Differences: Fighter1 - Fighter2
   - Ratios: Fighter1 / Fighter2
   - Applied to all engineered features

### Model Training

- **Framework:** AutoGluon (ensemble learning)
- **Optimization:** Log loss (for probability calibration)
- **Validation:** Temporal split (no data leakage)
- **Models:** Automatic ensemble of gradient boosting, neural networks, and other algorithms

## Credits

Methodology based on research from [MMA-AI.net](https://www.mma-ai.net) documenting:
- Feature engineering approaches
- Opponent-adjusted performance calculations
- Time-decay weighting strategies
- Model calibration techniques
- Betting strategy analysis

## License

This project is for educational and research purposes.

