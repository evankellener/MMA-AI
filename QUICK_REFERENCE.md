# MMA-AI Repository - Quick Reference

## What is this repository?
A UFC fight prediction system that achieves **71% accuracy** using advanced feature engineering and machine learning.

## Key Performance Metrics
- **Accuracy:** 71%
- **Log Loss:** 0.603
- **Brier Score:** 0.208
- **ROI (Betting):** ~10-35% depending on strategy

## Core Innovation: Adjusted Performance (AdjPerf)
Normalizes fighter performance by opponent quality and weight class using:
- Time-decay weighting (1.5-year half-life)
- Bayesian shrinkage for small samples
- Robust z-scoring with MAD (Median Absolute Deviation)
- Clipping to ±7 to prevent extreme values

## How it Works (6-Step Pipeline)

1. **Data Aggregation** → 22 base stats from 39,964 fight records
2. **Time-Decay** → Recent fights weighted more heavily
3. **Opponent-Adjusted** → Normalize by opponent difficulty
4. **Comparative Features** → Create fighter1 vs fighter2 differences/ratios
5. **Feature Selection** → Select ~30 best features from 20,000+
6. **Model Training** → AutoGluon with log loss optimization

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Step 1: Aggregate fighter statistics
python feature_engineering_pipeline.py

# Step 2: Generate time-decayed and adjusted features
python feature_engineering_pipeline_step2.py
```

### Run OCR (Extract documentation)
```bash
python OCR_script.py
```

## File Structure
```
├── sqlite_scrapper.db                    # Source UFC data (26 MB)
├── feature_engineering_pipeline.py       # Step 1: Aggregation
├── feature_engineering_pipeline_step2.py # Steps 2-5: Advanced features
├── feature_schema.py                     # Feature timing classification
├── OCR_script.py                         # Extract text from images
├── requirements.txt                      # Python dependencies
├── REPOSITORY_ANALYSIS.md                # Full analysis (this is comprehensive!)
└── QUICK_REFERENCE.md                    # This file
```

## Main Data Files (Generated)
- `fighter_aggregated_stats_with_advanced_features.csv` (48 MB) - Step 1 output
- `fighter_aggregated_stats_with_decayed_diffs.csv` (120 MB) - Step 2 output
- `matchup_comparisons.csv` (93 MB) - All comparative features
- `matchup_selected_features.csv` (3.1 MB) - Selected features only
- `feature_rankings.csv` (6.7 KB) - Feature importance scores

## Top 5 Most Predictive Features
1. `diff_change_avg_win_differential` - How win rate differential has changed
2. `diff_win_differential_vs_valley` - Win differential vs. career low point
3. `f1_change_avg_win_differential` - Fighter 1's win differential change
4. `diff_win_differential_vs_peak` - Win differential vs. career high
5. `f1_win_differential_vs_peak` - Fighter 1's win differential vs. peak

## Key Statistics
- **Database:** 757 events, 2,632 fighters, 8,494 fights
- **Features:** 22 base → 59 derived → ~1,000 fighter-level → ~20,000 matchup → 30 final
- **Code:** 4 Python files, 2,264 lines

## Betting Performance
Based on backtest from Aug 2024 to present ($1,000 initial, $10 per bet):
- **Best Strategy:** AI all picks (7-day) - **ROI: 10.87%**, Final: $1,287
- **Underdog Strategy:** **ROI: 34.97%** (high risk, high reward)

## Comparison to Vegas
| Metric | Vegas (2024) | MMA-AI (Calibrated) |
|--------|-------------|---------------------|
| Accuracy | 70.0% | 71.0% |
| Log Loss | 0.563 | 0.598 |
| Brier Score | 0.194 | 0.206 |

MMA-AI slightly beats Vegas on accuracy, though Vegas has better calibration (lower log loss).

## Most Important Concept: AdjPerf Formula
```
adjperf = clip((observed - μ_shrunk) / MAD_shrunk, -7, +7)

where:
  μ_shrunk = blend of opponent history + weight-class prior
  MAD_shrunk = robust measure of spread (not std dev)
  observed = fighter's per-minute rate or accuracy in that fight
```

**Example:** Landing 10 strikes against an elite defender (adjperf +3.5) is more impressive than 20 strikes against a weak defender (adjperf +1.0).

## Technology Stack
- **Language:** Python 3.8+
- **Data Processing:** pandas, numpy
- **Database:** SQLite
- **ML:** AutoGluon, scikit-learn
- **Visualization:** matplotlib
- **OCR:** pytesseract, Pillow

## Future Improvements Recommended
1. Add unit tests for statistical functions
2. Improve error handling and logging
3. Use parquet format for large datasets
4. Create REST API for predictions
5. Implement continuous integration

## Related Documentation
- **Full Analysis:** `REPOSITORY_ANALYSIS.md` (510 lines, comprehensive)
- **Pipeline Details:** `PIPELINE_README.md` (137 lines)
- **Methodology:** `mma_ai.md` (4,543 lines, extracted from images)
- **Basic Info:** `README.md` (30 lines)

## Contact / Attribution
Repository Owner: evankellener
Created: 2024-2026
Purpose: UFC fight prediction research and betting strategy development

---

**For complete technical details, statistical methodology, and recommendations, see REPOSITORY_ANALYSIS.md**
