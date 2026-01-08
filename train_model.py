#!/usr/bin/env python3
"""
MMA Fight Prediction Model Training Script

This script trains a fight prediction model using the cleaned precomp features.
It includes:
- Proper train/test split (temporal to avoid leakage)
- Model training with AutoGluon
- Cross-validation
- Feature importance analysis
- Performance evaluation

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json


def load_data(csv_path='matchup_selected_features.csv'):
    """Load the training data with precomp features only."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} fights from {csv_path}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"✓ Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    return df


def prepare_features(df):
    """Prepare features and target for modeling."""
    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)
    
    # Metadata columns to exclude from features
    metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 'METHOD', 
                    'fighter1', 'fighter2', 'fighter1_win']
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Target
    target_col = 'fighter1_win'
    
    print(f"✓ Feature columns: {len(feature_cols)}")
    print(f"✓ Target column: {target_col}")
    
    # Check for missing values
    missing = df[feature_cols].isnull().sum().sum()
    print(f"✓ Missing values in features: {missing}")
    
    # Target distribution
    target_dist = df[target_col].value_counts()
    print(f"\n✓ Target distribution:")
    print(f"  fighter1_win = 1: {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  fighter1_win = 0: {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
    
    return feature_cols, target_col


def temporal_train_test_split(df, test_size=0.2, val_size=0.1):
    """
    Split data temporally to avoid information leakage.
    
    Args:
        df: DataFrame with DATE column
        test_size: Fraction of most recent data for test set
        val_size: Fraction of remaining data for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    print("\n" + "=" * 70)
    print("TEMPORAL TRAIN/VAL/TEST SPLIT")
    print("=" * 70)
    
    # Sort by date
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split
    train_df = df.iloc[:val_idx].copy()
    val_df = df.iloc[val_idx:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()
    
    print(f"✓ Total fights: {n}")
    print(f"\n✓ Train set: {len(train_df)} fights ({len(train_df)/n*100:.1f}%)")
    print(f"  Date range: {train_df['DATE'].min()} to {train_df['DATE'].max()}")
    
    print(f"\n✓ Validation set: {len(val_df)} fights ({len(val_df)/n*100:.1f}%)")
    print(f"  Date range: {val_df['DATE'].min()} to {val_df['DATE'].max()}")
    
    print(f"\n✓ Test set: {len(test_df)} fights ({len(test_df)/n*100:.1f}%)")
    print(f"  Date range: {test_df['DATE'].min()} to {test_df['DATE'].max()}")
    
    return train_df, val_df, test_df


def train_autogluon_model(train_df, val_df, feature_cols, target_col, time_limit=300):
    """
    Train model using AutoGluon.
    
    Args:
        train_df: Training data
        val_df: Validation data (used for tuning)
        feature_cols: List of feature column names
        target_col: Target column name
        time_limit: Training time limit in seconds
        
    Returns:
        predictor: Trained AutoGluon predictor
    """
    print("\n" + "=" * 70)
    print("TRAINING AUTOGLUON MODEL")
    print("=" * 70)
    
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("✗ AutoGluon not installed. Install with:")
        print("  pip install autogluon")
        return None
    
    # Prepare data for AutoGluon
    train_data = train_df[feature_cols + [target_col]].copy()
    val_data = val_df[feature_cols + [target_col]].copy()
    
    # Combine train and val for AutoGluon (it will handle internal validation)
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    
    # Handle missing values - fill with 0 (reasonable for fighter stats)
    missing_before = combined_data[feature_cols].isnull().sum().sum()
    if missing_before > 0:
        print(f"⚠ Filling {missing_before} missing values with 0")
        combined_data[feature_cols] = combined_data[feature_cols].fillna(0)
    
    print(f"✓ Training data: {len(combined_data)} fights")
    print(f"✓ Time limit: {time_limit} seconds")
    print(f"✓ Evaluation metric: log_loss")
    
    # Train with error handling - use verbosity=0 to avoid Mac GPU detection crash
    try:
        predictor = TabularPredictor(
            label=target_col,
            eval_metric='log_loss',
            path='./autogluon_models'
        ).fit(
            combined_data,
            time_limit=time_limit,
            presets='best_quality',
            num_gpus=0,  # Disable GPU to avoid detection issues
            verbosity=0  # Reduced verbosity to avoid Mac GPU detection crash
        )
        
        print(f"\n✓ Model trained successfully")
        print(f"✓ Best model: {predictor.get_model_best()}")
        
        return predictor
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        print("\nTrying with 'medium_quality' preset instead...")
        
        try:
            predictor = TabularPredictor(
                label=target_col,
                eval_metric='log_loss',
                path='./autogluon_models_fallback'
            ).fit(
                combined_data,
                time_limit=time_limit,
                presets='medium_quality',
                num_gpus=0,
                verbosity=0
            )
            
            print(f"\n✓ Model trained successfully with fallback preset")
            print(f"✓ Best model: {predictor.get_model_best()}")
            
            return predictor
        
        except Exception as e2:
            print(f"\n✗ Training failed: {e2}")
            print("\nPlease check:")
            print("  1. AutoGluon is properly installed")
            print("  2. Sufficient memory is available")
            print("  3. Data format is correct")
            return None


def evaluate_model(predictor, test_df, feature_cols, target_col):
    """
    Evaluate model on test set.
    
    Args:
        predictor: Trained predictor
        test_df: Test data
        feature_cols: Feature columns
        target_col: Target column
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    if predictor is None:
        print("✗ No model to evaluate")
        return None
    
    # Prepare test data
    test_data = test_df[feature_cols + [target_col]].copy()
    
    # Handle missing values
    missing_test = test_data[feature_cols].isnull().sum().sum()
    if missing_test > 0:
        print(f"⚠ Filling {missing_test} missing values in test set with 0")
        test_data[feature_cols] = test_data[feature_cols].fillna(0)
    
    # Evaluate
    metrics = predictor.evaluate(test_data)
    
    print(f"\n✓ Test Set Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get predictions
    y_pred_proba = predictor.predict_proba(test_data)
    y_pred = predictor.predict(test_data)
    y_true = test_data[target_col]
    
    # Calculate additional metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[1])
    logloss = log_loss(y_true, y_pred_proba[1])
    brier = brier_score_loss(y_true, y_pred_proba[1])
    
    print(f"\n✓ Additional Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    
    # Compare to baseline (always predict majority class)
    baseline_accuracy = max(y_true.value_counts()) / len(y_true)
    print(f"\n✓ Baseline (majority class): {baseline_accuracy:.4f}")
    print(f"✓ Improvement: {(accuracy - baseline_accuracy)*100:.2f} percentage points")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': logloss,
        'brier_score': brier,
        'baseline_accuracy': baseline_accuracy
    }


def analyze_feature_importance(predictor, feature_cols):
    """
    Analyze feature importance.
    
    Args:
        predictor: Trained predictor
        feature_cols: Feature columns
        
    Returns:
        importance_df: DataFrame with feature importance
    """
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    if predictor is None:
        print("✗ No model to analyze")
        return None
    
    # Get feature importance
    importance = predictor.feature_importance(predictor.get_model_best())
    
    print(f"\n✓ Top 20 Most Important Features:")
    for idx, (feature, score) in enumerate(importance.head(20).items(), 1):
        print(f"  {idx:2d}. {feature:50s} {score:8.4f}")
    
    # Save to file
    importance.to_csv('feature_importance.csv', header=['importance'])
    print(f"\n✓ Saved full importance to: feature_importance.csv")
    
    return importance


def save_model_report(metrics, importance, test_df):
    """Save model training report."""
    print("\n" + "=" * 70)
    print("SAVING MODEL REPORT")
    print("=" * 70)
    
    report = {
        'training_date': datetime.now().isoformat(),
        'test_set_size': len(test_df),
        'test_date_range': {
            'start': str(test_df['DATE'].min()),
            'end': str(test_df['DATE'].max())
        },
        'metrics': metrics,
        'top_10_features': importance.head(10).to_dict() if importance is not None else {}
    }
    
    with open('model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Saved model report to: model_report.json")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("MMA FIGHT PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load data
    df = load_data('matchup_selected_features.csv')
    
    # 2. Prepare features
    feature_cols, target_col = prepare_features(df)
    
    # 3. Temporal split
    train_df, val_df, test_df = temporal_train_test_split(df, test_size=0.2, val_size=0.1)
    
    # 4. Train model
    predictor = train_autogluon_model(
        train_df, val_df, feature_cols, target_col, 
        time_limit=300  # 5 minutes
    )
    
    # 5. Evaluate
    metrics = evaluate_model(predictor, test_df, feature_cols, target_col)
    
    # 6. Feature importance
    importance = analyze_feature_importance(predictor, feature_cols)
    
    # 7. Save report
    if metrics and importance is not None:
        save_model_report(metrics, importance, test_df)
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if predictor:
        print(f"\n✓ Model saved to: ./autogluon_models")
        print(f"✓ To load later: TabularPredictor.load('./autogluon_models')")


if __name__ == "__main__":
    main()
