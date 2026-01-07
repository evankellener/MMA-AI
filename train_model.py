"""
MMA AI Model Training Script
Step 6: Train prediction model using AutoGluon

This script trains a UFC fight prediction model using the selected features
from the feature engineering pipeline.

Target Metrics:
- Accuracy: 71%
- Log Loss: 0.602
- Brier Score: 0.207
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("Warning: AutoGluon not installed. Install with: pip install autogluon.tabular")


class MMAPredictor:
    """MMA fight prediction model trainer."""
    
    def __init__(self, data_path='matchup_selected_features.csv', model_dir='models'):
        """
        Initialize the predictor.
        
        Args:
            data_path: Path to the matchup features CSV
            model_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.predictor = None
        
    def load_data(self):
        """Load and prepare the matchup data."""
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded data: {df.shape}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Check target distribution
        if 'fighter1_win' in df.columns:
            target_counts = df['fighter1_win'].value_counts()
            print(f"\nTarget distribution:")
            print(f"  Fighter 1 wins: {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
            print(f"  Fighter 2 wins: {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_train_test_split(self, df, test_size=0.2):
        """
        Split data into train and test sets using temporal ordering.
        
        Important: We use temporal split (not random) to prevent data leakage.
        The model should be tested on future fights it hasn't seen.
        
        Args:
            df: DataFrame with matchup data
            test_size: Fraction of data to use for testing
            
        Returns:
            train_df, test_df
        """
        print("\n" + "=" * 60)
        print("Preparing Train/Test Split")
        print("=" * 60)
        
        # Sort by date to maintain temporal ordering
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.sort_values('DATE')
        
        # Filter to rows with valid target
        df = df[df['fighter1_win'].notna()]
        
        # Temporal split: train on earlier fights, test on later fights
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"✓ Split strategy: Temporal (time-ordered)")
        print(f"  Train set: {len(train_df)} fights")
        if len(train_df) > 0:
            print(f"    Date range: {train_df['DATE'].min()} to {train_df['DATE'].max()}")
        print(f"  Test set: {len(test_df)} fights")
        if len(test_df) > 0:
            print(f"    Date range: {test_df['DATE'].min()} to {test_df['DATE'].max()}")
        
        # Identify metadata columns to exclude from training
        metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 
                        'METHOD', 'fighter1', 'fighter2']
        
        # Drop metadata columns for training
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        train_features = train_df[feature_cols].copy()
        test_features = test_df[feature_cols].copy()
        
        print(f"  Features: {len(feature_cols) - 1} (excluding target)")
        
        return train_features, test_features
    
    def train(self, train_df, time_limit=600, preset='medium_quality'):
        """
        Train the prediction model using AutoGluon.
        
        Args:
            train_df: Training data with features and target
            time_limit: Training time limit in seconds
            preset: AutoGluon preset ('best_quality', 'high_quality', 'medium_quality', 'optimize_for_deployment')
            
        Returns:
            Trained predictor
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is required for training. Install with: pip install autogluon.tabular")
        
        print("\n" + "=" * 60)
        print("Training Model with AutoGluon")
        print("=" * 60)
        print(f"Preset: {preset}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Optimization metric: log_loss (for probability calibration)")
        
        # Train the model
        self.predictor = TabularPredictor(
            label='fighter1_win',
            problem_type='binary',
            eval_metric='log_loss',  # Optimize for probability calibration
            path=str(self.model_dir / 'autogluon_model')
        )
        
        print("\nStarting training...")
        self.predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            presets=preset,
            verbosity=2
        )
        
        print("\n✓ Training complete!")
        
        # Show model leaderboard
        print("\n" + "=" * 60)
        print("Model Leaderboard")
        print("=" * 60)
        leaderboard = self.predictor.leaderboard(silent=True)
        print(leaderboard[['model', 'score_val', 'pred_time_val', 'fit_time']].head(10))
        
        return self.predictor
    
    def evaluate(self, test_df):
        """
        Evaluate the model on test data.
        
        Args:
            test_df: Test data with features and target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.predictor is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Get predictions
        y_true = test_df['fighter1_win']
        y_pred = self.predictor.predict(test_df)
        y_pred_proba = self.predictor.predict_proba(test_df)
        
        # If predict_proba returns DataFrame, extract the probability for class 1
        if isinstance(y_pred_proba, pd.DataFrame):
            if 1 in y_pred_proba.columns:
                y_pred_proba = y_pred_proba[1]
            else:
                y_pred_proba = y_pred_proba.iloc[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logloss = log_loss(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'brier_score': brier
        }
        
        print("\n=== Performance Metrics ===")
        print(f"Accuracy:     {accuracy:.3f} (target: 0.710)")
        print(f"Precision:    {precision:.3f}")
        print(f"Recall:       {recall:.3f}")
        print(f"F1 Score:     {f1:.3f}")
        print(f"Log Loss:     {logloss:.3f} (target: 0.602)")
        print(f"Brier Score:  {brier:.3f} (target: 0.207)")
        
        # Compare to targets
        print("\n=== Comparison to Targets ===")
        target_accuracy = 0.710
        target_logloss = 0.602
        target_brier = 0.207
        
        acc_diff = accuracy - target_accuracy
        print(f"Accuracy: {acc_diff:+.3f} {'✓' if accuracy >= target_accuracy else '✗'}")
        
        ll_diff = logloss - target_logloss
        print(f"Log Loss: {ll_diff:+.3f} {'✓' if logloss <= target_logloss else '✗'}")
        
        brier_diff = brier - target_brier
        print(f"Brier Score: {brier_diff:+.3f} {'✓' if brier <= target_brier else '✗'}")
        
        return metrics
    
    def save_model_info(self, metrics):
        """Save model information and metrics to file."""
        info_file = self.model_dir / 'model_info.txt'
        
        with open(info_file, 'w') as f:
            f.write("MMA AI Prediction Model\n")
            f.write("=" * 60 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n")
            f.write("\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            f.write("Target Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write("Accuracy: 0.710\n")
            f.write("Log Loss: 0.602\n")
            f.write("Brier Score: 0.207\n")
        
        print(f"\n✓ Saved model info to: {info_file}")
    
    def load_model(self):
        """Load a previously trained model."""
        model_path = self.model_dir / 'autogluon_model'
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.predictor = TabularPredictor.load(str(model_path))
        print("✓ Model loaded successfully")
        return self.predictor


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("MMA AI Model Training")
    print("=" * 60)
    print("Step 6: Train prediction model using AutoGluon")
    print()
    
    # Check if AutoGluon is available
    if not AUTOGLUON_AVAILABLE:
        print("=" * 60)
        print("AutoGluon Not Found")
        print("=" * 60)
        print("\nAutoGluon is required for training the model.")
        print("Install it with:")
        print("  pip install autogluon.tabular")
        print("\nNote: AutoGluon requires Python 3.8-3.11")
        print("=" * 60)
        return
    
    # Initialize predictor
    predictor = MMAPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Split data
    train_df, test_df = predictor.prepare_train_test_split(df, test_size=0.2)
    
    # Train model
    predictor.train(
        train_df, 
        time_limit=600,  # 10 minutes
        preset='medium_quality'  # Balance between speed and accuracy
    )
    
    # Evaluate model
    metrics = predictor.evaluate(test_df)
    
    # Save model info
    predictor.save_model_info(metrics)
    
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {predictor.model_dir / 'autogluon_model'}")
    print(f"\nTo use the model for predictions:")
    print(f"  python predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
