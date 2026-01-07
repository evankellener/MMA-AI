"""
MMA AI Simple Model Training Script
Alternative training using scikit-learn (works with any Python version)

This is a simpler alternative to train_model.py that doesn't require AutoGluon.
Uses RandomForest and GradientBoosting from scikit-learn.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class SimpleMLPredictor:
    """Simple MMA fight prediction model using scikit-learn."""
    
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
        self.model = None
        self.model_name = None
        
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
        
        Args:
            df: DataFrame with matchup data
            test_size: Fraction of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
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
        
        # Temporal split
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
        
        # Identify metadata columns to exclude
        metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 
                        'METHOD', 'fighter1', 'fighter2', 'fighter1_win']
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Extract features and target
        X_train = train_df[feature_cols].fillna(0)
        X_test = test_df[feature_cols].fillna(0)
        y_train = train_df['fighter1_win'].astype(int)
        y_test = test_df['fighter1_win'].astype(int)
        
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, model_type='gradient_boosting'):
        """
        Train the prediction model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic_regression')
            
        Returns:
            Trained model
        """
        print("\n" + "=" * 60)
        print("Training Model")
        print("=" * 60)
        print(f"Model type: {model_type}")
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            self.model_name = 'random_forest'
            
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                verbose=1
            )
            self.model_name = 'gradient_boosting'
            
        else:  # logistic_regression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                verbose=1,
                n_jobs=-1
            )
            self.model_name = 'logistic_regression'
        
        print("\nStarting training...")
        self.model.fit(X_train, y_train)
        
        print("\n✓ Training complete!")
        
        # Show feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "=" * 60)
            print("Top 10 Feature Importances")
            print("=" * 60)
            importances = self.model.feature_importances_
            feature_names = X_train.columns
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices):
                print(f"{i+1:2d}. {feature_names[idx]:50s} {importances[idx]:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        logloss = log_loss(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        
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
        
        # Show classification report
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=['Fighter 2 wins', 'Fighter 1 wins']))
        
        return metrics
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_file = self.model_dir / f'{self.model_name}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n✓ Saved model to: {model_file}")
        return model_file
    
    def save_model_info(self, metrics):
        """Save model information and metrics to file."""
        info_file = self.model_dir / f'{self.model_name}_info.txt'
        
        with open(info_file, 'w') as f:
            f.write(f"MMA AI Prediction Model - {self.model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n")
            f.write(f"Model Type: {self.model_name}\n")
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
        
        print(f"✓ Saved model info to: {info_file}")
    
    def load_model(self, model_name='gradient_boosting'):
        """Load a previously trained model."""
        model_file = self.model_dir / f'{model_name}_model.pkl'
        if not model_file.exists():
            raise FileNotFoundError(f"No model found at {model_file}")
        
        print(f"Loading model from: {model_file}")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.model_name = model_name
        print("✓ Model loaded successfully")
        return self.model


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MMA prediction model')
    parser.add_argument('--model', default='gradient_boosting',
                       choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                       help='Type of model to train')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MMA AI Model Training (scikit-learn)")
    print("=" * 60)
    print("Alternative training script using scikit-learn")
    print(f"Model: {args.model}")
    print()
    
    # Initialize predictor
    predictor = SimpleMLPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.prepare_train_test_split(df, test_size=0.2)
    
    # Train model
    predictor.train(X_train, y_train, model_type=args.model)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    
    # Save model and info
    predictor.save_model()
    predictor.save_model_info(metrics)
    
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {predictor.model_dir}")
    print(f"\nTo use the model for predictions:")
    print(f"  python predict_simple.py --model {args.model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
