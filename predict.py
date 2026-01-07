"""
MMA AI Prediction Script

This script makes predictions for upcoming UFC fights using the trained model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class FightPredictor:
    """Makes predictions for UFC fights."""
    
    def __init__(self, model_dir='models/autogluon_model'):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing the trained model
        """
        self.model_dir = Path(model_dir)
        self.predictor = None
        
    def load_model(self):
        """Load the trained model."""
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon is required. Install with: pip install autogluon.tabular")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model not found at {self.model_dir}")
        
        print(f"Loading model from: {self.model_dir}")
        self.predictor = TabularPredictor.load(str(self.model_dir))
        print("✓ Model loaded successfully\n")
        
    def predict_from_features(self, features_df):
        """
        Make predictions from a DataFrame of fight features.
        
        Args:
            features_df: DataFrame with fight features (same format as training data)
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.predictor is None:
            self.load_model()
        
        # Make predictions
        predictions = self.predictor.predict(features_df)
        probabilities = self.predictor.predict_proba(features_df)
        
        # If predict_proba returns DataFrame, extract probabilities
        if isinstance(probabilities, pd.DataFrame):
            if 1 in probabilities.columns:
                proba_fighter1_win = probabilities[1]
            else:
                proba_fighter1_win = probabilities.iloc[:, 1]
        else:
            proba_fighter1_win = probabilities
        
        # Create results DataFrame
        results = pd.DataFrame({
            'fighter1_win_prediction': predictions,
            'fighter1_win_probability': proba_fighter1_win,
            'fighter2_win_probability': 1 - proba_fighter1_win
        })
        
        return results
    
    def predict_matchup(self, fighter1_features, fighter2_features):
        """
        Make a prediction for a single matchup.
        
        Args:
            fighter1_features: Dictionary of features for fighter 1
            fighter2_features: Dictionary of features for fighter 2
            
        Returns:
            Dictionary with prediction results
        """
        # Create matchup features (differences and ratios)
        matchup_features = {}
        
        # Get common features
        common_features = set(fighter1_features.keys()) & set(fighter2_features.keys())
        
        for feature in common_features:
            f1_val = fighter1_features[feature]
            f2_val = fighter2_features[feature]
            
            # Add individual values
            matchup_features[f'f1_{feature}'] = f1_val
            matchup_features[f'f2_{feature}'] = f2_val
            
            # Add difference
            if pd.notna(f1_val) and pd.notna(f2_val):
                matchup_features[f'diff_{feature}'] = f1_val - f2_val
                
                # Add ratio (avoid division by zero)
                if f2_val != 0 and abs(f2_val) > 1e-10:
                    matchup_features[f'ratio_{feature}'] = f1_val / f2_val
        
        # Create DataFrame
        features_df = pd.DataFrame([matchup_features])
        
        # Make prediction
        results = self.predict_from_features(features_df)
        
        return {
            'fighter1_wins': bool(results['fighter1_win_prediction'].iloc[0]),
            'fighter1_probability': float(results['fighter1_win_probability'].iloc[0]),
            'fighter2_probability': float(results['fighter2_win_probability'].iloc[0])
        }


def predict_from_csv(model_dir, input_file, output_file=None):
    """
    Make predictions for fights in a CSV file.
    
    Args:
        model_dir: Directory containing the trained model
        input_file: Path to CSV file with fight features
        output_file: Optional path to save predictions
    """
    print("=" * 60)
    print("MMA Fight Predictions")
    print("=" * 60)
    
    # Load predictor
    predictor = FightPredictor(model_dir)
    predictor.load_model()
    
    # Load fight data
    print(f"Loading fights from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} fights\n")
    
    # Identify metadata columns
    metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 
                    'METHOD', 'fighter1', 'fighter2', 'fighter1_win']
    
    # Extract metadata if present
    metadata = df[[col for col in metadata_cols if col in df.columns]].copy()
    
    # Extract features (non-metadata, non-target columns)
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    features = df[feature_cols].copy()
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict_from_features(features)
    
    # Combine with metadata
    results = pd.concat([metadata, predictions], axis=1)
    
    # Display results
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)
    
    for idx, row in results.iterrows():
        print(f"\nFight {idx + 1}:")
        if 'fighter1' in row and 'fighter2' in row:
            print(f"  {row['fighter1']} vs {row['fighter2']}")
        if 'EVENT' in row:
            print(f"  Event: {row['EVENT']}")
        if 'DATE' in row:
            print(f"  Date: {row['DATE']}")
        
        f1_prob = row['fighter1_win_probability']
        f2_prob = row['fighter2_win_probability']
        predicted_winner = 1 if row['fighter1_win_prediction'] else 2
        
        print(f"  Prediction: Fighter {predicted_winner} wins")
        print(f"  Fighter 1 win probability: {f1_prob:.1%}")
        print(f"  Fighter 2 win probability: {f2_prob:.1%}")
        
        # Show actual outcome if available
        if 'fighter1_win' in row and pd.notna(row['fighter1_win']):
            actual_winner = 1 if row['fighter1_win'] == 1 else 2
            correct = predicted_winner == actual_winner
            print(f"  Actual: Fighter {actual_winner} won {'✓' if correct else '✗'}")
    
    # Calculate accuracy if we have actual outcomes
    if 'fighter1_win' in results.columns:
        valid_results = results[results['fighter1_win'].notna()]
        if len(valid_results) > 0:
            correct = (valid_results['fighter1_win_prediction'] == valid_results['fighter1_win']).sum()
            accuracy = correct / len(valid_results)
            print(f"\n{'=' * 60}")
            print(f"Accuracy: {accuracy:.1%} ({correct}/{len(valid_results)})")
            print(f"{'=' * 60}")
    
    # Save results if output file specified
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\n✓ Saved predictions to: {output_file}")
    
    return results


def main():
    """Main prediction interface."""
    parser = argparse.ArgumentParser(description='Make UFC fight predictions')
    parser.add_argument('--model-dir', default='models/autogluon_model',
                       help='Directory containing trained model')
    parser.add_argument('--input', default='matchup_selected_features.csv',
                       help='CSV file with fight features')
    parser.add_argument('--output', default=None,
                       help='Output file for predictions (optional)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of random fights to predict (for testing)')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_dir)
    if not model_path.exists():
        print("=" * 60)
        print("Model Not Found")
        print("=" * 60)
        print(f"\nNo trained model found at: {model_path}")
        print("\nPlease train a model first:")
        print("  python train_model.py")
        print("=" * 60)
        return
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Load data and optionally sample
    if args.sample:
        print(f"Loading {args.sample} random fights for testing...\n")
        df = pd.read_csv(input_path)
        # Filter to fights with known outcomes for testing
        if 'fighter1_win' in df.columns:
            df = df[df['fighter1_win'].notna()]
        df_sample = df.sample(n=min(args.sample, len(df)), random_state=42)
        
        # Save sample to temp file
        temp_file = 'temp_sample.csv'
        df_sample.to_csv(temp_file, index=False)
        results = predict_from_csv(args.model_dir, temp_file, args.output)
        Path(temp_file).unlink()  # Clean up temp file
    else:
        # Predict all fights
        results = predict_from_csv(args.model_dir, args.input, args.output)


if __name__ == "__main__":
    main()
