"""
MMA AI Simple Prediction Script
Alternative prediction script for models trained with train_model_simple.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle


class SimpleFightPredictor:
    """Makes predictions using scikit-learn models."""
    
    def __init__(self, model_path):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model (.pkl file)
        """
        self.model_path = Path(model_path)
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully\n")
        
    def predict_from_features(self, features_df):
        """
        Make predictions from a DataFrame of fight features.
        
        Args:
            features_df: DataFrame with fight features
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            self.load_model()
        
        # Make predictions
        predictions = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)
        
        # Extract probabilities for class 1 (Fighter 1 wins)
        proba_fighter1_win = probabilities[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'fighter1_win_prediction': predictions,
            'fighter1_win_probability': proba_fighter1_win,
            'fighter2_win_probability': 1 - proba_fighter1_win
        })
        
        return results


def predict_from_csv(model_path, input_file, output_file=None):
    """
    Make predictions for fights in a CSV file.
    
    Args:
        model_path: Path to the saved model
        input_file: Path to CSV file with fight features
        output_file: Optional path to save predictions
    """
    print("=" * 60)
    print("MMA Fight Predictions")
    print("=" * 60)
    
    # Load predictor
    predictor = SimpleFightPredictor(model_path)
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
    
    # Extract features
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    features = df[feature_cols].fillna(0)
    
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
    parser.add_argument('--model', default='gradient_boosting',
                       help='Model type (random_forest, gradient_boosting, logistic_regression)')
    parser.add_argument('--model-file', default=None,
                       help='Direct path to model file (overrides --model)')
    parser.add_argument('--input', default='matchup_selected_features.csv',
                       help='CSV file with fight features')
    parser.add_argument('--output', default=None,
                       help='Output file for predictions (optional)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of random fights to predict (for testing)')
    
    args = parser.parse_args()
    
    # Determine model file path
    if args.model_file:
        model_path = args.model_file
    else:
        model_path = f'models/{args.model}_model.pkl'
    
    # Check if model exists
    if not Path(model_path).exists():
        print("=" * 60)
        print("Model Not Found")
        print("=" * 60)
        print(f"\nNo trained model found at: {model_path}")
        print("\nPlease train a model first:")
        print(f"  python train_model_simple.py --model {args.model}")
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
        results = predict_from_csv(model_path, temp_file, args.output)
        Path(temp_file).unlink()  # Clean up temp file
    else:
        # Predict all fights
        results = predict_from_csv(model_path, args.input, args.output)


if __name__ == "__main__":
    main()
