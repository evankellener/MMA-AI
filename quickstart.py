#!/usr/bin/env python3
"""
Quick Start Guide for MMA-AI Predictor

This script helps you get started with the MMA prediction system.
"""

import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    missing = []
    
    try:
        import pandas
        print("✓ pandas installed")
    except ImportError:
        missing.append("pandas")
        print("✗ pandas not installed")
    
    try:
        import numpy
        print("✓ numpy installed")
    except ImportError:
        missing.append("numpy")
        print("✗ numpy not installed")
    
    try:
        import sklearn
        print("✓ scikit-learn installed")
    except ImportError:
        missing.append("scikit-learn")
        print("✗ scikit-learn not installed")
    
    try:
        from autogluon.tabular import TabularPredictor
        print("✓ autogluon installed (optional, enables advanced training)")
    except ImportError:
        print("⚠ autogluon not installed (optional)")
        print("  For advanced training, install with: pip install autogluon.tabular")
        print("  Note: Requires Python 3.8-3.11")
    
    if missing:
        print("\n" + "-" * 70)
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install " + " ".join(missing))
        return False
    
    return True


def check_data_files():
    """Check if required data files exist."""
    print_header("Checking Data Files")
    
    required_files = [
        'sqlite_scrapper.db',
        'matchup_selected_features.csv',
        'fighter_aggregated_stats_with_advanced_features.csv'
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    if not all_exist:
        print("\n" + "-" * 70)
        print("Missing data files. Run feature engineering:")
        print("  python feature_engineering_pipeline.py")
    
    return all_exist


def check_trained_models():
    """Check if trained models exist."""
    print_header("Checking Trained Models")
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("✗ No models directory found")
        print("\nTrain a model with:")
        print("  python train_model_simple.py")
        return False
    
    model_files = list(models_dir.glob('*.pkl'))
    autogluon_model = models_dir / 'autogluon_model'
    
    if model_files:
        print("✓ Found scikit-learn models:")
        for model_file in model_files:
            print(f"  - {model_file.name}")
    
    if autogluon_model.exists():
        print("✓ Found AutoGluon model")
    
    if not model_files and not autogluon_model.exists():
        print("✗ No trained models found")
        print("\nTrain a model with:")
        print("  python train_model_simple.py")
        return False
    
    return True


def show_next_steps(has_deps, has_data, has_models):
    """Show next steps based on current state."""
    print_header("Next Steps")
    
    if not has_deps:
        print("\n1. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Then run this script again: python quickstart.py")
        return
    
    if not has_data:
        print("\n1. Generate features (may take several minutes):")
        print("   python feature_engineering_pipeline.py")
        print("\n2. Then run this script again: python quickstart.py")
        return
    
    if not has_models:
        print("\n1. Train a model:")
        print("   python train_model_simple.py")
        print("\n   Options:")
        print("     --model random_forest")
        print("     --model gradient_boosting (recommended)")
        print("     --model logistic_regression")
        print("\n2. Make predictions:")
        print("   python predict_simple.py --sample 10")
        return
    
    # All set up!
    print("\n✓ You're all set up! Here's what you can do:\n")
    
    print("1. Make predictions on sample fights:")
    print("   python predict_simple.py --sample 10\n")
    
    print("2. Make predictions on all fights:")
    print("   python predict_simple.py\n")
    
    print("3. Save predictions to file:")
    print("   python predict_simple.py --output predictions.csv\n")
    
    print("4. Train a new model:")
    print("   python train_model_simple.py --model gradient_boosting\n")
    
    print("5. Read the full documentation:")
    print("   - README.md - Complete guide")
    print("   - PIPELINE_README.md - Pipeline details")
    print("   - DATA_LEAKAGE_ISSUE.md - Important notes on data quality\n")


def main():
    """Main quickstart function."""
    print_header("MMA-AI Prediction System - Quick Start")
    print("\nThis script will check your setup and guide you through getting started.")
    
    has_deps = check_dependencies()
    has_data = check_data_files()
    has_models = check_trained_models()
    
    show_next_steps(has_deps, has_data, has_models)
    
    print("\n" + "=" * 70)
    print("For more information, see README.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
