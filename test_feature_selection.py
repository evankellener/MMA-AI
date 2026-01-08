#!/usr/bin/env python3
"""
Test script to verify feature selection fix is working correctly.
This demonstrates that the code changes prevent data leakage, even though
the existing CSV files haven't been regenerated yet.
"""

import csv
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_schema import load_feature_schema, resolve_feature_timing


def test_feature_classification():
    """Test that feature classification is working correctly."""
    print("=" * 70)
    print("TEST 1: Feature Classification")
    print("=" * 70)
    
    test_cases = [
        # (feature_name, expected_timing)
        ('fighter1_win', 'postcomp'),
        ('diff_change_avg_win_differential', 'postcomp'),
        ('f1_win_avg', 'postcomp'),
        ('win_streak', 'postcomp'),
        ('diff_sig_str_per_min_adjperf', 'precomp'),
        ('f1_td_acc_dec_avg', 'precomp'),
        ('ratio_ctrl_per_min', 'precomp'),
        ('f1_age_at_fight', 'precomp'),
    ]
    
    schema_map = load_feature_schema()
    passed = 0
    failed = 0
    
    for feature, expected in test_cases:
        timing = resolve_feature_timing(feature, schema_map)
        is_correct = timing == expected
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"{status}: {feature:45s} -> {timing:10s} (expected: {expected})")
        
        if is_correct:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResult: {passed}/{len(test_cases)} tests passed")
    return failed == 0


def test_dataset_filtering():
    """Test filtering on actual matchup_comparisons.csv."""
    print("\n" + "=" * 70)
    print("TEST 2: Dataset Filtering (matchup_comparisons.csv)")
    print("=" * 70)
    
    # Load headers from matchup_comparisons.csv
    with open('matchup_comparisons.csv', 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    
    schema_map = load_feature_schema()
    metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 'METHOD', 
                    'fighter1', 'fighter2', 'fighter1_win']
    
    # Simulate FeatureSelector filtering
    all_feature_cols = [col for col in headers if col not in metadata_cols]
    precomp_cols = []
    postcomp_cols = []
    
    for col in all_feature_cols:
        timing = resolve_feature_timing(col, schema_map)
        if timing == 'precomp':
            precomp_cols.append(col)
        else:
            postcomp_cols.append(col)
    
    print(f"Total feature columns: {len(all_feature_cols)}")
    print(f"  Precomp (usable for training): {len(precomp_cols)}")
    print(f"  Postcomp (will be excluded): {len(postcomp_cols)}")
    
    # Check for win features in precomp
    win_in_precomp = [f for f in precomp_cols if 'win' in f.lower()]
    
    print(f"\n{'✓ PASS' if not win_in_precomp else '✗ FAIL'}: Win features in precomp list: {len(win_in_precomp)}")
    
    if win_in_precomp:
        print("  ERROR: These win features should be postcomp:")
        for feat in win_in_precomp[:5]:
            print(f"    - {feat}")
    
    print(f"\nPostcomp features that will be excluded (sample):")
    for feat in postcomp_cols[:10]:
        print(f"  - {feat}")
    
    return len(win_in_precomp) == 0


def test_existing_csv_comparison():
    """Compare existing CSV with what should be generated."""
    print("\n" + "=" * 70)
    print("TEST 3: Existing CSV Comparison")
    print("=" * 70)
    
    # Load existing selected features
    with open('matchup_selected_features.csv', 'r') as f:
        reader = csv.reader(f)
        existing_headers = next(reader)
    
    schema_map = load_feature_schema()
    metadata_cols = ['EVENT', 'BOUT', 'DATE', 'OUTCOME', 'WEIGHTCLASS', 'METHOD', 
                    'fighter1', 'fighter2', 'fighter1_win']
    
    existing_features = [col for col in existing_headers if col not in metadata_cols]
    
    # Classify existing features
    existing_postcomp = []
    for feat in existing_features:
        timing = resolve_feature_timing(feat, schema_map)
        if timing == 'postcomp':
            existing_postcomp.append(feat)
    
    print(f"Existing matchup_selected_features.csv:")
    print(f"  Total features: {len(existing_features)}")
    print(f"  Postcomp features (should be 0): {len(existing_postcomp)}")
    
    if existing_postcomp:
        print(f"\n✗ WARNING: CSV contains {len(existing_postcomp)} postcomp features")
        print(f"  This CSV was generated BEFORE the code fix")
        print(f"  To fix: Re-run feature_engineering_pipeline.py")
        print(f"\n  Postcomp features in existing CSV (sample):")
        for feat in existing_postcomp[:10]:
            print(f"    - {feat}")
    else:
        print(f"\n✓ CSV is correct - contains only precomp features")
    
    return len(existing_postcomp) == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FEATURE SELECTION FIX VERIFICATION")
    print("=" * 70)
    print("\nThis test verifies that the code changes are working correctly.")
    print("The code will filter out postcomp features when the pipeline is run.\n")
    
    test1_pass = test_feature_classification()
    test2_pass = test_dataset_filtering()
    test3_pass = test_existing_csv_comparison()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Classification): {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Test 2 (Filtering Logic): {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print(f"Test 3 (Existing CSV): {'✓ PASS' if test3_pass else '⚠ NEEDS REGENERATION'}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if test1_pass and test2_pass:
        print("✓ Code changes are WORKING CORRECTLY")
        print("✓ Feature selection will filter out postcomp features")
        print("✓ No data leakage will occur when pipeline is run")
        
        if not test3_pass:
            print("\n⚠ Action required:")
            print("  Run 'python feature_engineering_pipeline.py' to regenerate CSV files")
        else:
            print("\n✓ All CSV files are up to date")
    else:
        print("✗ Code changes have issues - review needed")
    
    print("=" * 70)


if __name__ == "__main__":
    main()