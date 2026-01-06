"""
MMA AI Feature Engineering Pipeline
Step 1: Data Pipeline to Aggregate Fight-Level Stats to Fighter-Level

This script implements the first phase of the feature engineering pipeline:
- Extract base stats from the SQLite database
- Aggregate fight-level statistics to fighter-level
- Calculate per-fight and cumulative statistics for each fighter
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class FighterStatsAggregator:
    """Aggregates fight-level statistics to fighter-level statistics."""
    
    def __init__(self, db_path='sqlite_scrapper.db'):
        """Initialize the aggregator with database connection."""
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        print(f"✓ Connected to database: {self.db_path}")
        
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
    
    def load_base_data(self):
        """Load base data from all relevant tables."""
        print("\n=== Loading Base Data ===")
        
        # Load fight stats (round-by-round data)
        query_fight_stats = """
        SELECT 
            EVENT,
            BOUT,
            ROUND,
            FIGHTER,
            CAST(KD AS REAL) as KD,
            "SIG.STR." as sig_str,
            "SIG.STR. %" as sig_str_pct,
            "TOTAL STR." as total_str,
            TD,
            "TD %" as td_pct,
            "SUB.ATT" as sub_att,
            "REV." as rev,
            CTRL as ctrl,
            HEAD as head,
            BODY as body,
            LEG as leg,
            DISTANCE as distance,
            CLINCH as clinch,
            GROUND as ground
        FROM ufc_fight_stats
        """
        self.fight_stats = pd.read_sql_query(query_fight_stats, self.conn)
        # Normalize EVENT and BOUT fields
        self.fight_stats['EVENT'] = self.fight_stats['EVENT'].str.strip()
        self.fight_stats['BOUT'] = self.fight_stats['BOUT'].str.strip().str.replace(r'\s+', ' ', regex=True)
        print(f"✓ Loaded {len(self.fight_stats)} fight stat records")
        
        # Load fight results (outcome data)
        query_results = """
        SELECT 
            EVENT,
            BOUT,
            OUTCOME,
            WEIGHTCLASS,
            METHOD,
            CAST(ROUND AS INTEGER) as ROUND,
            TIME,
            "TIME FORMAT" as time_format,
            sex,
            weightindex
        FROM ufc_fight_results
        """
        self.fight_results = pd.read_sql_query(query_results, self.conn)
        # Normalize EVENT and BOUT fields
        self.fight_results['EVENT'] = self.fight_results['EVENT'].str.strip()
        self.fight_results['BOUT'] = self.fight_results['BOUT'].str.strip().str.replace(r'\s+', ' ', regex=True)
        print(f"✓ Loaded {len(self.fight_results)} fight results")
        
        # Load event details (dates)
        query_events = """
        SELECT 
            EVENT,
            DATE,
            LOCATION
        FROM ufc_event_details
        """
        self.events = pd.read_sql_query(query_events, self.conn)
        # Normalize EVENT field
        self.events['EVENT'] = self.events['EVENT'].str.strip()
        print(f"✓ Loaded {len(self.events)} events")
        
        # Load fighter physical attributes
        query_fighter_details = """
        SELECT 
            FIGHTER,
            HEIGHT,
            WEIGHT,
            REACH,
            STANCE,
            DOB,
            sex,
            weightindex
        FROM ufc_fighter_tott
        """
        self.fighter_details = pd.read_sql_query(query_fighter_details, self.conn)
        print(f"✓ Loaded {len(self.fighter_details)} fighter profiles")
        
        # Load win/loss/ko data
        query_winlossko = """
        SELECT 
            DATE,
            EVENT,
            BOUT,
            fighter,
            win,
            loss,
            udec,
            mdec,
            sdec,
            subw,
            ko,
            fight_time_minutes
        FROM ufc_winlossko
        """
        self.winlossko = pd.read_sql_query(query_winlossko, self.conn)
        print(f"✓ Loaded {len(self.winlossko)} win/loss/ko records")
        
    def parse_stat_string(self, stat_str):
        """Parse stat strings like '17 of 26' into (landed, attempted)."""
        if pd.isna(stat_str) or stat_str == '---' or stat_str == '':
            return 0, 0
        try:
            if ' of ' in str(stat_str):
                parts = str(stat_str).split(' of ')
                landed = int(parts[0].strip())
                attempted = int(parts[1].strip())
                return landed, attempted
            else:
                # Single number (no attempts tracked)
                return float(stat_str), 0
        except:
            return 0, 0
    
    def parse_time_string(self, time_str):
        """Parse control time strings like '0:04' into seconds."""
        if pd.isna(time_str) or time_str == '---' or time_str == '':
            return 0
        try:
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            else:
                return 0
        except:
            return 0
    
    def parse_percentage(self, pct_str):
        """Parse percentage strings like '65%' into floats."""
        if pd.isna(pct_str) or pct_str == '---' or pct_str == '':
            return 0.0
        try:
            return float(str(pct_str).replace('%', ''))
        except:
            return 0.0
    
    def calculate_fight_duration(self, round_num, time_str, time_format):
        """Calculate total fight duration in minutes."""
        try:
            # Parse the round time
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                round_minutes = int(parts[0])
                round_seconds = int(parts[1])
            else:
                return 0.0
            
            # Calculate completed rounds time
            completed_rounds = int(round_num) - 1
            completed_time = completed_rounds * 5.0  # Each round is 5 minutes
            
            # Add partial round time
            partial_time = round_minutes + round_seconds / 60.0
            
            total_time = completed_time + partial_time
            return total_time
        except:
            return 0.0
    
    def process_fight_stats(self):
        """Process and aggregate fight statistics."""
        print("\n=== Processing Fight Statistics ===")
        
        # Create a copy to work with
        df = self.fight_stats.copy()
        
        # Parse all stat strings
        print("Parsing stat strings...")
        df[['sig_str_landed', 'sig_str_att']] = df['sig_str'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['total_str_landed', 'total_str_att']] = df['total_str'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['td_landed', 'td_att']] = df['TD'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['head_landed', 'head_att']] = df['head'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['body_landed', 'body_att']] = df['body'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['leg_landed', 'leg_att']] = df['leg'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['distance_landed', 'distance_att']] = df['distance'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['clinch_landed', 'clinch_att']] = df['clinch'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        df[['ground_landed', 'ground_att']] = df['ground'].apply(
            lambda x: pd.Series(self.parse_stat_string(x))
        )
        
        # Parse control time
        df['ctrl_seconds'] = df['ctrl'].apply(self.parse_time_string)
        
        # Parse percentages
        df['sig_str_pct_val'] = df['sig_str_pct'].apply(self.parse_percentage)
        df['td_pct_val'] = df['td_pct'].apply(self.parse_percentage)
        
        # Parse submission attempts and reversals (single numbers)
        df['sub_att_val'] = df['sub_att'].apply(lambda x: float(x) if str(x).replace('.','').isdigit() else 0)
        df['rev_val'] = df['rev'].apply(lambda x: float(x) if str(x).replace('.','').isdigit() else 0)
        
        print(f"✓ Parsed statistics for {len(df)} records")
        
        # Aggregate by fight (sum across all rounds)
        print("Aggregating round-level data to fight-level...")
        fight_agg = df.groupby(['EVENT', 'BOUT', 'FIGHTER']).agg({
            'KD': 'sum',
            'sig_str_landed': 'sum',
            'sig_str_att': 'sum',
            'total_str_landed': 'sum',
            'total_str_att': 'sum',
            'td_landed': 'sum',
            'td_att': 'sum',
            'sub_att_val': 'sum',
            'rev_val': 'sum',
            'ctrl_seconds': 'sum',
            'head_landed': 'sum',
            'head_att': 'sum',
            'body_landed': 'sum',
            'body_att': 'sum',
            'leg_landed': 'sum',
            'leg_att': 'sum',
            'distance_landed': 'sum',
            'distance_att': 'sum',
            'clinch_landed': 'sum',
            'clinch_att': 'sum',
            'ground_landed': 'sum',
            'ground_att': 'sum'
        }).reset_index()
        
        print(f"✓ Aggregated to {len(fight_agg)} fight-level records")
        
        # Add fight results data
        print("Merging with fight results...")
        fight_agg = fight_agg.merge(
            self.fight_results[['EVENT', 'BOUT', 'OUTCOME', 'WEIGHTCLASS', 'METHOD', 'ROUND', 'TIME', 'time_format']],
            on=['EVENT', 'BOUT'],
            how='left'
        )
        
        # Add event dates
        print("Merging with event dates...")
        fight_agg = fight_agg.merge(
            self.events[['EVENT', 'DATE']],
            on='EVENT',
            how='left'
        )
        
        # Calculate fight duration
        fight_agg['fight_duration_minutes'] = fight_agg.apply(
            lambda row: self.calculate_fight_duration(row['ROUND'], row['TIME'], row['time_format']),
            axis=1
        )
        
        # Determine winner/loser
        # Parse OUTCOME field (e.g., "L/W" means first fighter lost, second won)
        fight_agg['fighters_in_bout'] = fight_agg.groupby(['EVENT', 'BOUT'])['FIGHTER'].transform(lambda x: '|||'.join(sorted(x)))
        
        print(f"✓ Processed {len(fight_agg)} complete fight records")
        
        self.processed_fights = fight_agg
        return fight_agg
    
    def create_fighter_aggregates(self):
        """Create fighter-level aggregated statistics."""
        print("\n=== Creating Fighter-Level Aggregates ===")
        
        if not hasattr(self, 'processed_fights'):
            raise ValueError("Must run process_fight_stats() first")
        
        df = self.processed_fights.copy()
        
        # Sort by date to maintain chronological order
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.sort_values(['FIGHTER', 'DATE'])
        
        # Add fight number for each fighter
        df['fight_number'] = df.groupby('FIGHTER').cumcount() + 1
        df['total_fights'] = df.groupby('FIGHTER')['FIGHTER'].transform('count')
        
        print(f"✓ Processing {df['FIGHTER'].nunique()} unique fighters")
        
        # Calculate per-minute rates
        print("Calculating per-minute rates...")
        df['sig_str_per_min'] = df['sig_str_landed'] / df['fight_duration_minutes'].replace(0, np.nan)
        df['total_str_per_min'] = df['total_str_landed'] / df['fight_duration_minutes'].replace(0, np.nan)
        df['td_per_min'] = df['td_landed'] / df['fight_duration_minutes'].replace(0, np.nan)
        df['sub_att_per_min'] = df['sub_att_val'] / df['fight_duration_minutes'].replace(0, np.nan)
        df['rev_per_min'] = df['rev_val'] / df['fight_duration_minutes'].replace(0, np.nan)
        df['ctrl_per_min'] = df['ctrl_seconds'] / 60.0 / df['fight_duration_minutes'].replace(0, np.nan)
        df['kd_per_min'] = df['KD'] / df['fight_duration_minutes'].replace(0, np.nan)
        
        # Calculate accuracy rates
        print("Calculating accuracy rates...")
        df['sig_str_acc'] = df['sig_str_landed'] / df['sig_str_att'].replace(0, np.nan)
        df['total_str_acc'] = df['total_str_landed'] / df['total_str_att'].replace(0, np.nan)
        df['td_acc'] = df['td_landed'] / df['td_att'].replace(0, np.nan)
        df['head_acc'] = df['head_landed'] / df['head_att'].replace(0, np.nan)
        df['body_acc'] = df['body_landed'] / df['body_att'].replace(0, np.nan)
        df['leg_acc'] = df['leg_landed'] / df['leg_att'].replace(0, np.nan)
        df['distance_acc'] = df['distance_landed'] / df['distance_att'].replace(0, np.nan)
        df['clinch_acc'] = df['clinch_landed'] / df['clinch_att'].replace(0, np.nan)
        df['ground_acc'] = df['ground_landed'] / df['ground_att'].replace(0, np.nan)
        
        # Fill NaN values with 0 for rates and accuracies
        rate_cols = [col for col in df.columns if '_per_min' in col or '_acc' in col]
        df[rate_cols] = df[rate_cols].fillna(0)
        
        print(f"✓ Calculated {len(rate_cols)} derived statistics")
        
        # Merge with fighter physical attributes
        print("Merging with fighter physical attributes...")
        df = df.merge(
            self.fighter_details[['FIGHTER', 'HEIGHT', 'WEIGHT', 'REACH', 'STANCE', 'DOB']],
            on='FIGHTER',
            how='left'
        )
        
        # Calculate age at time of fight
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['age_at_fight'] = (df['DATE'] - df['DOB']).dt.days / 365.25
        
        # Calculate days since last fight
        df['days_since_last_fight'] = df.groupby('FIGHTER')['DATE'].diff().dt.days
        
        print(f"✓ Final dataset shape: {df.shape}")
        
        self.fighter_level_stats = df
        return df
    
    def save_aggregated_data(self, output_path='fighter_aggregated_stats.csv'):
        """Save the aggregated fighter-level statistics."""
        if not hasattr(self, 'fighter_level_stats'):
            raise ValueError("Must run create_fighter_aggregates() first")
        
        self.fighter_level_stats.to_csv(output_path, index=False)
        print(f"\n✓ Saved aggregated data to: {output_path}")
        print(f"  - Total records: {len(self.fighter_level_stats)}")
        print(f"  - Unique fighters: {self.fighter_level_stats['FIGHTER'].nunique()}")
        print(f"  - Columns: {len(self.fighter_level_stats.columns)}")
        
    def print_summary_statistics(self):
        """Print summary statistics of the aggregated data."""
        if not hasattr(self, 'fighter_level_stats'):
            raise ValueError("Must run create_fighter_aggregates() first")
        
        df = self.fighter_level_stats
        
        print("\n=== Summary Statistics ===")
        print(f"\nDataset Overview:")
        print(f"  Total fight records: {len(df)}")
        print(f"  Unique fighters: {df['FIGHTER'].nunique()}")
        print(f"  Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        print(f"  Weight classes: {df['WEIGHTCLASS'].nunique()}")
        
        print(f"\nBase Statistics (22 categories):")
        base_stats = [
            'sig_str_landed', 'total_str_landed', 'head_landed', 'body_landed', 'leg_landed',
            'distance_landed', 'clinch_landed', 'ground_landed', 'td_landed', 'td_acc',
            'sub_att_val', 'rev_val', 'ctrl_seconds', 'KD', 'HEIGHT', 'WEIGHT', 'REACH',
            'age_at_fight', 'fight_duration_minutes', 'days_since_last_fight', 'fight_number'
        ]
        print(f"  Core stats tracked: {len(base_stats)}")
        
        print(f"\nDerived Features:")
        per_min_cols = [col for col in df.columns if '_per_min' in col]
        acc_cols = [col for col in df.columns if '_acc' in col]
        print(f"  Per-minute rates: {len(per_min_cols)}")
        print(f"  Accuracy rates: {len(acc_cols)}")
        print(f"  Total derived features: {len(per_min_cols) + len(acc_cols)}")
        
        print(f"\n✓ Step 1 Complete: Data aggregated to fighter-level")


def main():
    """Main execution function."""
    print("=" * 60)
    print("MMA AI Feature Engineering Pipeline - Step 1")
    print("Aggregating Fight-Level Stats to Fighter-Level")
    print("=" * 60)
    
    # Initialize aggregator
    aggregator = FighterStatsAggregator()
    
    try:
        # Step 1: Connect and load data
        aggregator.connect()
        aggregator.load_base_data()
        
        # Step 2: Process fight statistics
        aggregator.process_fight_stats()
        
        # Step 3: Create fighter-level aggregates
        aggregator.create_fighter_aggregates()
        
        # Step 4: Save results
        aggregator.save_aggregated_data()
        
        # Step 5: Print summary
        aggregator.print_summary_statistics()
        
        print("\n" + "=" * 60)
        print("✓ Step 1 completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        aggregator.disconnect()


if __name__ == "__main__":
    main()
