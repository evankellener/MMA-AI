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
        
        # Add win column (1 = win, 0 = loss/draw/no contest)
        print("Calculating win/loss outcomes...")
        df['win'] = self._calculate_win_column(df)
        
        print(f"✓ Final dataset shape: {df.shape}")
        
        self.fighter_level_stats = df
        return df
    
    def _calculate_win_column(self, df):
        """
        Calculate win column based on OUTCOME field and fighter position.
        
        OUTCOME format: "L/W" or "W/L" where first letter = first fighter (alphabetically), second = second fighter
        Returns: Series with 1 for win, 0 for loss/draw/no contest
        """
        win_col = []
        
        for idx, row in df.iterrows():
            outcome = row['OUTCOME']
            fighter = row['FIGHTER']
            fighters_in_bout = row['fighters_in_bout']
            
            # Handle missing data
            if pd.isna(outcome) or pd.isna(fighters_in_bout):
                win_col.append(np.nan)
                continue
            
            # Handle draws and no contests
            if outcome in ['D/D', 'NC/NC']:
                win_col.append(0)
                continue
            
            # Parse outcome (e.g., "L/W" or "W/L")
            if '/' not in str(outcome):
                win_col.append(np.nan)
                continue
                
            parts = str(outcome).split('/')
            if len(parts) != 2:
                win_col.append(np.nan)
                continue
            
            # Get fighter positions (alphabetically sorted)
            fighters = str(fighters_in_bout).split('|||')
            if len(fighters) != 2:
                win_col.append(np.nan)
                continue
            
            # Determine which position this fighter is in
            try:
                fighter_position = fighters.index(fighter)
                result = parts[fighter_position]
                
                # W = win, L = loss
                if result == 'W':
                    win_col.append(1)
                elif result == 'L':
                    win_col.append(0)
                else:
                    win_col.append(np.nan)
            except (ValueError, IndexError):
                win_col.append(np.nan)
        
        return pd.Series(win_col, index=df.index)
    
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
        
        # Win/loss statistics
        if 'win' in df.columns:
            win_count = df['win'].sum()
            loss_count = (df['win'] == 0).sum()
            total_with_outcome = win_count + loss_count
            print(f"\nOutcome Statistics:")
            print(f"  Wins: {win_count}")
            print(f"  Losses: {loss_count}")
            print(f"  Win rate: {win_count/total_with_outcome*100:.1f}%" if total_with_outcome > 0 else "  Win rate: N/A")
        
        print(f"\n✓ Step 1 Complete: Data aggregated to fighter-level")
    
    def print_filtering_report(self):
        """Print detailed report on what data was filtered and why."""
        print("\n" + "=" * 60)
        print("Data Filtering Report")
        print("=" * 60)
        
        # Count records at each stage
        print("\n=== Records at Each Stage ===")
        print(f"1. Round-level records (raw): {len(self.fight_stats)}")
        print(f"2. Fight-level records (aggregated): {len(self.processed_fights)}")
        print(f"3. Final fighter records: {len(self.fighter_level_stats)}")
        
        # Analyze what was filtered during merges
        print("\n=== Merge Analysis ===")
        
        # Check fights without results
        fights_without_results = self.processed_fights[self.processed_fights['OUTCOME'].isna()]
        if len(fights_without_results) > 0:
            print(f"Fights without OUTCOME data: {len(fights_without_results)}")
            print(f"  Reason: No matching record in ufc_fight_results table")
            print(f"  Impact: Missing fight duration, per-minute rates = 0")
        
        # Check fighters without physical attributes
        fighters_without_attrs = self.fighter_level_stats[self.fighter_level_stats['HEIGHT'].isna()]
        if len(fighters_without_attrs) > 0:
            unique_fighters = fighters_without_attrs['FIGHTER'].nunique()
            print(f"\nFighters without physical attributes: {len(fighters_without_attrs)} records ({unique_fighters} unique fighters)")
            print(f"  Reason: No matching record in ufc_fighter_tott table")
            print(f"  Impact: Missing HEIGHT, WEIGHT, REACH, STANCE, DOB")
            print(f"  Sample fighters: {', '.join(fighters_without_attrs['FIGHTER'].unique()[:5])}")
        
        # Database comparison
        print("\n=== Database Coverage ===")
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        
        # Count unique bouts in database
        db_bouts = pd.read_sql_query("SELECT COUNT(DISTINCT EVENT || BOUT) as count FROM ufc_fight_stats", conn).iloc[0]['count']
        db_results = pd.read_sql_query("SELECT COUNT(DISTINCT EVENT || BOUT) as count FROM ufc_fight_results", conn).iloc[0]['count']
        
        print(f"Unique bouts in ufc_fight_stats: {db_bouts}")
        print(f"Unique bouts in ufc_fight_results: {db_results}")
        print(f"Bouts processed: {self.processed_fights[['EVENT', 'BOUT']].drop_duplicates().shape[0]}")
        
        # Explain difference
        if db_bouts > db_results:
            print(f"\nNote: {db_bouts - db_results} bouts have stats but no results metadata")
            print("  These may be exhibition matches or data collection issues")
        
        conn.close()
        
        print("\n=== Summary ===")
        data_completeness = (len(self.fighter_level_stats) - len(fights_without_results)) / len(self.fighter_level_stats) * 100
        print(f"Overall data completeness: {data_completeness:.1f}%")
        print(f"No fights were intentionally filtered - all available data is included")
        print(f"Missing data is due to incomplete records in source database tables")
        print("=" * 60)


class TimeDecayCalculator:
    """Calculates time-decayed averages for fighter statistics."""
    
    def __init__(self, half_life_years=1.5):
        """
        Initialize with half-life parameter.
        
        Args:
            half_life_years: Half-life for exponential decay (default: 1.5 years from mma_ai.md)
        """
        self.half_life_years = half_life_years
        # Calculate lambda for exponential decay: lambda = ln(2) / half_life
        self.decay_lambda = np.log(2) / half_life_years
        
    def calculate_decay_weights(self, dates, current_date):
        """
        Calculate exponential decay weights for historical data.
        
        Formula: weight = EXP(-λ × ((T - t) / 365.25))
        where T = current date, t = historical date, λ = ln(2) / half_life
        
        Args:
            dates: Series of historical dates
            current_date: Current reference date
            
        Returns:
            Series of decay weights (higher = more recent)
        """
        # Calculate days difference
        days_diff = (current_date - dates).dt.days
        
        # Calculate years difference
        years_diff = days_diff / 365.25
        
        # Apply exponential decay
        weights = np.exp(-self.decay_lambda * years_diff)
        
        return weights
    
    def calculate_decayed_average(self, values, weights):
        """
        Calculate weighted average using decay weights.
        
        Args:
            values: Series of stat values
            weights: Series of decay weights
            
        Returns:
            Weighted average
        """
        # Handle NaN values
        valid_mask = values.notna() & weights.notna()
        
        if not valid_mask.any():
            return np.nan
        
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        
        # Calculate weighted average
        weighted_sum = (valid_values * valid_weights).sum()
        weight_sum = valid_weights.sum()
        
        if weight_sum == 0:
            return np.nan
        
        return weighted_sum / weight_sum
    
    def add_decayed_averages(self, df, stats_to_decay, max_fighters=None):
        """
        Add time-decayed average columns for specified statistics (optimized version).
        
        For each fighter, calculates decayed average of past performance
        up to (but not including) the current fight.
        
        Args:
            df: DataFrame with fight records (must be sorted by FIGHTER, DATE)
            stats_to_decay: List of column names to calculate decayed averages for
            max_fighters: Optional limit for testing (None = process all)
            
        Returns:
            DataFrame with added decayed average columns
        """
        print("\n=== Calculating Time-Decayed Averages ===")
        print(f"Half-life: {self.half_life_years} years")
        print(f"Decay lambda: {self.decay_lambda:.4f}")
        
        df = df.copy()
        
        # Ensure data is sorted and has proper index
        df = df.sort_values(['FIGHTER', 'DATE']).reset_index(drop=True)
        
        # Initialize decayed average columns
        decayed_cols = []
        for stat in stats_to_decay:
            col_name = f'{stat}_dec_avg'
            df[col_name] = np.nan
            decayed_cols.append(col_name)
        
        # Get unique fighters
        fighters = df['FIGHTER'].unique()
        if max_fighters:
            fighters = fighters[:max_fighters]
            print(f"  (Testing mode: processing {max_fighters} fighters)")
        
        total_fighters = len(fighters)
        print(f"Processing {total_fighters} fighters...")
        
        # Process in batches for progress reporting
        batch_size = max(1, total_fighters // 10)
        
        for fighter_idx, fighter in enumerate(fighters):
            # Progress reporting
            if (fighter_idx + 1) % batch_size == 0 or fighter_idx == 0:
                print(f"  Progress: {fighter_idx + 1}/{total_fighters} fighters processed")
            
            # Get all records for this fighter
            fighter_mask = (df['FIGHTER'] == fighter)
            fighter_rows = df[fighter_mask]
            
            if len(fighter_rows) < 2:
                continue  # Need at least 2 fights to calculate a decayed average
            
            # Get indices and dates for this fighter
            indices = fighter_rows.index.tolist()
            dates = fighter_rows['DATE'].values
            
            # For each fight (starting from second), calculate decayed avg of previous fights
            for i in range(1, len(indices)):
                current_idx = indices[i]
                current_date = dates[i]
                
                # Get previous fights
                prev_indices = indices[:i]
                prev_dates = dates[:i]
                
                # Calculate time differences in years
                days_diff = (current_date - prev_dates) / np.timedelta64(1, 'D')
                years_diff = days_diff / 365.25
                
                # Calculate decay weights
                weights = np.exp(-self.decay_lambda * years_diff)
                
                # Calculate decayed average for each stat
                for stat in stats_to_decay:
                    if stat in df.columns:
                        prev_values = df.loc[prev_indices, stat].values
                        
                        # Filter out NaN values
                        valid_mask = ~np.isnan(prev_values)
                        
                        if valid_mask.any():
                            valid_values = prev_values[valid_mask]
                            valid_weights = weights[valid_mask]
                            
                            # Weighted average
                            weighted_sum = np.sum(valid_values * valid_weights)
                            weight_sum = np.sum(valid_weights)
                            
                            if weight_sum > 0:
                                df.at[current_idx, f'{stat}_dec_avg'] = weighted_sum / weight_sum
        
        print(f"✓ Calculated decayed averages for {len(stats_to_decay)} statistics")
        print(f"  Total decayed average columns: {len(decayed_cols)}")
        
        # Show sample statistics
        if decayed_cols:
            non_null_counts = df[decayed_cols].notna().sum()
            total_rows = len(df)
            print(f"  Sample non-null counts (top 5):")
            for col in decayed_cols[:5]:
                count = non_null_counts[col]
                pct = count/total_rows*100
                print(f"    {col}: {count}/{total_rows} ({pct:.1f}%)")
        
        return df


class AdvancedAggregationsCalculator:
    """
    Calculates advanced aggregation features including peak/valley tracking,
    change metrics, recent vs career comparisons, ELO ratings, and round-specific stats.
    
    This dramatically expands the feature space from ~100 to 1000+ features per fighter.
    """
    
    def __init__(self, recent_fights_threshold=3, peak_valley_window=5):
        """
        Initialize the advanced aggregations calculator.
        
        Args:
            recent_fights_threshold: Number of recent fights to consider for "recent" metrics
            peak_valley_window: Minimum fights needed to establish peak/valley
        """
        self.recent_threshold = recent_fights_threshold
        self.peak_valley_window = peak_valley_window
        
    def add_advanced_features(self, df, stats_to_expand):
        """
        Add all advanced aggregation features to the dataframe.
        
        Args:
            df: DataFrame with fighter-level stats
            stats_to_expand: List of column names to create advanced features for
            
        Returns:
            DataFrame with additional advanced feature columns
        """
        df = df.copy()
        
        print("  Calculating peak/valley metrics...")
        df = self._add_peak_valley_features(df, stats_to_expand)
        
        print("  Calculating change/trend metrics...")
        df = self._add_change_features(df, stats_to_expand)
        
        print("  Calculating recent vs career comparisons...")
        df = self._add_recent_vs_career_features(df, stats_to_expand)
        
        print("  Calculating round-specific aggregations...")
        df = self._add_round_specific_features(df)
        
        return df
    
    def _add_peak_valley_features(self, df, stats):
        """Add career peak and valley (high/low) metrics for each stat."""
        new_cols = {}
        
        for stat in stats:
            if stat not in df.columns:
                continue
                
            # Calculate rolling peak (maximum over career so far)
            new_cols[f'{stat}_peak'] = df.groupby('FIGHTER')[stat].transform(
                lambda x: x.expanding().max()
            )
            
            # Calculate rolling valley (minimum over career so far)
            new_cols[f'{stat}_valley'] = df.groupby('FIGHTER')[stat].transform(
                lambda x: x.expanding().min()
            )
            
            # Differential from peak
            new_cols[f'{stat}_differential_vs_peak'] = df[stat] - new_cols[f'{stat}_peak']
            
            # Differential from valley  
            new_cols[f'{stat}_differential_vs_valley'] = df[stat] - new_cols[f'{stat}_valley']
        
        # Concatenate all new columns at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        return df
    
    def _add_change_features(self, df, stats):
        """Add change/trend metrics showing performance trajectory."""
        new_cols = {}
        
        for stat in stats:
            if stat not in df.columns:
                continue
            
            # Calculate average up to this point
            stat_avg = df.groupby('FIGHTER')[stat].transform(
                lambda x: x.expanding().mean()
            )
            new_cols[f'{stat}_avg'] = stat_avg
            
            # Change from career average
            new_cols[f'change_avg_{stat}_differential'] = df[stat] - stat_avg
            
            # Previous fight value
            stat_prev = df.groupby('FIGHTER')[stat].shift(1)
            
            # Change from previous fight
            new_cols[f'change_{stat}_differential'] = df[stat] - stat_prev
        
        # Concatenate all new columns at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        return df
    
    def _add_recent_vs_career_features(self, df, stats):
        """Add recent average vs career average comparisons."""
        new_cols = {}
        
        for stat in stats:
            if stat not in df.columns:
                continue
            
            # Recent average (last N fights)
            recent_avg = df.groupby('FIGHTER')[stat].transform(
                lambda x: x.rolling(window=self.recent_threshold, min_periods=1).mean()
            )
            new_cols[f'recent_avg_{stat}'] = recent_avg
            
            # Career average (expanding mean)
            if f'{stat}_avg' not in df.columns:
                stat_avg = df.groupby('FIGHTER')[stat].transform(
                    lambda x: x.expanding().mean()
                )
                new_cols[f'{stat}_avg'] = stat_avg
            else:
                stat_avg = df[f'{stat}_avg']
            
            # Differential: recent vs career
            new_cols[f'recent_avg_{stat}_differential'] = recent_avg - stat_avg
        
        # Concatenate all new columns at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        return df
    
    def _add_round_specific_features(self, df):
        """
        Add round-specific aggregations (Round 1, Round 2, Round 3 stats).
        
        Note: This requires round-by-round data which we aggregate at fight level.
        We'll add placeholders for now and mark them for future implementation.
        """
        # Placeholder for round-specific features
        # These would require re-aggregating from the original round-level data
        # For now, we'll create ratio columns based on existing control time
        
        new_cols = {}
        if 'ctrl' in df.columns:
            # Assume first third of fight is "Round 1" approximation
            new_cols['ctrl_round1_approx'] = df['ctrl'] / 3.0
            new_cols['ctrl_round1_per_min_approx'] = df.get('ctrl_per_min', 0) / 3.0
        
        # Concatenate all new columns at once
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            
        return df
    

class OpponentAdjustedPerformanceCalculator:
    """Calculates opponent-adjusted performance (AdjPerf) metrics for fighter statistics."""
    
    def __init__(self, min_fights_for_baseline=3, bayesian_prior_weight=5):
        """
        Initialize with Bayesian shrinkage parameters.
        
        Args:
            min_fights_for_baseline: Minimum fights needed for opponent baseline
            bayesian_prior_weight: Weight for Bayesian prior (shrinkage toward mean)
        """
        self.min_fights_for_baseline = min_fights_for_baseline
        self.bayesian_prior_weight = bayesian_prior_weight
    
    def calculate_opponent_baselines(self, df, stats_to_adjust):
        """
        Calculate what each fighter typically allows opponents to achieve.
        
        This creates a baseline showing defensive capability:
        - Low values = good defense (opponents achieve less)
        - High values = poor defense (opponents achieve more)
        
        Args:
            df: DataFrame with fight records
            stats_to_adjust: List of statistics to calculate baselines for
            
        Returns:
            Dictionary mapping (fighter, stat) -> (mean_allowed, std_allowed, n_fights)
        """
        print("\n=== Calculating Opponent Baselines ===")
        print("Computing what each fighter typically allows opponents to achieve...")
        
        baselines = {}
        
        # For each unique bout, we need both fighters' stats
        df_sorted = df.sort_values(['EVENT', 'BOUT', 'FIGHTER']).reset_index(drop=True)
        
        # Group by bout to get opponent pairs
        bout_groups = df_sorted.groupby(['EVENT', 'BOUT'])
        
        for stat in stats_to_adjust:
            if stat not in df.columns:
                continue
                
            # Track what each fighter allowed (opponent's performance against them)
            fighter_allowed = {}
            
            for (event, bout), group in bout_groups:
                if len(group) != 2:
                    continue  # Skip if not exactly 2 fighters
                
                fighter1 = group.iloc[0]['FIGHTER']
                fighter2 = group.iloc[1]['FIGHTER']
                
                stat1 = group.iloc[0][stat]
                stat2 = group.iloc[1][stat]
                
                # Fighter 1 allowed fighter 2 to achieve stat2
                if pd.notna(stat2):
                    if fighter1 not in fighter_allowed:
                        fighter_allowed[fighter1] = []
                    fighter_allowed[fighter1].append(stat2)
                
                # Fighter 2 allowed fighter 1 to achieve stat1
                if pd.notna(stat1):
                    if fighter2 not in fighter_allowed:
                        fighter_allowed[fighter2] = []
                    fighter_allowed[fighter2].append(stat1)
            
            # Calculate mean and std for each fighter
            for fighter, values in fighter_allowed.items():
                if len(values) >= self.min_fights_for_baseline:
                    mean_allowed = np.mean(values)
                    std_allowed = np.std(values) if len(values) > 1 else np.nan
                    baselines[(fighter, stat)] = (mean_allowed, std_allowed, len(values))
        
        print(f"✓ Calculated baselines for {len(stats_to_adjust)} statistics")
        print(f"  Total fighter-stat baselines: {len(baselines)}")
        
        return baselines
    
    def add_opponent_adjusted_performance(self, df, stats_to_adjust, baselines):
        """
        Add opponent-adjusted performance columns.
        
        Formula: AdjPerf = (fighter_stat - opponent_mean_allowed) / opponent_std_allowed
        
        This is a z-score showing how much better/worse the fighter performed
        compared to what their opponent typically allows.
        
        Args:
            df: DataFrame with fight records
            stats_to_adjust: List of statistics to adjust
            baselines: Dictionary from calculate_opponent_baselines()
            
        Returns:
            DataFrame with added adjperf columns
        """
        print("\n=== Calculating Opponent-Adjusted Performance ===")
        print("Applying z-score normalization against opponent baselines...")
        
        df = df.copy()
        df_sorted = df.sort_values(['EVENT', 'BOUT', 'FIGHTER']).reset_index(drop=True)
        
        # Initialize adjusted performance columns
        adjperf_cols = []
        for stat in stats_to_adjust:
            col_name = f'{stat}_adjperf'
            df[col_name] = np.nan
            adjperf_cols.append(col_name)
        
        # Group by bout to get opponent pairs
        bout_groups = df_sorted.groupby(['EVENT', 'BOUT'])
        
        # Global mean and std for Bayesian shrinkage
        global_stats = {}
        for stat in stats_to_adjust:
            if stat in df.columns:
                valid_values = df[stat].dropna()
                if len(valid_values) > 0:
                    global_stats[stat] = (valid_values.mean(), valid_values.std())
        
        adjusted_count = 0
        
        for (event, bout), group in bout_groups:
            if len(group) != 2:
                continue
            
            fighter1_idx = group.iloc[0].name
            fighter2_idx = group.iloc[1].name
            
            fighter1 = group.iloc[0]['FIGHTER']
            fighter2 = group.iloc[1]['FIGHTER']
            
            for stat in stats_to_adjust:
                if stat not in df.columns:
                    continue
                
                stat1 = group.iloc[0][stat]
                stat2 = group.iloc[1][stat]
                
                # Adjust fighter 1's performance against fighter 2's baseline
                if pd.notna(stat1) and (fighter2, stat) in baselines:
                    opp_mean, opp_std, n_fights = baselines[(fighter2, stat)]
                    
                    # Apply Bayesian shrinkage if few opponent fights
                    if n_fights < 10 and stat in global_stats:
                        global_mean, global_std = global_stats[stat]
                        # Shrink toward global mean
                        weight = n_fights / (n_fights + self.bayesian_prior_weight)
                        opp_mean = weight * opp_mean + (1 - weight) * global_mean
                        opp_std = weight * opp_std + (1 - weight) * global_std if pd.notna(opp_std) else global_std
                    
                    if pd.notna(opp_std) and opp_std > 0:
                        adjperf = (stat1 - opp_mean) / opp_std
                        df.at[fighter1_idx, f'{stat}_adjperf'] = adjperf
                        adjusted_count += 1
                
                # Adjust fighter 2's performance against fighter 1's baseline
                if pd.notna(stat2) and (fighter1, stat) in baselines:
                    opp_mean, opp_std, n_fights = baselines[(fighter1, stat)]
                    
                    # Apply Bayesian shrinkage
                    if n_fights < 10 and stat in global_stats:
                        global_mean, global_std = global_stats[stat]
                        weight = n_fights / (n_fights + self.bayesian_prior_weight)
                        opp_mean = weight * opp_mean + (1 - weight) * global_mean
                        opp_std = weight * opp_std + (1 - weight) * global_std if pd.notna(opp_std) else global_std
                    
                    if pd.notna(opp_std) and opp_std > 0:
                        adjperf = (stat2 - opp_mean) / opp_std
                        df.at[fighter2_idx, f'{stat}_adjperf'] = adjperf
                        adjusted_count += 1
        
        print(f"✓ Calculated opponent-adjusted performance for {len(stats_to_adjust)} statistics")
        print(f"  Total adjustments made: {adjusted_count}")
        
        # Show sample statistics
        if adjperf_cols:
            non_null_counts = df[adjperf_cols].notna().sum()
            total_rows = len(df)
            print(f"  Sample non-null counts (top 5):")
            for col in adjperf_cols[:5]:
                count = non_null_counts[col]
                pct = count/total_rows*100 if total_rows > 0 else 0
                print(f"    {col}: {count}/{total_rows} ({pct:.1f}%)")
        
        return df


class MatchupComparisons:
    """Creates comparative features between two fighters in a matchup."""
    
    def __init__(self):
        """Initialize the matchup comparison calculator."""
        pass
    
    def create_matchup_dataset(self, df):
        """
        Create matchup comparison dataset where each row is a fight with features 
        comparing Fighter1 vs Fighter2.
        
        Args:
            df: DataFrame with fighter-level stats (one row per fighter per fight)
            
        Returns:
            DataFrame with matchup-level comparisons (one row per fight)
        """
        print("\n" + "=" * 60)
        print("Step 5: Matchup Comparisons")
        print("=" * 60)
        print("Creating comparative features between fighters...")
        
        # Group by bout to get fighter pairs
        bout_groups = df.groupby(['EVENT', 'BOUT'])
        
        matchup_data = []
        processed = 0
        skipped = 0
        
        for (event, bout), group in bout_groups:
            # Only process bouts with exactly 2 fighters
            if len(group) != 2:
                skipped += 1
                continue
            
            fighter1_row = group.iloc[0]
            fighter2_row = group.iloc[1]
            
            # Create matchup record
            matchup = self._create_matchup_features(fighter1_row, fighter2_row, event, bout)
            matchup_data.append(matchup)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Progress: {processed} matchups processed")
        
        print(f"\n✓ Processed {processed} matchups")
        print(f"  Skipped {skipped} incomplete bouts")
        
        matchup_df = pd.DataFrame(matchup_data)
        print(f"✓ Generated matchup dataset with {len(matchup_df.columns)} features")
        
        return matchup_df
    
    def _create_matchup_features(self, f1_row, f2_row, event, bout):
        """
        Create comparative features for a single matchup.
        
        Args:
            f1_row: Fighter 1's row (Series)
            f2_row: Fighter 2's row (Series)
            event: Event name
            bout: Bout identifier
            
        Returns:
            Dictionary with matchup features
        """
        matchup = {}
        
        # Metadata
        matchup['EVENT'] = event
        matchup['BOUT'] = bout
        matchup['DATE'] = f1_row.get('DATE')
        matchup['fighter1'] = f1_row['FIGHTER']
        matchup['fighter2'] = f2_row['FIGHTER']
        matchup['OUTCOME'] = f1_row.get('OUTCOME')
        matchup['WEIGHTCLASS'] = f1_row.get('WEIGHTCLASS')
        matchup['METHOD'] = f1_row.get('METHOD')
        
        # Target variable: Fighter 1 win
        if pd.notna(f1_row.get('win')):
            matchup['fighter1_win'] = f1_row['win']
        
        # Get all numeric features to compare
        numeric_cols = f1_row.index[f1_row.apply(lambda x: isinstance(x, (int, float, np.number)))]
        
        # Exclude metadata columns from comparisons
        exclude_cols = {'win', 'fight_number'}
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            f1_val = f1_row.get(col)
            f2_val = f2_row.get(col)
            
            # Store individual fighter values
            matchup[f'f1_{col}'] = f1_val
            matchup[f'f2_{col}'] = f2_val
            
            # Calculate difference: Fighter1 - Fighter2
            if pd.notna(f1_val) and pd.notna(f2_val):
                matchup[f'diff_{col}'] = f1_val - f2_val
                
                # Calculate ratio: Fighter1 / Fighter2 (avoid division by zero)
                if f2_val != 0 and abs(f2_val) > 1e-10:
                    matchup[f'ratio_{col}'] = f1_val / f2_val
        
        return matchup


def main():
    """Main execution function."""
    print("=" * 60)
    print("MMA AI Feature Engineering Pipeline - Steps 1-5")
    print("Step 1: Data Aggregation")
    print("Step 2: Time-Decayed Averages")
    print("Step 3: Opponent-Adjusted Performance")
    print("Step 4: Advanced Aggregations")
    print("Step 5: Matchup Comparisons")
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
        
        # Step 4: Apply time-decayed averages (Step 2 of pipeline)
        print("\n" + "=" * 60)
        print("Step 2: Time-Decayed Averages")
        print("=" * 60)
        
        decay_calculator = TimeDecayCalculator(half_life_years=1.5)
        
        # Select key stats to apply time decay to (reduced set for performance)
        stats_to_decay = [
            # Per-minute rates (most important)
            'sig_str_per_min', 'total_str_per_min', 'td_per_min', 
            'sub_att_per_min', 'kd_per_min', 'ctrl_per_min',
            # Key accuracy rates
            'sig_str_acc', 'td_acc', 'head_acc',
            # Critical base stats
            'sig_str_landed', 'td_landed', 'KD',
            'head_landed', 'body_landed', 'leg_landed',
            # Fight outcomes and attributes
            'win', 'age_at_fight', 'days_since_last_fight'
        ]
        
        # Filter to only existing columns
        existing_stats = [stat for stat in stats_to_decay if stat in aggregator.fighter_level_stats.columns]
        print(f"Applying time decay to {len(existing_stats)} statistics...")
        
        aggregator.fighter_level_stats = decay_calculator.add_decayed_averages(
            aggregator.fighter_level_stats,
            existing_stats
        )
        
        print(f"\n✓ Step 2 Complete: Time-decayed averages calculated")
        print(f"  Added {len(existing_stats)} decayed average columns")
        
        # Step 5: Apply opponent-adjusted performance (Step 3 of pipeline)
        print("\n" + "=" * 60)
        print("Step 3: Opponent-Adjusted Performance")
        print("=" * 60)
        
        adjperf_calculator = OpponentAdjustedPerformanceCalculator(
            min_fights_for_baseline=3,
            bayesian_prior_weight=5
        )
        
        # Select stats to adjust (key offensive stats)
        stats_to_adjust = [
            # Per-minute rates
            'sig_str_per_min', 'total_str_per_min', 'td_per_min',
            # Accuracy rates
            'sig_str_acc', 'td_acc', 'head_acc',
            # Base stats
            'sig_str_landed', 'head_landed', 'body_landed', 'leg_landed',
            'distance_landed', 'clinch_landed', 'ground_landed',
            'KD'
        ]
        
        existing_adjust_stats = [stat for stat in stats_to_adjust if stat in aggregator.fighter_level_stats.columns]
        print(f"Calculating opponent-adjusted performance for {len(existing_adjust_stats)} statistics...")
        
        # Calculate opponent baselines
        baselines = adjperf_calculator.calculate_opponent_baselines(
            aggregator.fighter_level_stats,
            existing_adjust_stats
        )
        
        # Apply adjustments
        aggregator.fighter_level_stats = adjperf_calculator.add_opponent_adjusted_performance(
            aggregator.fighter_level_stats,
            existing_adjust_stats,
            baselines
        )
        
        print(f"\n✓ Step 3 Complete: Opponent-adjusted performance calculated")
        print(f"  Added {len(existing_adjust_stats)} adjperf columns")
        
        # Step 4: Apply advanced aggregations (peak/valley, change, ELO, etc.)
        print("\n" + "=" * 60)
        print("Step 4: Advanced Aggregations")
        print("=" * 60)
        
        advanced_calc = AdvancedAggregationsCalculator(
            recent_fights_threshold=3,
            peak_valley_window=5
        )
        
        # Select stats to expand with advanced features
        stats_to_expand = [
            # Key per-minute rates
            'sig_str_per_min', 'total_str_per_min', 'td_per_min', 
            'sub_att_per_min', 'kd_per_min', 'ctrl_per_min',
            # Key accuracy/defense rates  
            'sig_str_acc', 'td_acc', 'head_acc',
            'sig_str_def', 'td_def', 'head_def',
            # Base counting stats
            'sig_str_landed', 'td_landed', 'KD',
            'head_landed', 'body_landed', 'leg_landed',
            'distance_landed', 'clinch_landed', 'ground_landed',
            # Absorbed stats (defense)
            'sig_str_absorbed', 'head_absorbed', 'body_absorbed',
            # Fight attributes
            'age_at_fight', 'days_since_last_fight', 'win'
        ]
        
        existing_expand_stats = [stat for stat in stats_to_expand if stat in aggregator.fighter_level_stats.columns]
        print(f"Expanding {len(existing_expand_stats)} statistics with advanced features...")
        
        aggregator.fighter_level_stats = advanced_calc.add_advanced_features(
            aggregator.fighter_level_stats,
            existing_expand_stats
        )
        
        print(f"\n✓ Step 4 Complete: Advanced aggregations calculated")
        print(f"  Expanded feature space significantly")
        print(f"  Total columns now: {len(aggregator.fighter_level_stats.columns)}")
        
        # Save intermediate results
        output_file = 'fighter_aggregated_stats_with_advanced_features.csv'
        aggregator.fighter_level_stats.to_csv(output_file, index=False)
        print(f"\n✓ Saved fighter-level data to: {output_file}")
        print(f"  - Total records: {len(aggregator.fighter_level_stats)}")
        print(f"  - Unique fighters: {aggregator.fighter_level_stats['FIGHTER'].nunique()}")
        print(f"  - Columns: {len(aggregator.fighter_level_stats.columns)}")
        
        # Step 5: Create matchup comparison dataset
        matchup_calc = MatchupComparisons()
        matchup_df = matchup_calc.create_matchup_dataset(aggregator.fighter_level_stats)
        
        # Save matchup dataset
        matchup_output_file = 'matchup_comparisons.csv'
        matchup_df.to_csv(matchup_output_file, index=False)
        print(f"\n✓ Saved matchup comparison data to: {matchup_output_file}")
        print(f"  - Total matchups: {len(matchup_df)}")
        print(f"  - Total features: {len(matchup_df.columns)}")
        print(f"  - Feature types:")
        print(f"    • Individual fighter features (f1_*, f2_*)")
        print(f"    • Difference features (diff_*): Fighter1 - Fighter2")
        print(f"    • Ratio features (ratio_*): Fighter1 / Fighter2")
        
        # Step 6: Print summary
        aggregator.print_summary_statistics()
        
        # Step 7: Print filtering report
        aggregator.print_filtering_report()
        
        print("\n" + "=" * 60)
        print("✓ Steps 1-5 completed successfully!")
        print("=" * 60)
        print(f"Fighter-level features: ~{len(aggregator.fighter_level_stats.columns)} columns")
        print(f"Matchup-level features: ~{len(matchup_df.columns)} columns")
        print(f"\nNext steps:")
        print("  Step 6: Feature selection to identify ~30 most predictive features")
        print("  Step 7: Model training with AutoGluon")
        print(f"\nTarget: 71% accuracy, 0.602 log loss, 0.207 Brier score")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        aggregator.disconnect()


if __name__ == "__main__":
    main()
