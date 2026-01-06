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


def main():
    """Main execution function."""
    print("=" * 60)
    print("MMA AI Feature Engineering Pipeline - Steps 1 & 2")
    print("Step 1: Data Aggregation")
    print("Step 2: Time-Decayed Averages")
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
        
        # Step 5: Save results
        aggregator.save_aggregated_data('fighter_aggregated_stats_with_decay.csv')
        
        # Step 6: Print summary
        aggregator.print_summary_statistics()
        
        # Step 7: Print filtering report
        aggregator.print_filtering_report()
        
        print("\n" + "=" * 60)
        print("✓ Steps 1 & 2 completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        aggregator.disconnect()


if __name__ == "__main__":
    main()
