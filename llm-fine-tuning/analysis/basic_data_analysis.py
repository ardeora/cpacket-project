#!/usr/bin/env python3
"""
Basic Data Analysis Tool for DDoS Datasets

This script performs comprehensive analysis on DDoS datasets including basic statistics,
feature distributions, correlations, and feature relationships. It can analyze any CSV
or Parquet dataset dynamically through command-line arguments.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


class DataAnalyzer:
    """Class to handle comprehensive data analysis for DDoS datasets."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize the DataAnalyzer.
        
        Args:
            dataset_path: Path to the dataset file (CSV or Parquet)
        """
        self.dataset_path = dataset_path
        self.df = None
        self.numeric_cols = None
        
    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from file (supports CSV and Parquet)."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        file_extension = self.dataset_path.suffix.lower()
        
        if file_extension == '.csv':
            self.df = pd.read_csv(self.dataset_path)
        elif file_extension == '.parquet':
            self.df = pd.read_parquet(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported: .csv, .parquet")
        
        # Get numerical columns
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        return self.df
    
    def basic_info_analysis(self) -> None:
        """Perform basic data information analysis."""
        print("=" * 50)
        print("BASIC DATA INFO")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Dataset file: {self.dataset_path.name}")
        
        # Check for activity and label columns
        if 'activity' in self.df.columns:
            activities = self.df['activity'].unique()
            print(f"Activities: {activities}")
        
        if 'label' in self.df.columns:
            labels = self.df['label'].unique()
            print(f"Labels: {labels}")
        
        print(f"Numerical columns: {len(self.numeric_cols)}")
        print(f"Total columns: {len(self.df.columns)}")
    
    def feature_statistics_analysis(self) -> None:
        """Perform feature statistics analysis."""
        print("\n" + "=" * 50)
        print("FEATURE STATISTICS")
        print("=" * 50)
        
        if not self.numeric_cols:
            print("No numerical columns found for analysis.")
            return
        
        # Basic statistics
        stats = self.df[self.numeric_cols].describe()
        print(stats.round(4))
        
        print(f"\n=== MISSING VALUES ===")
        missing_total = self.df.isnull().sum().sum()
        print(f"Total missing values: {missing_total}")
        
        if missing_total > 0:
            missing_by_column = self.df.isnull().sum()
            missing_cols = missing_by_column[missing_by_column > 0]
            print("Missing values by column:")
            for col, count in missing_cols.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({percentage:.2f}%)")
    
    def feature_distributions_analysis(self, max_features: int = 10) -> None:
        """
        Perform feature distributions analysis.
        
        Args:
            max_features: Maximum number of features to analyze in detail
        """
        print("\n" + "=" * 50)
        print("FEATURE RANGES AND DISTRIBUTIONS")
        print("=" * 50)
        
        analyzed_features = min(max_features, len(self.numeric_cols))
        print(f"Analyzing first {analyzed_features} numerical features:")
        
        for col in self.numeric_cols[:analyzed_features]:
            values = self.df[col].dropna()
            if len(values) == 0:
                print(f"\n{col}: No valid values")
                continue
                
            print(f"\n{col}:")
            print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
            print(f"  Mean: {values.mean():.4f}, Std: {values.std():.4f}")
            print(f"  Unique values: {values.nunique()}")
            
            # Check if mostly zeros
            zero_pct = (values == 0).mean() * 100
            print(f"  Zero percentage: {zero_pct:.2f}%")
    
    def correlation_analysis(self, custom_features: Optional[List[str]] = None) -> None:
        """
        Perform correlation analysis with key features.
        
        Args:
            custom_features: Custom list of features to analyze correlations
        """
        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Default key features for DDoS analysis
        default_key_features = [
            'total_header_bytes', 'packets_rate', 'syn_flag_percentage_in_total',
            'rst_flag_percentage_in_total', 'packet_IAT_mean'
        ]
        
        key_features = custom_features or default_key_features
        
        # Filter features that exist in the dataset
        existing_features = [f for f in key_features if f in self.df.columns]
        
        if not existing_features:
            print("No key features found in dataset for correlation analysis.")
            print(f"Available numerical columns: {self.numeric_cols[:10]}...")
            return
        
        print(f"Analyzing correlations for: {existing_features}")
        corr_matrix = self.df[existing_features].corr()
        print(corr_matrix.round(3))
    
    def feature_relationships_analysis(self) -> None:
        """Analyze relationships between features for pattern recognition."""
        print("\n" + "=" * 50)
        print("FEATURE RELATIONSHIPS ANALYSIS")
        print("=" * 50)
        
        # Group related features by common patterns
        feature_groups = self._group_features_by_pattern()
        
        for group_name, features in feature_groups.items():
            if features:
                print(f"\n{group_name} Features ({len(features)}): {features[:5]}...")
        
        # Analyze value patterns for each group
        print("\n=== VALUE PATTERNS ===")
        for group_name, features in feature_groups.items():
            if features:
                print(f"\n{group_name} Features:")
                for col in features[:3]:  # First 3 of each group
                    if col in self.df.columns:
                        values = self.df[col].dropna()
                        if len(values) > 0:
                            print(f"  {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
    
    def _group_features_by_pattern(self) -> dict:
        """Group features by common naming patterns."""
        feature_groups = {
            "Header": [col for col in self.df.columns if 'header' in col.lower()],
            "Flag": [col for col in self.df.columns if 'flag' in col.lower()],
            "Timing": [col for col in self.df.columns if 'IAT' in col or 'rate' in col],
            "Packet": [col for col in self.df.columns if 'packet' in col.lower() and 'rate' not in col.lower()],
            "Flow": [col for col in self.df.columns if 'flow' in col.lower()],
            "Protocol": [col for col in self.df.columns if any(proto in col.lower() for proto in ['tcp', 'udp', 'ip'])],
        }
        return feature_groups
    
    def sample_data_analysis(self, num_samples: int = 3) -> None:
        """Display sample rows from the dataset."""
        print("\n" + "=" * 50)
        print(f"SAMPLE DATA ({num_samples} rows)")
        print("=" * 50)
        print(self.df.head(num_samples))
    
    def run_complete_analysis(self, max_features: int = 10, custom_correlation_features: Optional[List[str]] = None, num_samples: int = 3) -> None:
        """
        Run complete data analysis pipeline.
        
        Args:
            max_features: Maximum number of features for distribution analysis
            custom_correlation_features: Custom features for correlation analysis
            num_samples: Number of sample rows to display
        """
        try:
            # Load dataset
            print("Loading dataset...")
            self.load_dataset()
            
            # Run all analyses
            self.basic_info_analysis()
            self.feature_statistics_analysis()
            self.feature_distributions_analysis(max_features)
            self.correlation_analysis(custom_correlation_features)
            self.feature_relationships_analysis()
            self.sample_data_analysis(num_samples)
            
            print("\n" + "=" * 50)
            print("ANALYSIS COMPLETE")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def get_available_datasets(datasets_dir: Path) -> List[str]:
    """Get list of available datasets in the datasets directory."""
    if not datasets_dir.exists():
        return []
    
    dataset_files = []
    for pattern in ['*.csv', '*.parquet']:
        dataset_files.extend(datasets_dir.glob(pattern))
    
    return [f.name for f in dataset_files]


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Perform comprehensive analysis on DDoS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d attack_tcp_flag_osyn.csv
  %(prog)s --dataset ddos.parquet --max-features 15
  %(prog)s -d attack_tcp_flag_ack_psh.csv --correlation-features total_header_bytes,packets_rate
  %(prog)s --list-datasets
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        help='Dataset filename to analyze (CSV or Parquet file)'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=Path,
        help='Full path to dataset file (overrides --dataset)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=10,
        help='Maximum number of features to analyze in detail (default: 10)'
    )
    
    parser.add_argument(
        '--correlation-features',
        type=str,
        help='Comma-separated list of features for correlation analysis'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='Number of sample rows to display (default: 3)'
    )
    
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets in the datasets directory'
    )
    
    return parser


def main():
    """Main function to run the data analysis tool."""
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    datasets_dir = project_dir / 'datasets'
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle list command
        if args.list_datasets:
            datasets = get_available_datasets(datasets_dir)
            print("Available datasets:")
            if datasets:
                for dataset in sorted(datasets):
                    print(f"  - {dataset}")
            else:
                print("  No datasets found in datasets directory")
            return
        
        # Determine dataset path
        if args.dataset_path:
            dataset_path = args.dataset_path
        elif args.dataset:
            dataset_path = datasets_dir / args.dataset
        else:
            parser.print_help()
            print("\nError: Either --dataset or --dataset-path is required (unless using --list-datasets)")
            sys.exit(1)
        
        # Parse correlation features if provided
        correlation_features = None
        if args.correlation_features:
            correlation_features = [f.strip() for f in args.correlation_features.split(',')]
        
        # Create analyzer and run analysis
        analyzer = DataAnalyzer(dataset_path)
        analyzer.run_complete_analysis(
            max_features=args.max_features,
            custom_correlation_features=correlation_features,
            num_samples=args.samples
        )
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()