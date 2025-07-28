#!/usr/bin/env python3
"""
Data Extraction Tool for DDoS Dataset

This script extracts data from a DDoS dataset based on specific activities and features.
It filters the dataset by activity and applies feature selection to create focused datasets
for analysis or machine learning tasks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


class DataExtractor:
    """Class to handle data extraction from DDoS dataset."""
    
    def __init__(self, datasets_dir: Path, features_dir: Path):
        """
        Initialize the DataExtractor.
        
        Args:
            datasets_dir: Path to the datasets directory
            features_dir: Path to the features directory
        """
        self.datasets_dir = datasets_dir
        self.features_dir = features_dir
        self.ddos_file_path = datasets_dir / 'ddos.parquet'
        
    def load_dataset(self) -> pd.DataFrame:
        """Load the DDoS dataset from parquet file."""
        if not self.ddos_file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.ddos_file_path}")
        return pd.read_parquet(self.ddos_file_path)
    
    def load_features(self, features_type: str) -> List[str]:
        """
        Load feature list from JSON file.
        
        Args:
            features_type: Type of features to load (e.g., 'extra_tree', 'anova', 'information_gain')
            
        Returns:
            List of feature names
        """
        features_file_path = self.features_dir / f'{features_type}.json'
        if not features_file_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_file_path}")
        
        with open(features_file_path, 'r') as f:
            features_data = json.load(f)
        
        return features_data['features']
    
    def get_available_activities(self) -> List[str]:
        """Get list of available activities in the dataset."""
        dataset = self.load_dataset()
        return sorted(dataset['activity'].unique().tolist())
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature types."""
        feature_files = list(self.features_dir.glob('*.json'))
        return [f.stem for f in feature_files]
    
    @staticmethod
    def format_file_name(name: str) -> str:
        """
        Format the file name to be more readable.
        
        Args:
            name: Original name
            
        Returns:
            Formatted name (lowercase with underscores)
        """
        return name.replace('-', '_').lower()
    
    def extract_data(self, activity: str, features_type: str, output_dir: Optional[Path] = None) -> Path:
        """
        Extract data for specified activity and features.
        
        Args:
            activity: Activity to filter by
            features_type: Type of features to use
            output_dir: Output directory (defaults to datasets directory)
            
        Returns:
            Path to the saved file
        """
        # Load dataset and features
        dataset = self.load_dataset()
        features = self.load_features(features_type)
        
        # Validate activity exists
        available_activities = dataset['activity'].unique()
        if activity not in available_activities:
            raise ValueError(f"Activity '{activity}' not found. Available activities: {sorted(available_activities)}")
        
        # Apply feature selection
        if 'activity' not in features:
            features.append('activity')  # Ensure activity column is included
        
        # Filter features that exist in the dataset
        existing_features = [f for f in features if f in dataset.columns]
        if len(existing_features) != len(features):
            missing_features = set(features) - set(existing_features)
            print(f"Warning: Some features not found in dataset: {missing_features}")
        
        dataset_filtered = dataset[existing_features]
        
        # Filter by activity
        activity_filtered = dataset_filtered[dataset_filtered['activity'] == activity]
        
        if activity_filtered.empty:
            raise ValueError(f"No data found for activity '{activity}'")
        
        # Save to CSV
        output_dir = output_dir or self.datasets_dir
        output_file_path = output_dir / f'{self.format_file_name(activity)}.csv'
        activity_filtered.to_csv(output_file_path, index=False)
        
        print(f"Extracted {len(activity_filtered)} rows for activity '{activity}'")
        print(f"Used {len(existing_features)} features from '{features_type}' feature set")
        print(f"Filtered dataset saved to: {output_file_path}")
        
        return output_file_path


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract data from DDoS dataset by activity and features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -a Attack-TCP-Flag-OSYN -f extra_tree
  %(prog)s --activity Benign --features anova --output /path/to/output
  %(prog)s --list-activities
  %(prog)s --list-features
        """
    )
    
    parser.add_argument(
        '-a', '--activity',
        type=str,
        help='Activity to extract (e.g., Attack-TCP-Flag-OSYN, Benign)'
    )
    
    parser.add_argument(
        '-f', '--features',
        type=str,
        help='Feature set to use (e.g., extra_tree, anova, information_gain)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory (default: datasets directory)'
    )
    
    parser.add_argument(
        '--list-activities',
        action='store_true',
        help='List all available activities in the dataset'
    )
    
    parser.add_argument(
        '--list-features',
        action='store_true',
        help='List all available feature sets'
    )
    
    return parser


def main():
    """Main function to run the data extraction tool."""
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    datasets_dir = project_dir / 'datasets'
    features_dir = project_dir / 'features'
    
    # Create data extractor
    extractor = DataExtractor(datasets_dir, features_dir)
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle list commands
        if args.list_activities:
            activities = extractor.get_available_activities()
            print("Available activities:")
            for activity in activities:
                print(f"  - {activity}")
            return
        
        if args.list_features:
            features = extractor.get_available_features()
            print("Available feature sets:")
            for feature in features:
                print(f"  - {feature}")
            return
        
        # Validate required arguments
        if not args.activity or not args.features:
            parser.print_help()
            print("\nError: Both --activity and --features are required (unless using --list-* options)")
            sys.exit(1)
        
        # Extract data
        output_path = extractor.extract_data(
            activity=args.activity,
            features_type=args.features,
            output_dir=args.output
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