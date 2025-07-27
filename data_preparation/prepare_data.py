#!/usr/bin/env python3
"""
Training Data Preparation Tool for DDoS Datasets

This script prepares training data for machine learning models by combining datasets,
XML templates, and instruction files. It creates JSON training data in the format
required for model fine-tuning with system/user/assistant message structures.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from lxml import etree


class DataPreparer:
    """Class to handle training data preparation for DDoS datasets."""
    
    def __init__(self, project_dir: Path):
        """
        Initialize the DataPreparer.
        
        Args:
            project_dir: Path to the project directory
        """
        self.project_dir = project_dir
        self.datasets_dir = project_dir / 'datasets'
        self.templates_dir = project_dir / 'templates'
        self.training_data_dir = project_dir / 'training_data'
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.datasets_dir, self.templates_dir, self.training_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_available_activities(self) -> List[str]:
        """Get list of available activities (datasets)."""
        dataset_files = list(self.datasets_dir.glob('*.csv'))
        return [f.stem for f in dataset_files]
    
    def get_available_feature_sets(self, activity: str) -> List[str]:
        """
        Get list of available feature sets for a given activity.
        
        Args:
            activity: Activity name
            
        Returns:
            List of available feature sets
        """
        activity_template_dir = self.templates_dir / activity
        if not activity_template_dir.exists():
            return []
        
        xml_files = list(activity_template_dir.glob('*.xml'))
        return [f.stem for f in xml_files]
    
    def load_dataset(self, activity: str) -> pd.DataFrame:
        """
        Load the dataset for the specified activity.
        
        Args:
            activity: Activity name
            
        Returns:
            Loaded DataFrame
        """
        file_path = self.datasets_dir / f'{activity}.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
        
        return pd.read_csv(file_path)
    
    def load_xml_template(self, activity: str, feature_set: str) -> str:
        """
        Load the XML template for the specified activity and feature set.
        
        Args:
            activity: Activity name
            feature_set: Feature set name
            
        Returns:
            XML template content as string
        """
        file_path = self.templates_dir / activity / f'{feature_set}.xml'
        if not file_path.exists():
            raise FileNotFoundError(f"Template file {file_path} does not exist.")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def load_instructions(self, activity: str, feature_set: str) -> Tuple[str, str]:
        """
        Load the system and user instructions for the specified activity and feature set.
        
        Args:
            activity: Activity name
            feature_set: Feature set name
            
        Returns:
            Tuple of (system_instructions, user_instructions)
        """
        base_path = self.templates_dir / activity / feature_set
        system_file = f'{base_path}_system_instructions.txt'
        user_file = f'{base_path}_user_instructions.txt'
        
        if not Path(system_file).exists():
            raise FileNotFoundError(f"System instructions file {system_file} does not exist.")
        if not Path(user_file).exists():
            raise FileNotFoundError(f"User instructions file {user_file} does not exist.")
        
        with open(system_file, 'r', encoding='utf-8') as file:
            system_instructions = file.read().strip()
        
        with open(user_file, 'r', encoding='utf-8') as file:
            user_instructions = file.read().strip()
        
        return system_instructions, user_instructions
    
    def process_xml_template(self, xml_template: str, row_data: pd.Series) -> str:
        """
        Process XML template by replacing placeholders with actual data.
        
        Args:
            xml_template: XML template string
            row_data: Row data from DataFrame
            
        Returns:
            Processed XML string
        """
        # Replace placeholders with actual values
        processed_xml = xml_template
        for column in row_data.index:
            placeholder = f'{{{column}}}'
            value = str(row_data[column])
            processed_xml = processed_xml.replace(placeholder, value)
        
        # Parse and format XML
        parser = etree.XMLParser(remove_blank_text=True)
        try:
            element = etree.fromstring(processed_xml, parser=parser)
            return etree.tostring(element, encoding='unicode', pretty_print=True)
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML template after processing: {e}")
    
    def prepare_training_data(self, activity: str, feature_set: str, 
                            output_file: Optional[Path] = None, 
                            max_samples: Optional[int] = None) -> Path:
        """
        Prepare training data for the specified activity and feature set.
        
        Args:
            activity: Activity name
            feature_set: Feature set name
            output_file: Custom output file path (optional)
            max_samples: Maximum number of samples to process (optional)
            
        Returns:
            Path to the generated training data file
        """
        print(f"Loading dataset for activity: {activity}")
        df = self.load_dataset(activity)
        
        print(f"Loading XML template for feature set: {feature_set}")
        xml_template = self.load_xml_template(activity, feature_set)
        
        print("Loading instruction files...")
        system_instructions, user_instructions = self.load_instructions(activity, feature_set)
        
        # Limit samples if specified
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)
            print(f"Processing first {max_samples} samples")
        
        print(f"Processing {len(df)} rows...")
        training_data = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                processed_xml = self.process_xml_template(xml_template, row)
                
                training_data.append({
                    'messages': [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_instructions},
                        {"role": "assistant", "content": processed_xml}
                    ]
                })
                
                # Progress indicator
                if (idx + 1) % 100 == 0 or idx == len(df) - 1:
                    print(f"  Processed {idx + 1}/{len(df)} rows")
                    
            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                continue
        
        # Determine output file path
        if output_file is None:
            output_file = self.training_data_dir / f'{activity}_{feature_set}.json'
        
        # Save training data
        print(f"Saving training data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(training_data, file, indent=2, ensure_ascii=False)
        
        print(f"Successfully created {len(training_data)} training samples")
        return output_file


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare training data for DDoS machine learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -a attack_tcp_flag_osyn -f extra_tree
  %(prog)s --activity attack_tcp_flag_ack_psh --feature-set anova --max-samples 1000
  %(prog)s -a attack_tcp_flag_osyn -f extra_tree --output custom_training_data.json
  %(prog)s --list-activities
  %(prog)s --list-feature-sets attack_tcp_flag_osyn
        """
    )
    
    parser.add_argument(
        '-a', '--activity',
        type=str,
        help='Activity name to prepare data for (e.g., attack_tcp_flag_osyn)'
    )
    
    parser.add_argument(
        '-f', '--feature-set',
        type=str,
        help='Feature set to use (e.g., extra_tree, anova, information_gain)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of samples to process (for testing/debugging)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path for training data (default: training_data/<activity>_<feature_set>.json)'
    )
    
    parser.add_argument(
        '--list-activities',
        action='store_true',
        help='List all available activities (datasets)'
    )
    
    parser.add_argument(
        '--list-feature-sets',
        type=str,
        metavar='ACTIVITY',
        help='List all available feature sets for the specified activity'
    )
    
    return parser


def main():
    """Main function to run the data preparation tool."""
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    # Create data preparer
    preparer = DataPreparer(project_dir)
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle list commands
        if args.list_activities:
            activities = preparer.get_available_activities()
            print("Available activities:")
            if activities:
                for activity in sorted(activities):
                    print(f"  - {activity}")
            else:
                print("  No activities found in datasets directory")
            return
        
        if args.list_feature_sets:
            feature_sets = preparer.get_available_feature_sets(args.list_feature_sets)
            print(f"Available feature sets for '{args.list_feature_sets}':")
            if feature_sets:
                for feature_set in sorted(feature_sets):
                    print(f"  - {feature_set}")
            else:
                print(f"  No feature sets found for activity '{args.list_feature_sets}'")
            return
        
        # Validate required arguments
        if not args.activity or not args.feature_set:
            parser.print_help()
            print("\nError: Both --activity and --feature-set are required (unless using --list-* options)")
            sys.exit(1)
        
        # Prepare training data
        output_path = preparer.prepare_training_data(
            activity=args.activity,
            feature_set=args.feature_set,
            output_file=args.output,
            max_samples=args.max_samples
        )
        
        print(f"\n✅ Training data preparation completed successfully!")
        print(f"📁 Output file: {output_path}")
        
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