#!/usr/bin/env python3
"""
XML Parser Tool for Generated Network Flow Data

This script parses XML files containing network flow data and converts them to CSV format.
It uses feature mapping configuration files to extract specific attributes from XML elements
and creates structured CSV output for further analysis.
"""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


class XMLFlowParser:
    """Class to handle XML parsing and CSV conversion for network flow data."""
    
    def __init__(self, project_dir: Path):
        """
        Initialize the XMLFlowParser.
        
        Args:
            project_dir: Path to the project directory
        """
        self.project_dir = project_dir
        self.templates_dir = project_dir / 'templates'
        self.generated_data_dir = project_dir / 'generated_data'
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.templates_dir, self.generated_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_available_xml_files(self) -> List[str]:
        """Get list of available XML files in the generated_data directory."""
        if not self.generated_data_dir.exists():
            return []
        
        xml_files = list(self.generated_data_dir.glob('*.xml'))
        return [f.name for f in xml_files]
    
    def get_available_activities(self) -> List[str]:
        """Get list of available activities (template directories)."""
        if not self.templates_dir.exists():
            return []
        
        activity_dirs = [d for d in self.templates_dir.iterdir() if d.is_dir()]
        return [d.name for d in activity_dirs]
    
    def load_feature_map(self, activity: str) -> Dict[str, Dict[str, str]]:
        """
        Load the feature mapping configuration for the specified activity.
        
        Args:
            activity: Activity name
            
        Returns:
            Feature mapping dictionary
        """
        feature_map_path = self.templates_dir / activity / 'xml_to_feature_map.json'
        
        if not feature_map_path.exists():
            raise FileNotFoundError(f"Feature map file not found: {feature_map_path}")
        
        with open(feature_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def parse_xml_file(self, xml_file_path: Path, feature_map: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Parse XML file and extract flow data based on feature map.
        
        Args:
            xml_file_path: Path to the XML file
            feature_map: Feature mapping configuration
            
        Returns:
            List of flow data dictionaries
        """
        if not xml_file_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file: {e}")
        
        flows = []
        
        for network_flow in root.findall('network_flow'):
            flow = {}
            
            for feature_name, mapping in feature_map.items():
                try:
                    path = mapping['path']
                    attribute = mapping['attribute']
                    
                    element = network_flow.find(path)
                    if element is not None:
                        value = element.get(attribute)
                        flow[feature_name] = value if value is not None else ''
                    else:
                        flow[feature_name] = ''
                        print(f"Warning: Element not found for path '{path}' in feature '{feature_name}'")
                
                except KeyError as e:
                    print(f"Warning: Invalid feature mapping for '{feature_name}': missing {e}")
                    flow[feature_name] = ''
                except Exception as e:
                    print(f"Warning: Error processing feature '{feature_name}': {e}")
                    flow[feature_name] = ''
            
            flows.append(flow)
        
        return flows
    
    def save_to_csv(self, flows: List[Dict[str, str]], output_path: Path) -> None:
        """
        Save flow data to CSV file.
        
        Args:
            flows: List of flow data dictionaries
            output_path: Path to the output CSV file
        """
        if not flows:
            raise ValueError("No flow data to save")
        
        fieldnames = flows[0].keys()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flows)
    
    def parse_and_convert(self, xml_filename: str, activity: str, 
                         output_path: Optional[Path] = None) -> Path:
        """
        Parse XML file and convert to CSV format.
        
        Args:
            xml_filename: Name of the XML file to parse
            activity: Activity name for feature mapping
            output_path: Custom output path (optional)
            
        Returns:
            Path to the generated CSV file
        """
        # Construct paths
        xml_file_path = self.generated_data_dir / xml_filename
        
        print(f"Loading feature map for activity: {activity}")
        feature_map = self.load_feature_map(activity)
        
        print(f"Parsing XML file: {xml_filename}")
        flows = self.parse_xml_file(xml_file_path, feature_map)
        
        if not flows:
            raise ValueError(f"No network flows found in {xml_filename}")
        
        # Determine output path
        if output_path is None:
            csv_filename = xml_filename.replace('.xml', '.csv')
            output_path = self.generated_data_dir / csv_filename
        
        print(f"Saving {len(flows)} flows to CSV: {output_path}")
        self.save_to_csv(flows, output_path)
        
        return output_path
    
    def get_xml_file_info(self, xml_filename: str) -> Dict[str, any]:
        """
        Get information about an XML file.
        
        Args:
            xml_filename: Name of the XML file
            
        Returns:
            Dictionary with file information
        """
        xml_file_path = self.generated_data_dir / xml_filename
        
        if not xml_file_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_file_path}")
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            flow_count = len(root.findall('network_flow'))
            file_size = xml_file_path.stat().st_size
            
            return {
                'filename': xml_filename,
                'path': xml_file_path,
                'flow_count': flow_count,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
        
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML file: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Parse XML network flow data and convert to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -x attack_tcp_osyn_2025-06-30.xml -a attack_tcp_flag_osyn
  %(prog)s --xml-file generated_flows.xml --activity attack_tcp_flag_ack_psh --output custom_output.csv
  %(prog)s --list-xml-files
  %(prog)s --list-activities
  %(prog)s --info attack_tcp_osyn_2025-06-30.xml
        """
    )
    
    parser.add_argument(
        '-x', '--xml-file',
        type=str,
        help='XML filename to parse (must be in generated_data directory)'
    )
    
    parser.add_argument(
        '-a', '--activity',
        type=str,
        help='Activity name for feature mapping (e.g., attack_tcp_flag_osyn)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output CSV file path (default: same name as XML but with .csv extension)'
    )
    
    parser.add_argument(
        '--list-xml-files',
        action='store_true',
        help='List all available XML files in generated_data directory'
    )
    
    parser.add_argument(
        '--list-activities',
        action='store_true',
        help='List all available activities (template directories)'
    )
    
    parser.add_argument(
        '--info',
        type=str,
        metavar='XML_FILE',
        help='Show information about the specified XML file'
    )
    
    return parser


def main():
    """Main function to run the XML parsing tool."""
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    
    # Create XML parser
    parser_tool = XMLFlowParser(project_dir)
    
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle list commands
        if args.list_xml_files:
            xml_files = parser_tool.get_available_xml_files()
            print("Available XML files:")
            if xml_files:
                for xml_file in sorted(xml_files):
                    print(f"  - {xml_file}")
            else:
                print("  No XML files found in generated_data directory")
            return
        
        if args.list_activities:
            activities = parser_tool.get_available_activities()
            print("Available activities:")
            if activities:
                for activity in sorted(activities):
                    print(f"  - {activity}")
            else:
                print("  No activities found in templates directory")
            return
        
        # Handle info command
        if args.info:
            info = parser_tool.get_xml_file_info(args.info)
            print(f"XML File Information:")
            print(f"  Filename: {info['filename']}")
            print(f"  Path: {info['path']}")
            print(f"  Flow Count: {info['flow_count']}")
            print(f"  File Size: {info['file_size_mb']} MB ({info['file_size_bytes']} bytes)")
            return
        
        # Validate required arguments
        if not args.xml_file or not args.activity:
            parser.print_help()
            print("\nError: Both --xml-file and --activity are required (unless using --list-* or --info options)")
            sys.exit(1)
        
        # Parse and convert XML to CSV
        output_path = parser_tool.parse_and_convert(
            xml_filename=args.xml_file,
            activity=args.activity,
            output_path=args.output
        )
        
        print(f"\n✅ XML parsing completed successfully!")
        print(f"📁 Output CSV file: {output_path}")
        
        # Show some statistics
        info = parser_tool.get_xml_file_info(args.xml_file)
        print(f"📊 Processed {info['flow_count']} network flows")
        print(f"📏 Input file size: {info['file_size_mb']} MB")
        
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
    


