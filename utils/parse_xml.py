# Parse the generated_flows.xml file to extract flow information
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import csv

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

selected_activity = "attack_tcp_flag_osyn"

feature_map_dir = project_dir / "templates" / selected_activity
feature_map_path = feature_map_dir / "xml_to_feature_map.json"

data_dir = project_dir / "generated_data"
file_name = "attack_tcp_osyn_2025-06-30.xml"
file_path = data_dir / file_name

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    flows = []

    # Read the feature map JSON file
    with open(feature_map_path, 'r') as f:
        feature_map = json.load(f)
        
    for network_flow in root.findall('network_flow'):
        flow = {}

        for feature in feature_map:
            path = feature_map[feature]['path']
            attribute = feature_map[feature]['attribute']
            flow[feature] = network_flow.find(path).get(attribute)


        flows.append(flow)
        
    
    return flows

if __name__ == "__main__":    
    flow_data = parse_xml(file_path)

    fieldnames = flow_data[0].keys()

    output_csv_path = data_dir / f"{file_name.replace('.xml', '.csv')}"
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flow_data)
    print(f"Parsed {len(flow_data)} flows from {file_name} and saved to {output_csv_path}")
    


