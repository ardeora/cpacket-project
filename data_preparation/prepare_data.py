from pathlib import Path
import pandas as pd
from lxml import etree
import json

script_path = Path(__file__).resolve().parent
project_directory = script_path.parent
dataset_path = project_directory / 'datasets'
template_path = project_directory / 'templates'
training_data_path = project_directory / 'training_data'

# Create directories if they do not exist
dataset_path.mkdir(parents=True, exist_ok=True)
template_path.mkdir(parents=True, exist_ok=True)
training_data_path.mkdir(parents=True, exist_ok=True)

selected_activity = 'attack_tcp_flag_osyn'
selected_feature_set = 'extra_tree'


def load_data():
    """
    Load the dataset based on the selected activity.
    """
    file_path = dataset_path / f'{selected_activity}.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    return df

def load_xml_template():
    """
    Load the XML template for the selected feature set.
    """
    file_path = template_path / f'{selected_activity}' / f'{selected_feature_set}.xml'
    if not file_path.exists():
        raise FileNotFoundError(f"Template file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        xml_content = file.read()
    
    return xml_content

def load_instructions():
    """
    Load the instructions for the selected feature set.
    """
    file_path = template_path / f'{selected_activity}' / f'{selected_feature_set}'
    system_instructions_file = f'{file_path}_system_instructions.txt'
    user_instructions_file = f'{file_path}_user_instructions.txt'
    
    if not Path(system_instructions_file).exists() or not Path(user_instructions_file).exists():
        raise FileNotFoundError(f"Instructions files {system_instructions_file} or {user_instructions_file} do not exist.")
    with open(system_instructions_file, 'r') as file:
        system_instructions = file.read()
    with open(user_instructions_file, 'r') as file:
        user_instructions = file.read()

    return system_instructions, user_instructions
    

def prepare_data():
    """
    Prepare the data by loading the dataset, XML template, and instructions.
    """
    df = load_data()
    system_instructions, user_instructions = load_instructions()
    parser = etree.XMLParser(remove_blank_text=True)

    # Clean the instructions by removing any leading or trailing whitespace
    system_instructions = system_instructions.strip()
    user_instructions = user_instructions.strip()
  
    training_data = []

    # Loop through each row in the DataFrame and replace each column name in the XML template
    for _, row in df.iterrows():
        xml_template = load_xml_template()
        for column in df.columns:
            xml_template = xml_template.replace(f'{{{column}}}', str(row[column]))
        
        element = etree.fromstring(xml_template, parser=parser)
        xml_template = etree.tostring(element, encoding='unicode')

        training_data.append({
            'messages': [
              {"role": "system", "content": system_instructions},
              {"role": "user", "content": user_instructions},
              {"role": "assistant", "content": xml_template}
            ]
        })
    
    # Save the prepared data to a JSON file
    output_file = training_data_path / f'{selected_activity}_{selected_feature_set}.json'
    with open(output_file, 'w') as file:
        json.dump(training_data, file, indent=2)
    
if __name__ == "__main__":
    try:
        prepare_data()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")