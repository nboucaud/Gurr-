import json

base_json_path = 'config/UniCATS_txt2vec.json'  
example_json_path = 'egs/tts/UniCATS/CTStxt2vec/exp_config.json'  

merged_output_path = base_json_path

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_merged_json(base_config, override_config, output_path):
    base_config.update(override_config)
    with open(output_path, 'w') as file:
        json.dump(base_config, file, indent=4)

if __name__ == "__main__":
    base_config = load_json(base_json_path)
    example_config = load_json(example_json_path)

    save_merged_json(base_config, example_config, merged_output_path)
