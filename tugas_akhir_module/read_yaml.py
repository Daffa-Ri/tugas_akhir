import yaml

script_dir = os.path.dirname(__file__)
absolute_path = os.path.join(script_dir, 'config.yaml')
with open(absolute_path, "r") as f:
	config = yaml.safe_load(f)

def config_image_size():
	return config['image_size']

def config_model_location():
	return os.path.join(script_dir, config['model_location'])

def config_class_names():
	return ["NORMAL", "PNEUMONIA"]