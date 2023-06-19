import yaml

with open("../config.yaml", "r") as f:
	config = yaml.safe_load(f)

def config_image_size():
	return config['image_size']

def config_model_location():
	return config['model_location']

def config_class_names():
	return ["NORMAL", "PNEUMONIA"]