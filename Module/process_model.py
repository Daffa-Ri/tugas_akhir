import tensorflow as tf
from tensorflow import keras

import gradio as gr

def load_model(name: str):
	try:
		model = keras.models.load_model(model_path+name)
	except Exception as e:
		raise gr.Error(e)

	return model

def list_model():
	model_list = []
	for model in os.listdir(model_path):
		if model.endswith(".h5"):
			model_list.append(model)
	return model_list

def save_model(model, name: str):
	try:
		model.save(model_path+name)
	except Exception as e:
		raise gr.Error(e)