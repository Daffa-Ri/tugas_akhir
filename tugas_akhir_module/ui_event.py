import gradio as gr
from .process_model import load_model, save_model, list_model
from .train_scheme import train_model_scheme

def change_dropdown(name: str):
	model = load_model(name=name)
	return model, gr.update(visible=True), gr.update(choices=list_model())

def train_update(model, train_path: str, val_path: str, name: str, epochs: int, finetune_epochs: int, learning_rate: float, finetune_learning_rate: float):
	if (not name.endswith(".h5")):
		name+=".h5"
	model, figure = train_model_scheme(model, train_path, val_path, epochs, finetune_epochs, learning_rate, finetune_learning_rate)
	save_model(model, name=name)
	return [
			model,
			figure,
			gr.update(choices=list_model(), value=name)]