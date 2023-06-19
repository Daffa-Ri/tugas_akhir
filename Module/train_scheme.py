import tensorflow as tf
from load_dataset import load_dataset
from train_model import train_model

import plotly.express as px

def train_model_scheme(model, train_path: str, val_path: str, epochs: int =1000, finetune_epochs: int =1000, learning_rate: float = 0.001, finetune_learning_rate: float = 0.0001):
	train_ds = load_dataset(train_path)
	val_ds = load_dataset(val_path)

	model, history = train_model(model=model, train_ds=train_ds, val_ds=val_ds, epochs=epochs, learning_rate=learning_rate)

	finetune_initial_epoch = len(history['loss'])

	model, history_fine = train_model(model=model, train_ds=train_ds, val_ds=val_ds, epochs=finetune_epochs, learning_rate=finetune_learning_rate, initial_epoch=finetune_initial_epoch, finetune=True)
	training_metrics = list(history.keys())

	history_fine = {training_metrics[i]: history[training_metrics[i]] + history_fine[training_metrics[i]] for i in range(len(training_metrics))}

	fig = px.line(history_fine, x=range(1, len(history_fine["loss"])+1),  y=training_metrics)
	fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
	fig.add_vline(finetune_initial_epoch, annotation_text="start finetuning",  line_dash="dash")

  	return [model, fig]