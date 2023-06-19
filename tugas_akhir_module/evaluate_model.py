import tensorflow as tf

from .calculate_metric import calculate_metric
from .read_yaml import config_class_names

import plotly.express as px

def evaluate_model(model, path: str):
	test_ds = load_dataset(path, 1600)

	image_batch, actual_label = test_ds.as_numpy_iterator().next()
	prediction_label = model.predict(image_batch)
	prediction_label = tf.math.argmax(prediction_label, axis=-1)

	confusion_matrix = tf.math.confusion_matrix(actual_label, prediction_label)
	class_names = config_class_names()

	fig = px.imshow(
			confusion_matrix,
			text_auto = True,
			labels = dict(
					x = "Predicition",
					y = "Actual"
			),
			x = class_names,
			y = class_names
	)

	return [ fig, calculate_metric(confusion_matrix) ]