import tensorflow as tf

import gradio as gr

from .read_yaml import config_image_size, config_class_names

def predict_image(model, img):
	try:
		img = tf.keras.utils.img_to_array(img)
	except:
		raise gr.Error("Please Enter an Image")
	else:
		img_size = config_image_size()
		class_names = config_class_names()
		img_resized = tf.image.resize(img, img_size, method='nearest')

		predictions = model.predict(tf.expand_dims(img_resized, 0))
		score = tf.nn.softmax(predictions[0])

		return {class_names[i]: float(score[i]) for i in range(2)}