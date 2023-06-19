import tensorflow as tf

import gradio as gr

def predict_image(model, img):
	try:
		img = tf.keras.utils.img_to_array(img)
	except:
		raise gr.Error("Please Enter an Image")
	else:
		img_resized = tf.image.resize(img, img_size, method='nearest')

		predictions = model.predict(tf.expand_dims(img_resized, 0))
		score = tf.nn.softmax(predictions[0])

		return [{class_names[i]: float(score[i]) for i in range(2)}, tf.keras.utils.array_to_img(img_resized)]