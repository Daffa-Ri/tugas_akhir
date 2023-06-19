import tensorflow as tf

import gradio as gr

import os


def load_dataset(path: str, batch_size: int = 32):
	pneu_path = os.path.join(path, 'PNEUMONIA')
	norm_path = os.path.join(path, 'NORMAL')

	if(not os.path.isdir(path)):
		raise gr.Error("Folder not exists")

	elif (not(os.path.isdir(pneu_path) and os.path.isdir(norm_path)) ):
		raise gr.Error("Folder invalid")

	try:
		dataset = tf.keras.utils.image_dataset_from_directory(
				path,
				image_size=img_size,
				interpolation='nearest',
				batch_size=batch_size
		)

	except Exception as e:
		raise gr.Error(e)

	AUTOTUNE = tf.data.AUTOTUNE
	dataset = dataset.prefetch(buffer_size=AUTOTUNE)

	return dataset

