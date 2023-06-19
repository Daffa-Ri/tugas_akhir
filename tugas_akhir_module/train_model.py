import tensorflow as tf

def callbacks (monitor: str, patience: int = 20):
	return tf.keras.callbacks.EarlyStopping(
		monitor = monitor,
		patience = patience,
		restore_best_weights=True
	)

def train_model(model, train_ds, val_ds, epochs:int, learning_rate: int, initial_epoch:int = 0, finetune: bool = False):
	model.layers[3].trainable = finetune

	model.compile(
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
	)

	history = model.fit(
			train_ds,
			epochs=initial_epoch+epochs,
			initial_epoch=initial_epoch,
			validation_data=val_ds,
			callbacks = [callbacks('val_loss')]
	)

	return [model, history.history]