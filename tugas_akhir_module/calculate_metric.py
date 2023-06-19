def calculate_metric(confusion_matrix):
	true_pneumonia = confusion_matrix[1][1]
	false_pneumonia = confusion_matrix[0][1]
	true_normal = confusion_matrix[0][0]
	false_normal = confusion_matrix [1][0]

	accuracy = (true_pneumonia + true_normal) / (true_pneumonia + false_pneumonia + true_normal + false_normal)
	precision = (true_pneumonia) / (true_pneumonia + false_pneumonia)
	recall = (true_pneumonia) / (true_pneumonia + false_normal)
	f1_score = (2 * precision * recall) / (precision + recall)

	return {
			"Accuracy" : float(accuracy),
			"Precision" : float(precision),
			"F1Score" : float(f1_score)
	}