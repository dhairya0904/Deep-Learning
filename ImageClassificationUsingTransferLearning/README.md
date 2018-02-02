# Image Classification Using Transfer learning

model.py contains Model class that uses transfer learning with pre-trained vgg 16 model to classify images. Notebook contains detailed explanation about use of class. Model does not take much time to train.

Model class methods:
	generateData:
		Arguments - path of train or test data
		return  - features, labels, ImageDataGenerator
	fit:
		Arguments - train_features , train_labels, validation_features, validation_labels, epochs, batch_size, verbose
		return - trained deep neural network
	predict:
		Arguments - test_features
		return - probabilities
	predict_class
		Arguments - test_features
		return - class labels

Data should be in following order:
data:
	train:
		class1
			class1-001.jpeg
			class1-002.jpeg
			.........

		class2
			class2-001.jpeg
			class2-002.jpeg
			.........

	test:
		class1
			class1-001.jpeg
			class1-002.jpeg
			.........

		class2
			class1-001.jpeg
			class1-002.jpeg
			.........


There should be a training data directory and validation data directory containing one subdirectory per image class, filled with .png or .jpg images

# Requirments
1. Keras
2. Numpy
