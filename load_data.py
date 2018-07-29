import numpy as np
import struct as st

def load_data(FEATURES, LABELS):
	train_features = FEATURES
	train_labels = LABELS

	#Features Dimensions
	with open(train_features, "rb") as tf:
		tf.seek(4)
		file_features = tf.read(12) 

	#Label Dimensions
	with open(train_labels, "rb") as tl:
		tl.seek(4)
		file_labels = tl.read(4)

	features_dim  = st.unpack(">III", file_features) #(no_of_images, nRows, nColumns)
	labels_dim = st.unpack(">I", file_labels) #(no_of_images)

	number_of_images = features_dim[0]
	rows = features_dim[1]
	cols = features_dim[2]

	#Labels data
	with open(train_labels, "rb") as a:
		a.seek(8)
		labels = a.read()

	labels = np.array(list(labels))

	#Features data
	with open(train_features, "rb") as b:
		b.seek(16)
		features = b.read()

	features = np.array(list(features)).reshape(number_of_images, rows, cols)

	return features, labels

'''
Test

TRAINING_FEATURES = "mnist/train-images.idx3-ubyte"
TRAINING_LABELS = "mnist/train-labels.idx1-ubyte"
TESTING_FEATURES = "mnist/t10k-images.idx3-ubyte"
TESTING_LABELS = "mnist/t10k-labels.idx1-ubyte"

f,l = load_data(TESTING_FEATURES, TESTING_LABELS)
print(l.shape)
'''