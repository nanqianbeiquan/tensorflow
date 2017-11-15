# coding=utf-8
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():
	dataset = []
	labelset = []
	label_map = {}
	base_dir = "/root/python/tensorflow/TensorflowPic/training_set/"
	labels = os.listdir(base_dir)
	for index, label in enumerate(labels):
		image_files = os.listdir(base_dir + label)
		for image_file in image_files:
			image_path = base_dir + label + "/" + image_file
			im = Image.open(image_path).convert('L').resize((28, 28))
			# im.show()
			dataset.append(np.asarray(im, dtype=np.float))
			labelset.append(index)
		label_map[index] = label
	return np.array(dataset), np.array(labelset), label_map

# dataset, labelset, label_map = load_dataset()

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

# dataset, labelset = randomize(dataset, labelset)

def reformat(dataset, labels, image_size, num_labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labelset)

# train_dataset, train_labels = reformat(train_dataset, train_labels, 32, len(label_map))

def check_dataset(dataset, labels, label_map, index):
	data = np.uint8(dataset[index]).reshape((28, 28))
	i = np.argwhere(labels[index] == 1)[0][0]
	im = Image.fromarray(data)
	im.show()
	print "label:", label_map[i]

def load_model():
	dataset, labelset, label_map = load_dataset()
	dataset, labelset = randomize(dataset, labelset)
	train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labelset)
	train_dataset, train_labels = reformat(train_dataset, train_labels, 28, len(label_map))
	test_dataset, test_labels = reformat(test_dataset, test_labels, 28, len(label_map))
	print "train_dataset:", train_dataset.shape
	print "train_labels:", train_labels.shape
	print "test_dataset:", test_dataset.shape
	print "test_labels:", test_labels.shape
	check_dataset(train_dataset, train_labels, label_map, 0)
	return train_dataset, train_labels, test_dataset, test_labels, label_map

if __name__ == '__main__':
	load_model()
