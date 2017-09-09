#!usr/bin/env python
import getopt
import sys
import logging
from os import listdir
from PIL import Image
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from pprint import pprint
import time
import datetime

# Logging configuration
logging.basicConfig(filename='tensorflow_example.log', level=logging.INFO)
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Location configuration
data_dir = "data"
preprocessed_data_dir = "preprocessed_data"
data_type_dirs = {
	'train': 'train',
	'test': 'test'
}
model_dir = "model"
vis_dir= "tensorboard_log"

def preprocess(data_type):
	for file in listdir(data_dir + "/" + data_type_dirs[data_type]):
		if file.endswith(".jpg"):
			orig_image = Image.open(data_dir + "/" + data_type_dirs[data_type] + "/" + file)
			resized_image = orig_image.resize((200, 200), Image.ANTIALIAS)
			resized_image.save(preprocessed_data_dir + "/" + data_type_dirs[data_type] + "/" + file, 'JPEG', quality=90)

def load_data(data_type):
	data = []
	labels = []
	for file in listdir(preprocessed_data_dir + "/" + data_type_dirs[data_type]):
		if file.endswith(".jpg"):
			image = Image.open(preprocessed_data_dir + "/" + data_type_dirs[data_type] + "/" + file)
			data.append(np.asarray(image))
			if file.split('.')[0] == "cat":
				labels.append([1, 0])
			else:
				labels.append([0, 1])
	return data, labels
				
def generate_model():
	convnet = input_data(shape=[None, 200, 200, 3], name='input')

	convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = max_pool_2d(convnet, [1, 2, 2, 1])

	convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = avg_pool_2d(convnet, [1, 2, 2, 1])

	convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = max_pool_2d(convnet, [1, 2, 2, 1])

	convnet = conv_2d(convnet, 256, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = max_pool_2d(convnet, [1, 2, 2, 1])

	convnet = conv_2d(convnet, 128, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = max_pool_2d(convnet, [1, 2, 2, 1])

	convnet = conv_2d(convnet, 64, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = avg_pool_2d(convnet, [1, 2, 2, 1])
	
	convnet = conv_2d(convnet, 32, 2, strides=[1, 2, 2, 1], activation='relu')
	convnet = max_pool_2d(convnet, [1, 2, 2, 1])
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, learning_rate=0.001, loss='categorical_crossentropy', optimizer='adam')
	
	model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir=vis_dir)	
	return model

def train(run=100):
	model_name = "example-model"
	data, labels = load_data('train')
	data = data[:run] + data[-run:]
	labels = labels[:run] + labels[-run:]
	startTs = time.time()
	model = generate_model()
	model.fit(data, labels, n_epoch=10, show_metric=True, snapshot_step=20, batch_size=100, run_id=model_name)
	endTs = time.time()
	logger.info("Training with run=" + str(run) + " runtime=" + str(endTs - startTs))
	model.save(model_dir + "/" + model_name)
	return model
	
def test(model):
	model_name = "example-model"
	if model == None:
		model = generate_model()
		model.load(model_dir + "/" + model_name)
	data, labels = load_data('test')
	index = 0
	result = []
	for image in data:
		image = image.reshape(-1, 200, 200, 3)
		startTs = time.time()
		mod = model.predict(image)
		endTs = time.time()
		logger.info("Testing runtime=" + str(endTs - startTs))
		result.append((labels[index], mod))
		index = index + 1
	return result
		

def main(argv):
	# NOTE: call debug first, or else debug will be enabled after running the workflow
	model = None
	try:
		opts, args = getopt.getopt(argv[1:], "dpt:T",
								   ["debug", "preprocess", "train=", "test"])
	except getopt.GetoptError:
		print(sys.argv[0] + " --debug --preprocess --train --test")
		sys.exit(2)
	if not opts:
		print(sys.argv[0] + " --debug --preprocess --train --test")
	for opt, arg in opts:
		if opt in ("-d", "--debug"):
			logging.getLogger().setLevel(logging.DEBUG)
		if opt in ("-p", "--preprocess"):
			preprocess('train')
			preprocess('test')
		if opt in ("-t", "--train"):
			model = train(int(arg))
		if opt in ("-T", "--test"):
			preprocess('test')
			pprint(test(model))


if __name__ == "__main__":
	main(sys.argv)