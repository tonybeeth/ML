# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import random
import time
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import keras
from itertools import cycle
import math

if __name__ == "__main__":
	image_size = 64
	batch_size = 53

	#directories containing images
	PKLOT_DIR = '../../../PKLot/PKLot/'
	PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

	#keras.callbacks.TensorBoard(log_dir='./logs')

	class PlotSequence(keras.utils.Sequence):

		def __init__(self, empty, occupied, batch_size):
			labels = [[1, 0] for i in range(len(empty))] + [[0,1] for i in range(len(occupied))]
			self.x, self.y = empty + occupied, labels
			self.batch_size = batch_size

		def __len__(self):
			return math.ceil(len(self.x) / self.batch_size)

		def __getitem__(self, idx):
			batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
			batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
			
			return np.array([
				cv2.resize(cv2.imread(file_name), (image_size, image_size))
					for file_name in batch_x]), np.asarray(batch_y)

	empty_dirs = glob.glob(PKLOT_SEGMENTED_DIR + 'UFPR05/*/*/Empty/*')
	occupied_dirs = glob.glob(PKLOT_SEGMENTED_DIR + 'UFPR05/*/*/Occupied/*')
	print('Train: ', len(empty_dirs), len(occupied_dirs))
	#print(empty_dirs, occupied_dirs)
	test_empty_dirs = glob.glob(PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-10-31/Empty/*')
	test_occupied_dirs = glob.glob(PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-10-31/Occupied/*')
	print('Test: ', len(test_empty_dirs), len(test_occupied_dirs))

	train_seq = PlotSequence(empty_dirs, occupied_dirs, batch_size)
	test_seq = PlotSequence(test_empty_dirs, test_occupied_dirs, 400)

	#keras.callbacks.TensorBoard(log_dir='./logs')

	model = Sequential(
		name='img_classifier',
		layers=[
			Conv2D(input_shape=(image_size, image_size,3), use_bias=True,
				filters=16, kernel_size=(5,5), padding='same',
				kernel_initializer='random_uniform', activation='relu', data_format="channels_last"),
			Conv2D(filters=32, kernel_size=(5,5), use_bias=True, padding='same', strides=(2,2), activation='relu', data_format="channels_last"),
			Conv2D(filters=64, kernel_size=(4,4), use_bias=True, padding='same', strides=(2,2), activation='relu', data_format="channels_last"),
			Flatten(),
			Dense(units=1024, activation='relu'),
			Dropout(rate=0.5),
			Dense(units=2, activation='softmax')		
	])

	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
	print(model.summary())
	#keras.callbacks.TensorBoard(log_dir='./log')

	model.fit_generator(train_seq, steps_per_epoch=len(train_seq)/batch_size, epochs=10, verbose=1)

	#score = model.evaluate(x_test, y_test, verbose=0)
	score = model.predict_generator(test_seq, steps=17, verbose=1)
	#for i in range(0,len(score),100):
		#print(i, score[i])
	print(len(score))
	print('Test :', score)
	#print('Test accuracy:', score[1])

		
