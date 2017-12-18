# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import random
import time
import cv2
import gc
import multiprocessing
import sys

image_size = 28
USING_TF_DATASET = True

#Retrieves images from modifiable directory
def GetImagesPaths(path_pattern, status):
	pattern = path_pattern + status + '/*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths
	
def map_func(file):
	#file = file.decode("utf-8") #convert bytes to string
	img = cv2.imread(file)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(img_gray, (image_size, image_size))
	return resized_image[..., np.newaxis] #adds a new axis(grayscale=1) and returns array of (size, size, 1)


class ImageSet(object):
	def __init__(self,
			paths_to_images,
			images_per_batch,
			image_size,
			sample_label,
			process_pool,
			name='Give this ImageSet a f**ing name',
			load_all=False):
		self._name = name
		self._img_paths = paths_to_images
		self._num_images = len(self._img_paths)
		self._batch_size = images_per_batch
		self._image_size = image_size
		self._label = sample_label
		self._process_pool = process_pool
		
		self._epochs = 0
		self._batch_index = 0
	
	def create_labels(self, size):
		#Generate labels
		return [self._label for i in range(size)]
		
	#Dump values to the screen
	def dump_log(self):
		print('')
		print('\tImageSet ->', self._name)
		print('\tNumber of images->', self._num_images)
		print('\tBatch size->', self._batch_size)
		print('\tImage size->', self._image_size)
		print('')
	
	def next(self):
		startIndex = self._batch_index * self._batch_size
		endIndex = startIndex + self._batch_size
		img_list = self._img_paths[startIndex:endIndex]
		
		if endIndex >= len(self._img_paths):
			self._batch_index = 0
			self._epochs = self._epochs + 1
		else:
			self._batch_index = self._batch_index + 1
			
		image_array = np.asarray(self._process_pool.map(map_func, img_list))
		labels = self.create_labels(len(image_array))
		
		return image_array, labels
		

class DataSet(object):
	def __init__(self,
			empty_spots_paths,
			occupied_spots_paths,
			batch_size,
			image_size,
			process_pool,
			name='Give this dataset a f**ing name'):
		self._name = name
		self._batch_size = batch_size
		self._image_size = image_size
		self._process_pool = process_pool
		self._num_images = len(empty_spots_paths) + len(occupied_spots_paths)
		empty_spots_batch_size = int(batch_size/2)
		occupied_spots_batch_size = batch_size - empty_spots_batch_size
		
		self._empty_spots = ImageSet(empty_spots_paths, empty_spots_batch_size, self._image_size, [1.0, 0.0], self._process_pool, 'Empty spots')
		self._occupied_spots = ImageSet(occupied_spots_paths, occupied_spots_batch_size, self._image_size, [0.0, 1.0], self._process_pool, 'Occupied spots')
		self._num_images_processed = 0
		
		self.dump_log()
		
	#Returns the next batch of data
	def next_batch(self):
		#Retrieve next batch of data 
		empty_spots_images, empty_spots_labels = self._empty_spots.next()
		occupied_spots_images, occupied_spots_labels = self._occupied_spots.next()
		#concatenate empty and occupied images and labels
		image_batch = np.concatenate([empty_spots_images, occupied_spots_images])
		label_batch = empty_spots_labels + occupied_spots_labels
		self._num_images_processed += self._batch_size #accumulate number of images processed
		return image_batch, label_batch
	
	#Dump values to the screen
	def dump_log(self):
		print('---------------------------------------------------')
		print('DataSet ->', self._name)
		print('Number of images->', self._num_images)
		print('Batch size->', self._batch_size)
		print('Image size->', self._image_size)
		self._empty_spots.dump_log()
		self._occupied_spots.dump_log()
		print('---------------------------------------------------')
		
	def batch_rem(self):
		return self._num_images_processed < self._num_images
	
class DataSets(object):		
	def __init__(self,
			path_pattern,
			train_percent,
			batch_size,
			process_pool):
		#data load routine
		print('Data stats. Training: ' ,train_percent, '. Test: ', 1-train_percent,'\n')
		empty_spots_paths = GetImagesPaths(path_pattern, 'Empty')
		occupied_spots_paths = GetImagesPaths(path_pattern, 'Occupied')

		#Data will be split into Testing and Training sets (train_percent is used for training)
		empty_spots_paths_split = int(train_percent * len(empty_spots_paths))
		occupied_spots_paths_split = int(train_percent * len(occupied_spots_paths))

		#create training and testing sets
		self.train = DataSet(empty_spots_paths[:empty_spots_paths_split], occupied_spots_paths[:occupied_spots_paths_split], batch_size, image_size, 
			process_pool=process_pool, name='Training')
		self.test = DataSet(empty_spots_paths[empty_spots_paths_split:], occupied_spots_paths[occupied_spots_paths_split:], batch_size=8000, 
			image_size=image_size, process_pool=process_pool, name='Testing')
			