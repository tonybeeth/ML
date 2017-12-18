# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import random
import time

USING_FIFO_Q_DATASET = True

train_percent = 0.5
image_size = 28

#directories containing images
PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

#Retrieves images from modifiable directory
def GetImagesPaths(status):
	pattern = PKLOT_SEGMENTED_DIR + '*/*/*/' + status + '/*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths
	
class DataSet(object):

	#Constructor: does most of the set up
	def __init__(self,
		   empty_spots_paths,
		   occupied_spots_paths,
		   batch_size = 50,
		   load_all=False,
		   output_log=False,
		   name='Give me a f***ing name'):
		self._name = name
		self._num_empty_spots = len(empty_spots_paths)
		self._num_occupied_spots = len(occupied_spots_paths)
		self._batch_size = batch_size
		self._image_size = image_size
		self._initial_spot_size = tf.constant([self._image_size, self._image_size]) #working with square images
		self._capacity_empty_spots_queue = self._capacity_occupied_spots_queue = 32 #default capacity
		self._empty_spots_batch_size = self._occupied_spots_batch_size = batch_size
		self._num_batches_processed=0
		
		#If all data need to be loaded at once
		if load_all is True:
			#set batch sizes to the maximum possible
			self._empty_spots_batch_size = self._capacity_empty_spots_queue = self._num_empty_spots
			self._occupied_spots_batch_size = self._capacity_occupied_spots_queue = self._num_occupied_spots
		
		#Generate labels [empty, occupied]
		self._empty_labels = [[1.0, 0.0] for i in range(self._empty_spots_batch_size)]
		self._occupied_labels = [[0.0, 1.0] for i in range(self._occupied_spots_batch_size)]
		
		#Create queues from files' paths
		self._empty_spots_file_names_queue = tf.train.string_input_producer(tf.train.match_filenames_once(empty_spots_paths))
		self._occupied_spots_file_names_queue = tf.train.string_input_producer(tf.train.match_filenames_once(occupied_spots_paths))
		
		#define reader and tensors to read image files from queues
		self._image_reader = tf.WholeFileReader()
		self._empty_image_name, self._empty_image_file = self._image_reader.read(self._empty_spots_file_names_queue)
		self._occupied_image_name, self._occupied_image_file = self._image_reader.read(self._occupied_spots_file_names_queue)
		
		#define batches that extract image names and files from the queues
		self._empty_spots_batch = self.input_pipeline2([self._empty_image_name, self.normalize(self._empty_image_file)], self._empty_spots_batch_size)
		self._occupied_spots_batch = self.input_pipeline2([self._occupied_image_name, self.normalize(self._occupied_image_file)], self._occupied_spots_batch_size)
		
		if output_log is True:
			self.dump_log()

	#Generates batch_size number of params data from its queue
	def input_pipeline(self, params, batch_size, num_epochs=None):
		# Generate batch
		num_preprocess_threads = 16
		min_after_dequeue = 50
		seed = 183
		batch = tf.train.batch(
			params,
			batch_size=batch_size,
			allow_smaller_final_batch=True,
			num_threads=num_preprocess_threads,
			capacity=min_after_dequeue + 3 * batch_size,
			)
		return batch
		
	
	#Generates batch_size number of params data from its queue
	def input_pipeline2(self, params, batch_size, num_epochs=None):
		# Generate batch
		num_preprocess_threads = 8
		min_after_dequeue = 50
		seed = 183
		batch = tf.train.shuffle_batch(
			params,
			batch_size=batch_size,
			allow_smaller_final_batch=True,
			num_threads=num_preprocess_threads,
			min_after_dequeue=min_after_dequeue,
			capacity=min_after_dequeue+num_preprocess_threads*batch_size
			)
		return batch
		
	#decode image as jpeg, apply grayscale and resize
	def normalize(self, image_file):
		# Decode the image as a JPEG file, this will turn it into a Tensor which we can
		# then use in training.
		image = tf.image.decode_jpeg(image_file)
		grayImg = tf.image.rgb_to_grayscale(image)
		return tf.image.resize_images(grayImg, self._initial_spot_size)
		
	#Returns the next batch of data
	def next_batch(self):
		#Retrieve next batch of data 
		#Note: this only evaluates and returns the image data, not the file names
		empty_spots_images = self._empty_spots_batch[1].eval()
		occupied_spots_images = self._occupied_spots_batch[1].eval()
		
		spots_images = np.concatenate([empty_spots_images, occupied_spots_images]) #join empty and occupied spots data
		labels = self._empty_labels + self._occupied_labels #join empty and occupied spots labels
		self._num_batches_processed+=self._batch_size
		return spots_images, labels
	
	#Dump values to the screen
	def dump_log(self):
		print('')
		print('DataSet -> ', self._name)
		print('Number of spots. Empty->', self._num_empty_spots, '. Occupied->', self._num_occupied_spots)
		print('Batch size->', self._batch_size)
		print('Image size->', self._image_size)
		print('Queue capacities. Empty->', self._capacity_empty_spots_queue, '. Occupied->', self._capacity_occupied_spots_queue)
		print('Separate batch sizes. Empty->', self._empty_spots_batch_size, '. Occupied->', self._occupied_spots_batch_size)
		print('')
		
	def batch_rem(self):
		if self._num_batches_processed < (self._num_empty_spots+self._num_occupied_spots):
			return True
		else:
			return False
		

#data load routine

print('Data stats. Training: ' ,train_percent, '. Test: ', 1-train_percent,'\n')

empty_spots_paths = GetImagesPaths('Empty')
occupied_spots_paths = GetImagesPaths('Occupied')

random.shuffle(empty_spots_paths)
random.shuffle(occupied_spots_paths)

#Data will be split into Testing and Training sets (train_percent is used for training)
empty_spots_paths_split = int(train_percent * len(empty_spots_paths))
occupied_spots_paths_split = int(train_percent * len(occupied_spots_paths))

#create training and testing sets
train = DataSet(empty_spots_paths[:empty_spots_paths_split], occupied_spots_paths[:occupied_spots_paths_split], output_log=True, name='Train')
test = DataSet(empty_spots_paths[empty_spots_paths_split:], occupied_spots_paths[occupied_spots_paths_split:], output_log=True, name='Test', batch_size=8000)