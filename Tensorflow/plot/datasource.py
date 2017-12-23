import tensorflow as tf
import glob
import numpy as np
import random
import cv2
import multiprocessing

image_size = 64
SEED = 50129

#Retrieves images' paths from directory
def GetImagesPaths(path_pattern, status):
	pattern = path_pattern + status + '/*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths
	
#Reads an image with RGB colors, resizes to image_sizeximage_size and returns it
def process_image(file):
	img = cv2.imread(file)
	img = cv2.resize(img, (image_size, image_size))
	return img

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
			
		image_array = np.asarray(self._process_pool.map(process_image, img_list))
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
		
	#Returns false after the last batch and starts over
	def batch_rem(self):
		return_val = self._num_images_processed < self._num_images
		if self._num_images_processed >= self._num_images:
			self._num_images_processed = 0
		return return_val
			
	@property
	def size(self):
		return self._num_images
	
class DataSource(object):
	def __init__(self,
			lot_path_patterns,
			train_percent,
			validation_percent,
			train_batch_size,
			validation_batch_size,
			test_batch_size,
			process_pool):

		test_percent = 1 - (train_percent + validation_percent)
		#data load routine
		print('Data stats. Training' ,train_percent, ', Validation', validation_percent, ', Testing', test_percent,'\n')
		print('Random SEED: ', SEED)
		if test_percent < 0:
			print('Cannot have a negative test percentage ', test_percent)
			exit()

		class PathSet(object):
			def __init__(self):
				self.train = []
				self.validation = []
				self.test = []
		empty_paths, occupied_paths = PathSet(), PathSet()
		
		self._num_images = 0
		random.seed(SEED)
		#This routine will ensure that each set (train, valid, test) contains the same proportion of data from each lot
		#E.g given 2 lots, training set will have [train_percent] of lot1 data and [train_percent] of lot2 data
		for path_pattern in lot_path_patterns:
			empty_set = GetImagesPaths(path_pattern, 'Empty')
			occupied_set = GetImagesPaths(path_pattern, 'Occupied')
			self._num_images = self._num_images + len(empty_set) + len(occupied_set)

			#Randomly shuffle dataset to avoid training batches with very similar images(https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks)
			random.shuffle(empty_set)
			random.shuffle(occupied_set)

			#Data will be split into Testing and Training sets (train_percent is used for training)
			empty_set_splits = [int(train_percent * len(empty_set)), int((train_percent + validation_percent) * len(empty_set))]
			occupied_set_splits = [int(train_percent * len(occupied_set)), int((train_percent + validation_percent) * len(occupied_set))]

			for paths, new_set, splits in [(empty_paths, empty_set, empty_set_splits), (occupied_paths, occupied_set, occupied_set_splits)]:
				paths.train.extend(new_set[:splits[0]])
				paths.validation.extend(new_set[splits[0]:splits[1]])
				paths.test.extend(new_set[splits[1]:])

		for paths in [empty_paths, occupied_paths]:
			random.shuffle(paths.train)
			random.shuffle(paths.validation)
			random.shuffle(paths.test)
		
		print('Total Images', self.size)
		#create training, validation and testing sets
		self.train = DataSet(empty_paths.train, occupied_paths.train, train_batch_size, image_size, process_pool=process_pool, name='Training')
		self.validation = DataSet(empty_paths.validation, occupied_paths.validation, validation_batch_size, image_size, process_pool=process_pool, name='Validation')
		self.test = DataSet(empty_paths.test, occupied_paths.test, test_batch_size, image_size, process_pool=process_pool, name='Testing')

	@property
	def size(self):
		return self._num_images

	
			