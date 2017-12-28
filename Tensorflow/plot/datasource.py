import tensorflow as tf
import glob
import numpy as np
import random
import cv2
import multiprocessing
import itertools
import re
import xmltodict
import json

image_size = 64
SEED = 50129

OCCUPIED = np.array([0.0,1.0])
EMPTY = np.array([1.0, 0.0])

#Retrieves images' paths from directory
def GetImagesPaths(path_pattern):
	pattern = path_pattern + '*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths
	
#Reads of a parking lot, parses the spots and return them with their corresponding labels
def process_image(img_path):
	xml_path = re.sub('jpg$', 'xml', img_path)
	image = cv2.imread(img_path, cv2.IMREAD_COLOR)

	spots = []
	labels = []
	with open(xml_path, 'rb') as xml_file:
		json_data = json.loads(json.dumps(xmltodict.parse(xml_file)))
	spaces = json_data['parking']['space']
	for space in spaces:
		try:
			points = space['contour']['point']
		except(KeyError):
			points = space['contour']['Point'] #Some xmls use caps 'P' for Point

		if '@occupied' in space.keys():
			spot_status = space['@occupied']
			#Extract coords with Lambda function
			coord = lambda i: (int(points[i]['@x']), int(points[i]['@y']))
			botleft, topleft, topright, botright = coord(0), coord(1), coord(2), coord(3)

			#Warp image
			#https://stackoverflow.com/questions/2992264/extracting-a-quadrilateral-image-to-a-rectangle/2992759#2992759
			corners = np.array([botleft, topleft, topright, botright], np.float32)
			target = np.array([(0,0), (0,image_size), (image_size, image_size), (image_size, 0)], np.float32)
			mat = cv2.getPerspectiveTransform(corners, target)
			spot = cv2.warpPerspective(image, mat, (image_size, image_size))

			spots.append(spot)
			if spot_status is '0':
				labels.append(EMPTY)
			else:
				labels.append(OCCUPIED)

	return [np.array(spots), np.array(labels)]

class DataSet(object):
	def __init__(self,
			paths,
			batch_size,
			image_size,
			process_pool,
			name='Give this dataset a f**ing name'):
		self._name = name
		self._batch_size = batch_size
		self._image_size = image_size
		self._process_pool = process_pool
		self._num_images = len(paths)
		self._paths = paths
		#Split paths into chunks of [batch_size] and create a cycle iterator to access them
		self._pBatches = [self._paths[i:i+self._batch_size] for i in range(0, len(self._paths), self._batch_size)]
		self._pBatches_cycle_iterator = itertools.cycle(self._pBatches)

		self._num_images_processed = 0
		self.dump_log()
		
	#Returns the next batch of data
	def next_batch(self):
		#Retrieve next batch of data
		batch_paths = next(self._pBatches_cycle_iterator)
		result = self._process_pool.map(process_image, batch_paths)
		images = [x[0] for x in result]
		images = np.concatenate(np.array(images))
		labels = [x[1] for x in result]
		labels = np.concatenate(np.array(labels))
		self._num_images_processed += self._batch_size #accumulate number of images processed
		return images, labels
	
	#Dump values to the screen
	def dump_log(self):
		print('---------------------------------------------------')
		print('DataSet ->', self._name)
		print('Number of images->', self._num_images)
		print('Batch size->', self._batch_size)
		print('Image size->', self._image_size)
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
		paths = PathSet()
		
		self._num_images = 0
		random.seed(SEED)
		#This routine will ensure that each set (train, valid, test) contains the same proportion of data from each lot
		#E.g given 2 lots, training set will have [train_percent] of lot1 data and [train_percent] of lot2 data
		for path_pattern in lot_path_patterns:
			lot_set = GetImagesPaths(path_pattern)
			self._num_images = self._num_images + len(lot_set)

			#Randomly shuffle dataset to avoid training batches with very similar images(https://www.quora.com/Does-the-order-of-training-data-matter-when-training-neural-networks)
			random.shuffle(lot_set)

			#Data will be split into Testing and Training sets (train_percent is used for training)
			splits = [int(train_percent * len(lot_set)), int((train_percent + validation_percent) * len(lot_set))]

			paths.train.extend(lot_set[:splits[0]])
			paths.validation.extend(lot_set[splits[0]:splits[1]])
			paths.test.extend(lot_set[splits[1]:])

			random.shuffle(paths.train)
			random.shuffle(paths.validation)
			random.shuffle(paths.test)
		
		print('Total Images', self.size)
		#create training, validation and testing sets
		self.train = DataSet(paths.train, train_batch_size, image_size, process_pool=process_pool, name='Training')
		self.validation = DataSet(paths.validation, validation_batch_size, image_size, process_pool=process_pool, name='Validation')
		self.test = DataSet(paths.test, test_batch_size, image_size, process_pool=process_pool, name='Testing')

	@property
	def size(self):
		return self._num_images

	
			