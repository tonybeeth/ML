# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import random
import time
import cv2
import gc

USING_TF_DATASET = True

train_percent = 0.5
image_size = 28
batch_size = 100

#directories containing images
PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

#Retrieves images from modifiable directory
def GetImagesPaths(status):
	pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/' + status + '/*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths

class ImageSet(object):
	
	def __init__(self,
			paths_to_images,
			images_per_batch,
			image_size,
			sample_label,
			name='Give this ImageSet a f**ing name',
			load_all=False):
		self._name = name
		self._num_images = len(paths_to_images)
		self._batch_size = images_per_batch
		self._image_size = image_size
		self._num_images_processed = 0
		self._label = sample_label
		
		#Create TF datasets from files' paths
		self._image_paths = tf.data.Dataset.list_files(paths_to_images)
		self._image_paths.shuffle(buffer_size=10000) #shuffles dataset initially
		
		self._image_path_batches = self._image_paths.batch(self._batch_size) #split data into batches
		self._image_path_batches.repeat()#batches should repeat infinitely so we can get batches on variable epochs
		
		self._image_paths_it = self._image_paths.make_one_shot_iterator()
		self._next_image_path = self._image_paths_it.get_next()
		#self._image_dataset = self._image_paths.from_generator(self.img_generator, output_types=tf.float32)
		#self._next_image_batch = _image_dataset#self._image_dataset.prefetch(self._batch_size)
		
		#iterator for batches
		self._image_path_batches_it = self._image_path_batches.make_one_shot_iterator()
		self._next_image_paths_batch = self._image_path_batches_it.get_next()
	
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
		
	def gen_images(self, files):
		for file in files:
			file = file.decode("utf-8") #convert bytes to string
			img = cv2.imread(file)
			img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			resized_image = cv2.resize(img_gray, (self._image_size, self._image_size))
			yield resized_image[..., np.newaxis] #adds a new axis(grayscale=1) and returns array of (size, size, 1)
			
	def next(self):
		try:
			img_list = self._next_image_paths_batch.eval()
		except tf.errors.OutOfRangeError:
			self._image_path_batches_it = self._image_path_batches.make_one_shot_iterator()
			self._next_image_paths_batch = self._image_path_batches_it.get_next()
			img_list = self._next_image_paths_batch.eval()
			
		image_dataset = tf.data.Dataset.from_generator(lambda: self.gen_images(img_list), output_types=tf.float32)
		next_image = image_dataset.make_one_shot_iterator()
		image_array = [next_image.get_next().eval()]
		while True:
			try:
				img = next_image.get_next()
				image_array = np.vstack((image_array, [img.eval()]))
			except tf.errors.OutOfRangeError:
				break
		labels = self.create_labels(len(image_array))
		return image_array, labels

class DataSet(object):

	def __init__(self,
			empty_spots_paths,
			occupied_spots_paths,
			batch_size,
			image_size,
			name='Give this dataset a f**ing name'):
		self._name = name
		self._batch_size = batch_size
		self._image_size = image_size
		self._num_images = len(empty_spots_paths) + len(occupied_spots_paths)
		empty_spots_batch_size = int(batch_size/2)
		occupied_spots_batch_size = batch_size - empty_spots_batch_size
		
		self._empty_spots = ImageSet(empty_spots_paths, empty_spots_batch_size, self._image_size, [1.0, 0.0], 'Empty spots')
		self._occupied_spots = ImageSet(occupied_spots_paths, occupied_spots_batch_size, self._image_size, [0.0, 1.0], 'Occupied spots')
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
		

		
#data load routine
print('Data stats. Training: ' ,train_percent, '. Test: ', 1-train_percent,'\n')

empty_spots_paths = GetImagesPaths('Empty')
occupied_spots_paths = GetImagesPaths('Occupied')

#Data will be split into Testing and Training sets (train_percent is used for training)
empty_spots_paths_split = int(train_percent * len(empty_spots_paths))
occupied_spots_paths_split = int(train_percent * len(occupied_spots_paths))

#train = DataSet(empty_spots_paths[:empty_spots_paths_split], occupied_spots_paths[:occupied_spots_paths_split], batch_size, image_size, name='Training')
#test = DataSet(empty_spots_paths[empty_spots_paths_split:], occupied_spots_paths[occupied_spots_paths_split:], batch_size, image_size, name='Testing')

startTime = time.time()
# Start a new session to show example output.
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

	#create training and testing sets
	train = DataSet(empty_spots_paths[:empty_spots_paths_split], occupied_spots_paths[:occupied_spots_paths_split], batch_size, image_size, name='Training')
	test = DataSet(empty_spots_paths[empty_spots_paths_split:], occupied_spots_paths[occupied_spots_paths_split:], batch_size=8000, 
		image_size=image_size, name='Testing')
	
	#tf.initialize_all_variables().run()
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()
	print('Nidaime')
	for i in range(10):
		a, b = train.next_batch()
		print(a.shape)
		#print(type(a), len(b))
	
	print("\nTime taken: %f" %(time.time() - startTime))
	