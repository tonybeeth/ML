# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import random
import time
import cv2

image_size = 28

#directories containing images
PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

#print(len(glob.glob(PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/' + 'Empty' + '/*#09*.jpg')))
def gen(files):
	for file in files:
		print(file)
		img = cv2.imread(file)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		resized_image = cv2.resize(img_gray, (1024, 1024))
		yield img_gray

def gen_images(files):
	for file in files:
		file = file.decode("utf-8")
		print(file)
		img = cv2.imread(file)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		resized_image = cv2.resize(img_gray, (1024, 1024))
		yield img_gray

def map_func(tens):
	file=tens.eval()
	print(file)
	img = cv2.imread(file)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(img_gray, (28, 28))
	return img_gray
	
class Test(object):
	def __init__(self):
		self.files = glob.glob(PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-11-11/' + 'Occupied' + '/*.jpg')
		print(len(self.files))
		self.file_dataset = tf.data.Dataset.list_files(self.files)
		self.file_batches = self.file_dataset.batch(5)
	
	def test(self):
		it = self.file_batches.make_one_shot_iterator()
		next_element = it.get_next()
		img_list = next_element.eval()
		#images = file_dataset.map(map_func)
		images = tf.data.Dataset.from_generator(lambda: gen_images(img_list), output_types=tf.float32)
		it = images.make_one_shot_iterator()
		print(it.get_next().eval())


obj = Test()		
# Start a new session to show example output.
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
	
	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()

	obj.test()
	# while True:
		# try:
			# img_list = sess.run(next_element)
			# images = tf.data.Dataset.from_generator(lambda: gen_images(img_list), output_types=tf.float32)
		# except tf.errors.OutOfRangeError:
			# break
	#image_paths = tf.data.Dataset.from_generator(lambda: gen(files), tf.float32)
	##batches = image_paths.batch(10)
	#it = image_paths.make_one_shot_iterator()
	#next_element = it.get_next()
	#print(next_element[0].eval())
	
	#a = gen()
	#print(type(next(a)))
	
	#num_parallel_calls = 1 #number of images to process in parallel
	#paths_batch = image_paths.batch(10)
	#print(paths_batch)
	#batch_it = paths_batch.make_one_shot_iterator()
	#a = batch_it.get_next()
	#print(a)
	#a = image_paths.map(process_image)
	#image_batch = a.map(process_image) #process images
	#Convert dataset of images to numpy array
	#iterator = image_batch.make_one_shot_iterator()
	#next_image = iterator.get_next()
	#a = process_image(next_image.eval())
	#print(a.eval())
	
	