import tensorflow as tf
import numpy as np
import time
import multiprocessing
import datasource
import os
from pprint import pprint

train_percent = 0.5
validation_percent = 0.3
train_batch_size = 10
validation_batch_size = train_batch_size
test_batch_size = 10

PKLOT_SEGMENTED_DIR =  os.environ['PKLOT_DATA'] + '/PKLot/PKLot/'
data_path_pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/'

if __name__ == "__main__":
	process_pool = multiprocessing.Pool(5)
	startTime = time.time()
	dataset = datasource.DataSource([data_path_pattern], train_percent, validation_percent, train_batch_size, validation_batch_size, test_batch_size, process_pool)
	print('Load time:', time.time()-startTime)	
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		#tf.initialize_all_variables().run()
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		startTime = time.time()
		for i in range(2):
			a, b = dataset.train.next_batch()
			#print(a.shape, b.shape)
			#print(type(a[0][0]), type(b[0]))
			
		
		print("\nnext batch time: %f" %(time.time() - startTime))