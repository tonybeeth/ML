import tensorflow as tf
import numpy as np
import time
import multiprocessing
import datasource
import os

train_percent = 0.5
train_batch_size = 256
test_batch_size = 3000

PKLOT_SEGMENTED_DIR =  os.environ['PKLOT_DATA'] + '/PKLot/PKLotSegmented/'
data_path_pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/'

if __name__ == "__main__":
	process_pool = multiprocessing.Pool(8)
	dataset = datasource.DataSource(data_path_pattern, train_percent, train_batch_size, test_batch_size, process_pool)

	startTime = time.time()
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		
		#tf.initialize_all_variables().run()
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		
		for i in range(1):
			a, b = dataset.train.next_batch()
			print(a.nbytes)
			#print(type(a), len(b))
		
		print("\nTime taken: %f" %(time.time() - startTime))