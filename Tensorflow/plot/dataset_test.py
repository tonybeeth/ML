import tensorflow as tf
import numpy as np
import time
import multiprocessing
import dataset_parallel_input as datasource

train_percent = 0.5
batch_size = 100

#directories containing images
PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

data_path_pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/'

if __name__ == "__main__":
	process_pool = multiprocessing.Pool(8)

	dataset = datasource.DataSets(data_path_pattern, train_percent, batch_size, process_pool)

	startTime = time.time()
	# Start a new session to show example output.
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		
		#tf.initialize_all_variables().run()
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		#print('Nidaime')
		for i in range(50):
			a, b = dataset.train.next_batch()
			#print(a.shape)
			#print(type(a), len(b))
		
		print("\nTime taken: %f" %(time.time() - startTime))