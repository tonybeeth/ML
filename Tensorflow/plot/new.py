# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob
import numpy as np
import time
import multiprocessing
import dataset_parallel_input as datasource

def initial_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def initial_biases(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))
def conv2d(images, weights, strideDim):
	return tf.nn.conv2d(images, weights, strides=strideDim, padding='SAME')

if __name__ == "__main__":
	train_percent = 0.8
	batch_size = 100

	#directories containing images
	PKLOT_DIR = '../../../PKLot/PKLot/'
	PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

	#data_path_pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/'
	data_path_pattern = PKLOT_SEGMENTED_DIR + '*/*/*/'
	process_pool = multiprocessing.Pool(8)
	dataset = datasource.DataSets(data_path_pattern, train_percent, batch_size, process_pool)
	
	with tf.device('/device:GPU:0'):
		#placeholders for images and correct labels
		images = tf.placeholder(tf.float32, shape=[None, datasource.image_size, datasource.image_size, 1], name='images')
		correct_labels = tf.placeholder(tf.float32, shape=[None, 2], name='correct_labels')

		#First convolutional layer
		w1 = initial_weights([5, 5, 1, 16])
		b1 = initial_biases([16])
		actv_map1 = tf.nn.relu(conv2d(images, w1, strideDim=[1, 1, 1, 1]) + b1)

		#Second convolutional layer
		w2 = initial_weights([5, 5, 16, 32])
		b2 = initial_biases([32])
		actv_map2 = tf.nn.relu(conv2d(actv_map1, w2, strideDim=[1, 2, 2, 1]) + b2)

		#Third convolutional layer
		w3 = initial_weights([4, 4, 32, 64])
		b3 = initial_biases([64])
		actv_map3 = tf.nn.relu(conv2d(actv_map2, w3, strideDim=[1, 2, 2, 1]) + b3)

		#Fully connected layer 1
		w_fc1 = initial_weights([7*7*64, 1024])
		b_fc1 = initial_biases([1024])
		#flatten activation map from conv 3
		actv_map3_flat = tf.reshape(actv_map3, [-1, 7*7*64])
		fc1_output = tf.nn.relu(tf.matmul(actv_map3_flat, w_fc1) + b_fc1)

		#Apply dropout to reduce overfitting before readout layer
		dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
		dropout_output = tf.nn.dropout(fc1_output, dropout_prob)

		#Readout layer
		w_fc2 = initial_weights([1024, 2])
		b_fc2 = initial_biases([2])
		predicted_labels = tf.add(tf.matmul(dropout_output, w_fc2), b_fc2, name='predicted')

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_labels, logits=predicted_labels))
		#train using more sophisticated adam optimizer
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		correct_pred = tf.equal(tf.argmax(predicted_labels, 1), tf.argmax(correct_labels, 1), name='correct_pred')
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	startTime = time.time()
	# Start a new session to show example output.
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		writer = tf.summary.FileWriter("dlog/", sess.graph)
		#saver = tf.train.Saver()
		# Required to get the filename matching to run.
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		trainTime = time.time()
		#batch_get_times = []
		#train_step_run_times = []
		for i in range(10000):
			
			#beforeBatch = time.time()
			image_data, labels = dataset.train.next_batch()
			#batch_get_times.append(time.time()-beforeBatch)
			#Print accuracy after 10 iterations
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={images: image_data, correct_labels: labels, dropout_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			#train_step_time = time.time()
			train_step.run(feed_dict={images: image_data, correct_labels: labels, dropout_prob: 0.5})
			#train_step_run_times.append(time.time() - train_step_time)
			
		
		testTime = time.time()
		print('\nMetrics:')
		print("Load Time: %f" %(trainTime - startTime))
		print("Train Time: %f" %(testTime - trainTime))
		#print('Average next_batch() time:', sum(batch_get_times)/float(len(batch_get_times)))
		#print('Max next_batch() time:', max(batch_get_times))
		#print('Min next_batch() time:', min(batch_get_times))
		#print('Average train_step.run() time:', sum(train_step_run_times)/float(len(train_step_run_times)))
		#print('Max train_step.run() time:', max(train_step_run_times))
		#print('Min train_step.run() time:', min(train_step_run_times))
		
		while dataset.test.batch_rem() is True:
			image_data, labels = dataset.test.next_batch()
			print('%d, %d' % (len(image_data), len(labels)))
			print('\nTest accuracy %g' % accuracy.eval(feed_dict={images: image_data, correct_labels: labels, dropout_prob: 1.0}))
		
		print("Test Time: %f" %(time.time() - testTime))
		
		
		print("\nTime taken: %f" %(time.time() - startTime))
		
		writer.close()
		
		#saver.save(sess, './model')
		