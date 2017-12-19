import tensorflow as tf
import numpy as np
import os
import time
import multiprocessing
import datasource

def initial_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def initial_biases(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))
def conv2d(images, weights, strideDim):
	return tf.nn.conv2d(images, weights, strides=strideDim, padding='SAME')

if __name__ == "__main__":

	if os.environ.get('PKLOT_DATA') is None:
		print('Cannot locate PKLOT_DATA in environment')
		exit()

	BATCH_NORM = True
	EPOCHS = 10
	train_percent = 0.7 #percentage of data to train
	train_batch_size = 256 #num images used in an ideal training batch
	test_batch_size = 4000 #num images used in an ideal testing batch

	#directories containing images
	PKLOT_SEGMENTED_DIR = os.environ.get('PKLOT_DATA') + '/PKLot/PKLotSegmented/'
	#data_path_pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/2012-09-12/'
	data_path_pattern = PKLOT_SEGMENTED_DIR + '*/*/*/'

	process_pool = multiprocessing.Pool(8)
	dataset = datasource.DataSource(data_path_pattern, train_percent, train_batch_size, test_batch_size, process_pool)
	TRAIN_BATCHES_PER_EPOCH = dataset.train.size/train_batch_size

	with tf.device('/device:GPU:0'):
		#placeholders for images and correct labels
		images = tf.placeholder(tf.float32, shape=[None, datasource.image_size, datasource.image_size, 3], name='images')
		correct_labels = tf.placeholder(tf.float32, shape=[None, 2], name='correct_labels')
		training = tf.placeholder(tf.bool)
		
		#First convolutional layer
		w1 = initial_weights([5, 5, 3, 16])
		b1 = initial_biases([16])
		actv_map1 = tf.nn.relu(conv2d(images, w1, strideDim=[1, 1, 1, 1]) + b1)
		if BATCH_NORM is True:
			actv_map1 = tf.layers.batch_normalization(actv_map1, axis=1, training=training)

		#Second convolutional layer
		w2 = initial_weights([5, 5, 16, 32])
		b2 = initial_biases([32])
		actv_map2 = tf.nn.relu(conv2d(actv_map1, w2, strideDim=[1, 2, 2, 1]) + b2)
		if BATCH_NORM is True:
			actv_map2 = tf.layers.batch_normalization(actv_map2, axis=1, training=training)
		
		#Third convolutional layer
		w3 = initial_weights([4, 4, 32, 64])
		b3 = initial_biases([64])
		actv_map3 = tf.nn.relu(conv2d(actv_map2, w3, strideDim=[1, 2, 2, 1]) + b3)
		if BATCH_NORM is True:
			actv_map3 = tf.layers.batch_normalization(actv_map3, axis=1, training=training)
		
		flatten_size = int(((datasource.image_size*datasource.image_size)/16)*64)
		#Fully connected layer 1
		w_fc1 = initial_weights([flatten_size, 1024])
		b_fc1 = initial_biases([1024])
		#flatten activation map from conv 3
		actv_map3_flat = tf.reshape(actv_map3, [-1, flatten_size])
		fc1_output = tf.nn.relu(tf.matmul(actv_map3_flat, w_fc1) + b_fc1)
		if BATCH_NORM is True:
			fc1_output = tf.layers.batch_normalization(fc1_output, training=training)
		
		#Apply dropout to reduce overfitting before readout layer
		dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
		#Uncomment this if dropout has to be applied
		#fc1_output = tf.nn.dropout(fc1_output, dropout_prob)

		#Readout layer
		w_fc2 = initial_weights([1024, 2])
		b_fc2 = initial_biases([2])
		predicted_labels = tf.add(tf.matmul(fc1_output, w_fc2), b_fc2, name='predicted')

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_labels, logits=predicted_labels))
		#train using more sophisticated adam optimizer
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		correct_pred = tf.equal(tf.argmax(predicted_labels, 1), tf.argmax(correct_labels, 1), name='correct_pred')
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

		#Extra operations with Batch normalization: https://stackoverflow.com/a/43285333
		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		
	startTime = time.time()
	
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		writer = tf.summary.FileWriter("log/", sess.graph)
		#saver = tf.train.Saver()
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()
		
		trainTime = time.time()
		print("EPOCHS: ", EPOCHS)
		print("TRAIN_BATCHES_PER_EPOCH: ", int(TRAIN_BATCHES_PER_EPOCH), '\n')
		for i in range(EPOCHS):
			print("Epoch", i)
			for j in range(int(TRAIN_BATCHES_PER_EPOCH)):
				image_data, labels = dataset.train.next_batch()
				if j % 10 == 0:
					feed_dict = {images: image_data, correct_labels: labels, dropout_prob: 1.0, training: False}
					train_accuracy = accuracy.eval(feed_dict=feed_dict)
					print('\tBatch %d, training accuracy %g' % (j, train_accuracy))
					
				sess.run([train_step, extra_update_ops], feed_dict={images: image_data, correct_labels: labels, dropout_prob: 0.5, training: True})		
		
		testTime = time.time()		
		accuracies = []
		while dataset.test.batch_rem() is True:
			image_data, labels = dataset.test.next_batch()
			feed_dict = {images: image_data, correct_labels: labels, dropout_prob: 1.0, training: False}
			accuracies.append(accuracy.eval(feed_dict=feed_dict))
		
		print('\nMetrics:')
		print("Load Time: %fs" %(trainTime - startTime))
		print("Train Time: %fs" %(testTime - trainTime))
		print("Test Time: %fs" %(time.time() - testTime))
		print("Total taken: %fs" %(time.time() - startTime))

		print('\nTest accuracy %g' % np.mean(np.asarray(accuracies)))
				
		process_pool.close()
		process_pool.join()
		writer.close()
		
		#saver.save(sess, './model')
		