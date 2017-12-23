import tensorflow as tf
import numpy as np
import os
import time
import multiprocessing
import datasource
import math

def initial_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def initial_biases(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

#Creates operations in the graph to allow the reset of variables used to compute a metric
def create_reset_metric(metric, scope='reset_metric', *metric_args):
	with tf.variable_scope(scope):
		metric_op, update_op = metric(*metric_args)
		vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
		reset_op = tf.variables_initializer(vars)
		return metric_op, update_op, reset_op

#Helper method to create update and reset ops for streaming mean metric
def create_streaming_mean_reset_metric(scope='streaming_mean_reset_metric', *metric_args):
	return create_reset_metric(tf.contrib.metrics.streaming_mean, scope, *metric_args)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def conv2d_layer(inputs, weightDim, biasDim, strideDim, use_batch_norm, is_training, layer_name):
	with tf.name_scope(layer_name) as scope:
		with tf.name_scope('weights') as scope:
			weights = initial_weights(weightDim)
			variable_summaries(weights)
		with tf.name_scope('biases') as scope:
			biases = initial_biases(biasDim)
			variable_summaries(biases)
		actv_map = tf.nn.relu(tf.nn.conv2d(inputs, weights, strides=strideDim, padding='SAME') + biases)
		tf.summary.histogram('activations', actv_map)
		if use_batch_norm is True:
			actv_map = tf.layers.batch_normalization(actv_map, axis=1, training=is_training)
			tf.summary.histogram('batch_norm', actv_map)
		return actv_map

if __name__ == "__main__":
	if os.environ.get('PKLOT_DATA') is None:
		print('Cannot locate PKLOT_DATA in environment')
		exit()

	BATCH_NORM = True
	EPOCHS = 10
	train_percent = 0.7 #percentage of data to train
	validation_percent = 0.15
	train_batch_size = 256 #num images used in an ideal training batch
	validation_batch_size = 3000
	test_batch_size = 3000 #num images used in an ideal testing batch

	#directories containing images
	PKLOT_SEGMENTED_DIR = os.environ.get('PKLOT_DATA') + '/PKLot/PKLotSegmented/'
	lot_names = ['PUC', 'UFPR04', 'UFPR05']
	lot_path_patterns = [PKLOT_SEGMENTED_DIR + name + '/*/*/' for name in lot_names]
	#lot_path_patterns = [PKLOT_SEGMENTED_DIR + 'PUCPR/Sunny/*/']

	process_pool = multiprocessing.Pool(8)
	dataset = datasource.DataSource(lot_path_patterns, train_percent, validation_percent, train_batch_size, validation_batch_size, test_batch_size, process_pool)
	TRAIN_BATCHES_PER_EPOCH = math.ceil(dataset.train.size/train_batch_size)
	VALIDATION_BATCHES_PER_EPOCH = math.ceil(dataset.validation.size/validation_batch_size)
	TEST_BATCHES_PER_EPOCH = math.ceil(dataset.test.size/test_batch_size)

	with tf.device('/device:GPU:0'):
		#placeholders for images and correct labels
		with tf.name_scope('input'):
			images = tf.placeholder(tf.float32, shape=[None, datasource.image_size, datasource.image_size, 3], name='images')
			correct_labels = tf.placeholder(tf.float32, shape=[None, 2], name='correct_labels')
			training = tf.placeholder(tf.bool)

		#First convolutional layer
		actv_map1 = conv2d_layer(images, [5, 5, 3, 16], [16], [1, 1, 1, 1], BATCH_NORM, training, 'CNN1')

		#Second convolutional layer
		actv_map2 = conv2d_layer(actv_map1, [5, 5, 16, 32], [32], [1, 2, 2, 1], BATCH_NORM, training, 'CNN2')

		#Third convolutional layer
		actv_map3 = conv2d_layer(actv_map2, [4, 4, 32, 64], [64], [1, 2, 2, 1], BATCH_NORM, training, 'CNN3')

		flatten_size = int(((datasource.image_size*datasource.image_size)/16)*64)
		#Fully connected layer 1
		with tf.name_scope('FC1') as scope:
			w_fc1 = initial_weights([flatten_size, 1024])
			b_fc1 = initial_biases([1024])
			#flatten activation map from conv 3
			actv_map3_flat = tf.reshape(actv_map3, [-1, flatten_size])
			fc1_output = tf.nn.relu(tf.matmul(actv_map3_flat, w_fc1) + b_fc1)
			if BATCH_NORM is True:
				fc1_output = tf.layers.batch_normalization(fc1_output, training=training)

		#Apply dropout to reduce overfitting before readout layer (Not needed when using batch normalization)
		with tf.name_scope('dropout'):
			dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
			if BATCH_NORM is not True: #No need for dropout if batch normalization is used
				fc1_output = tf.nn.dropout(fc1_output, dropout_prob)

		#Readout layer
		with tf.name_scope('Readout') as scope:
			w_fc2 = initial_weights([1024, 2])
			b_fc2 = initial_biases([2])
			predicted_labels = tf.add(tf.matmul(fc1_output, w_fc2), b_fc2, name='predicted')

		with tf.name_scope('cross_entropy_total'):
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_labels, logits=predicted_labels))
		#train using more sophisticated adam optimizer
		with tf.name_scope('train'):
			#Extra operations with Batch normalization: https://stackoverflow.com/a/43285333
			extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(extra_update_ops): #Add update ops as dependency of the train step
				train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		with tf.name_scope('accuracy'):
			correct_pred = tf.equal(tf.argmax(predicted_labels, 1), tf.argmax(correct_labels, 1), name='correct_pred')
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

		#Track accuracy average
		accuracy_streaming_mean, accuracy_streaming_mean_update, accuracy_streaming_mean_reset = create_streaming_mean_reset_metric('accuracy_streaming_mean', accuracy)
		tf.summary.scalar('accuracy_streaming_mean', accuracy_streaming_mean)

		#Merge all summaries
		merged_summary = tf.summary.merge_all()

	#Prevent Tensorflow from allocating all memory on GPU
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True

	startTime = time.time()
	with tf.Session(config=config) as sess:
		log_dir = 'log/'
		train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
		validation_writer = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
		writer = tf.summary.FileWriter(log_dir, sess.graph)
		saver = tf.train.Saver()
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()

		trainTime = time.time()
		print("EPOCHS: ", EPOCHS)
		print("TRAIN_BATCHES_PER_EPOCH: ", TRAIN_BATCHES_PER_EPOCH, '\n')
		print("VALIDATION_BATCHES_PER_EPOCH: ", VALIDATION_BATCHES_PER_EPOCH, '\n')
		print("TEST_BATCHES_PER_EPOCH: ", TEST_BATCHES_PER_EPOCH, '\n')

		for i in range(EPOCHS):
			print("Epoch", i)
			sess.run(accuracy_streaming_mean_reset)
			for j in range(TRAIN_BATCHES_PER_EPOCH):
				image_data, labels = dataset.train.next_batch()
				feed_dict = {images: image_data, correct_labels: labels, dropout_prob: 0.5, training: True}
				#Log summary for every last train batch
				if j == TRAIN_BATCHES_PER_EPOCH-1:
					summary, _, _ = sess.run([merged_summary, train_step, accuracy_streaming_mean_update], feed_dict=feed_dict)
					train_writer.add_summary(summary, i)
				else:
					sess.run([train_step, accuracy_streaming_mean_update], feed_dict=feed_dict)

			#Perform validation after each epoch of training
			sess.run(accuracy_streaming_mean_reset)
			for j in range(VALIDATION_BATCHES_PER_EPOCH):
				image_data, labels = dataset.validation.next_batch()
				feed_dict = {images: image_data, correct_labels: labels, dropout_prob: 1.0, training: False}
				if j == VALIDATION_BATCHES_PER_EPOCH-1: #Log summary for every last validation batch
					summary, validation_batch_accuracy, _ = sess.run([merged_summary, accuracy, accuracy_streaming_mean_update], feed_dict)
					validation_writer.add_summary(summary, i)
				else:
					validation_batch_accuracy, _ = sess.run([accuracy, accuracy_streaming_mean_update], feed_dict)
				print('\tBatch %d, Validation accuracy %g, streaming_acc_avg %g' % (j, validation_batch_accuracy, accuracy_streaming_mean.eval()))

		testTime = time.time()
		sess.run(accuracy_streaming_mean_reset)
		for i in range(TEST_BATCHES_PER_EPOCH):
			image_data, labels = dataset.test.next_batch()
			feed_dict = {images: image_data, correct_labels: labels, dropout_prob: 1.0, training: False}
			summary, test_batch_accuracy, _ = sess.run([merged_summary, accuracy, accuracy_streaming_mean_update], feed_dict)
			test_writer.add_summary(summary, i)
			print('Batch size %d, Test accuracy %g, Avg %g' % (len(labels), test_batch_accuracy, accuracy_streaming_mean.eval()))

		print('\nMetrics:')
		print("Load Time: %fs" %(trainTime - startTime))
		print("Train Time: %fs" %(testTime - trainTime))
		print("Test Time: %fs" %(time.time() - testTime))
		print("Total Time taken: %fs" %(time.time() - startTime))

		process_pool.close()
		process_pool.join()
		train_writer.close()
		validation_writer.close()
		test_writer.close()
		writer.close()

		saver.save(sess, './model/model')

