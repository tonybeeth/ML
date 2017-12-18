import tensorflow as tf
import cv2
import glob
import numpy as np

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

#directories containing images
PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Rainy/2012-09-16/' + 'Occupied' + '/*.jpg'

files = glob.glob(pattern)

# Access saved Variables directly
#print(sess.run('accuracy'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
#accuracy = graph.get_operation_by_name('accuracy')
accuracy = graph.get_tensor_by_name('accuracy:0')
#print(accuracy)
images = graph.get_tensor_by_name('images:0')
correct_labels = graph.get_tensor_by_name('correct_labels:0')
correct_pred = graph.get_tensor_by_name('correct_pred:0')
predicted = graph.get_tensor_by_name('predicted:0')
dropout_prob = graph.get_tensor_by_name('dropout_prob:0')
image_data = tf.random_uniform([1,28,28,1])
labels=tf.constant([0.0, 1.0], shape=[1, 2])

for file in files:
	img = cv2.imread(file)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(img_gray, (28, 28))
	np.reshape(resized_image, (1, 28, 28, 1))
	#print(type(resized_image))
	feed_dict={images: resized_image.reshape(1, 28, 28,1), correct_labels: labels.eval(session=sess), dropout_prob: 1.0}
	print(sess.run(accuracy, feed_dict), file)
#print(accuracy.eval(sess,feed_dict))

ops = graph.get_operations()
#print(ops)