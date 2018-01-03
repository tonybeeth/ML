import tkinter as tk
import cv2
from PIL import ImageTk, Image
import time
import numpy as np
import os
import glob
import random
from pprint import pprint
import tensorflow as tf
import re
import xmltodict
import json

image_size = 64

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return cv2.resize(warped, (image_size, image_size))

def extract_image_data(img_path):
    image = cv2.imread(img_path)
    #Get associated xml file with same path
    xml_path = re.sub('jpg$', 'xml', img_path)

    spots = []
    coords = []
    with open(xml_path, 'rb') as xml_file:
        json_data = json.loads(json.dumps(xmltodict.parse(xml_file)))
    spaces = json_data['parking']['space']
    for space in spaces:
        try:
            points = space['contour']['point']
        except(KeyError):
            points = space['contour']['Point'] #Some xmls use caps 'P' for Point

        #Extract coords with Lambda function
        coord = lambda i: (int(points[i]['@x']), int(points[i]['@y']))
        botleft, topleft, topright, botright = coord(0), coord(1), coord(2), coord(3)
        coords.append((botleft, topleft, topright, botright))
        #Extract min and max points with Lambda function
        coords_op = lambda op: (op(botleft[0], botright[0], topleft[0], topright[0]), op(botleft[1], botright[1], topleft[1], topright[1]))
        minpoint = coords_op(min)
        maxpoint = coords_op(max)

        #Warp image
        #https://stackoverflow.com/questions/2992264/extracting-a-quadrilateral-image-to-a-rectangle/2992759#2992759
        corners = np.array([botleft, topleft, topright, botright], np.float32)
        target = np.array([(0,0), (0,image_size), (image_size, image_size), (image_size, 0)], np.float32)
        mat = cv2.getPerspectiveTransform(corners, target)
        spot = cv2.warpPerspective(image, mat, (image_size, image_size))
        spots.append(spot)
    return image, coords, np.asarray(spots)

if __name__ == "__main__":
    if os.environ.get('PKLOT_DATA') is None:
        print('Cannot locate PKLOT_DATA in environment')
        exit()

    window = tk.Tk()
    canvas = tk.Canvas(window, width=1300, height=750)
    canvas.pack()

    PKLOT_DIR = os.environ['PKLOT_DATA'] + '/PKLot/PKLot/'
    lot_names = ['PUCPR', 'UFPR04', 'UFPR05']

    #Retrieve 'num_imgs_per_lot' random images from each parking lot
    img_paths = []
    num_imgs_per_lot = 10
    for lot_name in lot_names:
        lot_img_pattern = PKLOT_DIR + lot_name + '/*/*/*.jpg'
        lot_img_paths = glob.glob(lot_img_pattern)
        img_paths.extend(random.sample(lot_img_paths, num_imgs_per_lot))
    random.shuffle(img_paths)

    #Create session config with soft placement and prevent allocation of all GPU memory
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #Load meta graph and restore weights
        saver = tf.train.import_meta_graph('../NBM_model/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../NBM_model/'))
        #Get tensors from loaded graph
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('input/images:0')
        training_flag = graph.get_tensor_by_name('input/training_flag:0')
        keep_prob = graph.get_tensor_by_name('dropout/keep_prob:0')
        predicted_results = graph.get_tensor_by_name('Readout/predicted_results:0')

        #Summaries to visualize activation map from CNN2 on tensorboard
        #cnn1_batch_norm_output = graph.get_tensor_by_name('CNN2/batch_normalization/cond/Merge:0')
        #splits = tf.split(cnn1_batch_norm_output, 32, 3)
        #summaries = [tf.summary.image('cnn2_' + str(i), split, 32) for split, i in zip(splits, range(len(splits)))]
                
        #summary_writer = tf.summary.FileWriter('log/', sess.graph)

        img_paths = glob.glob('D:\PKLot/PKLot/PKLot/UFPR04/Rainy/2013-01-21/*.jpg')
        for img_path, i in zip(img_paths, range(len(img_paths))):
            image, coords, spots = extract_image_data(img_path)
            feed_dict = {images: spots, keep_prob: 1.0, training_flag: False}
            occupancy_results = sess.run(predicted_results, feed_dict)
            #summary = sess.run(summaries, feed_dict)

            #for s, j in zip(summary, range(len(summary))):
            #   summary_writer.add_summary(s, i*len(img_paths)+j)
            for spot_result, spot_coords in zip(occupancy_results, coords):
                #Green for empty spots, red for occupied
                if bool(spot_result) is True:
                    #BGR
                    color = (0,0,255)
                else:
                    color = (0,255,0)
                #Draw lines around spot on image
                cv2.polylines(image, [np.array(spot_coords)], isClosed=True, color=color, thickness=2)

            #Display results on image
            cv2.imshow(img_path, image)
            cv2.waitKey(0)

        # for img_path in img_paths:
        #     img = cv2.imread(img_path)
        #     xml_path = re.sub('jpg$', 'xml', img_path)

        #     img = ImageTk.PhotoImage(Image.fromarray(img))
        #     canvas.create_image(10,10, anchor=tk.NW, image=img)
        #     window.update()
        #     time.sleep(.5)

        #summary_writer.close()
