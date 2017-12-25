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

        #Crop and resize image
        spot = image[minpoint[1]:maxpoint[1], minpoint[0]:maxpoint[0]]
        spot = cv2.resize(spot, (image_size, image_size))
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
        saver = tf.train.import_meta_graph('../model_larger_batches/model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('../model_larger_batches/'))
        #Get tensors from loaded graph
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name('input/images:0')
        training_flag = graph.get_tensor_by_name('input/training_flag:0')
        keep_prob = graph.get_tensor_by_name('dropout/keep_prob:0')
        predicted_results = graph.get_tensor_by_name('Readout/predicted_results:0')

        #Summaries to visualize activation map from CNN2 on tensorboard
        cnn1_batch_norm_output = graph.get_tensor_by_name('CNN2/batch_normalization/cond/Merge:0')
        splits = tf.split(cnn1_batch_norm_output, 32, 3)
        summaries = [tf.summary.image('cnn2_' + str(i), split, 32) for split, i in zip(splits, range(len(splits)))]
                
        summary_writer = tf.summary.FileWriter('log/', sess.graph)

        img_paths = img_paths[:1]
        for img_path, i in zip(img_paths, range(len(img_paths))):
            image, coords, spots = extract_image_data(img_path)
            feed_dict = {images: spots, keep_prob: 1.0, training_flag: False}
            occupancy_results = sess.run(predicted_results, feed_dict)
            summary = sess.run(summaries, feed_dict)

            for s, j in zip(summary, range(len(summary))):
                summary_writer.add_summary(s, i*len(img_paths)+j)
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

        summary_writer.close()
