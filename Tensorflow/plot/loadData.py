# Typical setup to include TensorFlow.
import tensorflow as tf
import os
import glob

PKLOT_DIR = '../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

lot_Ids = ['PUCPR/','UFPR04/','UFPR05/']
weather_Ids = ['Cloudy/', 'Rainy/', 'Sunny/']

def GetEmptySpotsQueue():
	return GetSpotsQueue('Empty')
def GetOccupiedSpotsQueue():
	return GetSpotsQueue('Occupied')
def GetSpotsQueue(spotsStatus):
	pattern = PKLOT_SEGMENTED_DIR + '*/*/*/' + spotsStatus + '/*.jpg'
	print (pattern + '. #files: ' + str(len(glob.glob(pattern))))
	return tf.train.string_input_producer(tf.train.match_filenames_once(pattern))

def GetSPPPas():
	for lot_Id in lot_Ids:
		for weather_Id in weather_Ids:
			date_Ids = os.listdir(PKLOT_SEGMENTED_DIR + lot_Id + weather_Id)
			for date_Id in date_Ids:
				current_dir_jpg_pattern = PKLOT_SEGMENTED_DIR + lot_Id + weather_Id + date_Id + '/Empty/*.jpg'