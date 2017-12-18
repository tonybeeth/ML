from PIL import Image
import numpy as np
import glob
import time
import cv2
import multiprocessing

train_percent = 0.5
image_size = 28
batch_size = 100

#directories containing images
PKLOT_DIR = '../../../../PKLot/PKLot/'
PKLOT_SEGMENTED_DIR = PKLOT_DIR + 'PKLotSegmented/'

#Retrieves images from modifiable directory
def GetImagesPaths(status):
	pattern = PKLOT_SEGMENTED_DIR + 'PUCPR/Cloudy/*/' + status + '/*.jpg'
	paths = glob.glob(pattern)
	print (pattern + '. #files: ' + str(len(paths)))
	return paths

def processImg(file):
	#file = file.decode("utf-8") #convert bytes to string
	#print(file)
	img = cv2.imread(file)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img_gray
	#resized_image = cv2.resize(img_gray, (self._image_size, self._image_size))
	#yield resized_image[..., np.newaxis] #adds a new axis(grayscale=1) and returns array of (size, size, 1)
	
if __name__ == "__main__":
	paths = GetImagesPaths('Empty')
	print(len(paths))

	start = time.time()

	pool=multiprocessing.Pool(8)
	results=pool.map(processImg, paths)
	#print(results[0])
	# print results
	# for file in paths:
		# processImg(file)
		
	end = time.time()
	print('')
	print(end-start)

