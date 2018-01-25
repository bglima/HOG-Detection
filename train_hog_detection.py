# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:37:23 2018

@author: brunolima
"""

from imutils import paths
import imutils
import argparse
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage import exposure
from skimage import feature
from sliding_window import sliding_window
from image_pyramid import pyramid

#%%

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

#%%

#args = {}
#args["training"] = 'train'
#args["test"] = 'test'

#%% 
 
# initialize the data matrix and labels
print ("[INFO] extracting features...")
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images(args["training"]):
     # extract the make of the car
     make = imagePath.split("/")[-2]

     # load the image, convert it to grayscale, dand detect edges
     image = cv2.imread(imagePath)
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     edged = imutils.auto_canny(gray)
 
	# find contours in the edge map, keeping only the largest one which
	# is presumed to be the car logod
     image2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     c = max(cnts, key=cv2.contourArea)

	# extract the logo of the car and resize it to a canonical width
	# and height
     (x, y, w, h) = cv2.boundingRect(c)
     logo = gray[y:y + h, x:x + w]
     logo = cv2.resize(logo, (200, 100))

     # extract Histogram of Oriented Gradients from the logo
     H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
          cells_per_block=(2, 2), transform_sqrt=True)

     # update the data and labels
     data.append(H)
     labels.append(make)     
     
     print('Extracting features from {}.'.format(imagePath) )
     
# "train" the nearest neighbors classifier
print "[INFO] training classifier..."
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print "[INFO] training complete!"

#%%

print "[INFO] evaluating..."

for imagePath in paths.list_images(args["test"]):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (200, 100))
 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
	
	X = H.reshape(1, -1)
	neighbor, loss = model.kneighbors(X, 1, return_distance=True)
	pred = model.predict(X)[0]
	print('{} of Euclidian distance from class {}'.format(loss[0][0], pred))
     
	# visualize the HOG image
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image", hogImage)
 
	# draw the prediction on the test image and display it
	#cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
	#	(0, 255, 0), 3)
 
	cv2.imshow("Test Image", image)
	key = cv2.waitKey(0)
	if key == ord('q'): break 

cv2.destroyAllWindows()