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
(winW, winH) = (200, 100)

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
     _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     c = max(cnts, key=cv2.contourArea)

	# extract the logo of the car and resize it to a canonical width
	# and height
     (x, y, w, h) = cv2.boundingRect(c)
     logo = gray[y:y + h, x:x + w]
     logo = cv2.resize(logo, (winW, winH))

     # extract Histogram of Oriented Gradients from the logo
     H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
          cells_per_block=(2, 2), transform_sqrt=True)

     # update the data and labels
     data.append(H)
     labels.append(make)     
     print('Extracting features from {}'.format(imagePath) )
    
# "train" the nearest neighbors classifier
print "[INFO] training classifier..."
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print "[INFO] training complete!"

#%%

print "[INFO] evaluating..."
cv2.namedWindow('MyWindow')
for imagePath in paths.list_images(args["test"]):
    # load the test image, convert it to grayscale, and resize it to
    # the canonical size
    image_original = cv2.imread(imagePath)
    image_pre_sized = cv2.resize(image_original, (400, 200))

    detected = []

    for image_resized in pyramid(image_pre_sized, downscale=1.2):
        for (x, y, image) in sliding_window(image_resized, stepSize=16, windowSize=(winW, winH) ):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] != winH or gray.shape[1] != winW:
                continue


            #logo = cv2.resize(gray, (200, 100))     
            # extract Histogram of Oriented Gradients from the test image and
            # predict the make of the car
            (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                 cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
            	
            X = H.reshape(1, -1)
            loss, _ = model.kneighbors(X, 1, return_distance=True)
            pred = model.predict(X)[0]
            print('{} of distance from class {}'.format(loss, pred))
            
            if loss < 5.0:
                detected.append([x, y, loss, gray, pred])
    
    sorted_detection = sorted( detected, key=lambda x: x[2] )
    print('Best detection loss: {} as being from class {}'.format(sorted_detection[0][2], sorted_detection[0][4]))
    cv2.imshow('MyWindow', sorted_detection[0][3])
    key = cv2.waitKey(0)
    if ( key == ord('q') ):
        break
    

cv2.destroyAllWindows()