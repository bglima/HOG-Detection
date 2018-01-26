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

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print ("[INFO] extracting features...")

data = []   # Feature vectors
labels = [] # Label for each feature vector
(winW, winH) = (200, 100) # Window size for images
min_loss = 10.0 # Min loss so that a result is considered valid

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

print "[INFO] evaluating..."
cv2.namedWindow('MyWindow')
for imagePath in paths.list_images(args["test"]):
    # load the test image, convert it to grayscale, and resize it to
    # the canonical size
    image_original = cv2.imread(imagePath)
    # Make image size pre determined
    image_pre_sized = cv2.resize(image_original, (500, 250))
    # List of detections that will be sorted
    detected = []

    print "Analyzing {}...".format(imagePath)
    for image_resized in pyramid(image_pre_sized, downscale=1.2, minSize=(winW, winH)):
        for (x, y, image) in sliding_window(image_resized, stepSize=16, windowSize=(winW, winH) ):
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # discard if window does not match prescpecified dimensions.
            # This happens when window reaches end of image wither vertically or horizontally.
            if gray.shape[0] != winH or gray.shape[1] != winW:
                continue


            # extract Histogram of Oriented Gradients from the test image and
            # predict the make of the car
            (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
                 cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
            
            # Make H a line vector
            X = H.reshape(1, -1)
            # Predict loss from 1 nearest neighbor
            loss, _ = model.kneighbors(X, 1, return_distance=True)
            # Predict label from it
            pred = model.predict(X)[0]
            print('{} of distance from class {}'.format(loss, pred))
            
            # If loss is acceptable, add to detected vector
            if loss < min_loss:
                detected.append([x, y, loss, gray, pred])
    
    # Find minimum loss between 
    best_candidate = min( detected, key=lambda x: x[2] )
    print('Best detection from {} got loss {} as being from class {}'.format(imagePath, best_candidate[2], best_candidate[4]))
    cv2.imshow('MyWindow', best_candidate[3])
    key = cv2.waitKey(0)
    if ( key == ord('q') ):
        break
    

cv2.destroyAllWindows()