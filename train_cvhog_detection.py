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

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
winStride = (8, 8)                        
padding = (8, 8)
locations = ((10,20),)

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
    H = hog.compute(image,winStride,padding,locations)
    
    # update the data and labels
    data.append(H)
    labels.append(make)     
    print('Shape of H array is {}'.format(H.shape))
    print('Extracting features from {}'.format(imagePath) )
    
cv2.destroyAllWindows()