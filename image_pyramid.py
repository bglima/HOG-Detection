# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:26:43 2018

@author: brunolima
"""

import imutils
import cv2
from skimage.transform import pyramid_gaussian


def pyramid(image, downscale=2.0, minSize=(30, 30)):
    # tield the original image
    yield image
    
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / downscale)
        image = imutils.resize(image, width=w)
        
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        # yield the next image in the pyramid
        yield image

#%%

def main():
    # Loading an arbitrary image
    image = cv2.imread('train/audi/audi_01.jpg')
    
    # Testing our implemented function
    for (i, im) in enumerate(pyramid(image)):
        cv2.imshow('image_pyramid_{}'.format(i), im)
        print('Step {} in pyramid. Size is {} x {}'.format(i, im.shape[0], im.shape[1]))
        key = cv2.waitKey(0)
        if( key == ord('q') ): break
    
    # Testing the scikit-image function. It uses max_layer instead of minSize.
    # The total number of images is max_layer + 1.
    for (i, im) in enumerate(pyramid_gaussian(image, downscale=2, max_layer=2)):
        cv2.imshow('image_pyramid_gaussian_{}'.format(i), im)
        print('Step {} in pyramid. Size is {} x {}'.format(i, im.shape[0], im.shape[1]))
        key = cv2.waitKey(0)
        if( key == ord('q') ): break
    
    print('End of demo. Press any key to exit.')
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

#%%