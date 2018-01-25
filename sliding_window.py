# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:10:29 2018

@author: brunolima
"""

# Common stepSize vary from 4 to 8  
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[ y:y + windowSize[1], x:x + windowSize[0] ])
            
def main():
    from image_pyramid import pyramid
    import time
    import cv2
    
    # load the image and define the window width and height
    image = cv2.imread('subaru-logo.jpg')
    (winW, winH) = (200, 100)
    
    for resized in pyramid(image, downscale=1.5):
        # loop over the slideing window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            
            
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            key = cv2.waitKey(1)
            if( key == ord('q')): return
        
            time.sleep(0.025)
            
if __name__ == '__main__':
    main()
    print('All done')
    cv2.destroyAllWindows()  