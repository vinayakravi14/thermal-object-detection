
'''
Simple CV Object tracker
===================================================
This Object tracker uses 'SimpleBlobDetector' for identifying blobs and counting the no. of blobs detected.
Currently it takes in 'path' of folder containing the images as an input and can be used as follows:

Usage
-----
detector.py '/path/to/folder/' '.file_format'


===================================================

'''

import cv2
import numpy as np 
import os
import sys
sys.path.append('..')
from helpers.utility import import_data, blob_params


class ObjectDetector:

    def __init__(self, frame:str):
        self.frames = frame
 
    def feature_extractor(self, thresh_value=130):
        
        # Normalizing  the image
        normIR = cv2.normalize(self.frames, self.frames, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply a color heat map for visualization
        colorIR = cv2.applyColorMap(normIR, cv2.COLORMAP_JET)
        
        # Use a bilateral filter to blur while hopefully retaining edges
        brightBlurIR = cv2.bilateralFilter(normIR, 9, 150, 150)

        # Threshold the image to black and white
        retval, threshIR = cv2.threshold(brightBlurIR, thresh_value, 255, cv2.THRESH_BINARY)

        # Define kernal for erosion and dilation and closing operations
        kernel = np.ones((5, 5), np.uint8)

        erosionIR = cv2.erode(threshIR, kernel, iterations=1)

        dilationIR = cv2.dilate(erosionIR, kernel, iterations=1)

        closingIR = cv2.morphologyEx(threshIR, cv2.MORPH_CLOSE, kernel)
        
        # Detect edges with Canny detection, currently only for visual testing not counting
        edgesIR = cv2.Canny(closingIR, 105, 70, L2gradient=True)

        # Detect countours, you can use and track with contours additionally (not used here)
        contours, hierarchy = cv2.findContours(closingIR, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # getting number of contours
        ncontours = str(len(contours))

        # Inverting the image
        invertIR = cv2.bitwise_not(closingIR)

        # -------- Beginning Blob Detection -------- #
        # Setup SimpleBlobDetector parameters.
        
        params = blob_params()
        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(invertIR)

        # Draw detected blobs as red circles.
        
        IR_with_keypoints = cv2.drawKeypoints(invertIR, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Showing keypoints
        nblobs = str(len(keypoints))

        # -------- Ending Blob Detection -------- #
        
        # Put text number of Contour Keypoints on Screen in Blue
        cv2.putText(IR_with_keypoints, "No. of players are: " + nblobs, (80, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # make all arrays same color space befor concatenating
        RGBnormIR = cv2.cvtColor(normIR, cv2.COLOR_GRAY2RGB)
        brightBlurIR = cv2.cvtColor(brightBlurIR, cv2.COLOR_GRAY2RGB)
        edgesIR = cv2.cvtColor(edgesIR, cv2.COLOR_GRAY2RGB)

        # stack 2 sets of images side by side for testing 
        imstack1 = np.concatenate((edgesIR, colorIR), axis=1) 
        imstack2 = np.concatenate((brightBlurIR, IR_with_keypoints), axis=1) 

        # Then stack those 2 verticaly
        imstack = np.concatenate((imstack1, imstack2), axis=0) 

        # Resizing image to fit the window
        cv2.namedWindow("Object-detector window", cv2.WINDOW_GUI_EXPANDED)
        imS = cv2.resize(imstack, (1000, 800))
        cv2.imshow("Object-detector window", imS)
        
        #adjust frame_rate as per needed
        cv2.waitKey(100)

if __name__ == "__main__":
    print(__doc__)
        
    try:
        path = sys.argv[1]
        file_format = sys.argv[2]
    except :
        print("[INFO]... 'Enter proper folder/file format', \n Use folder path as: '/path/to/folder/' file formats as: '.png','.jpg'\n")
        print("[INFO]... 'Using from default folder example'")
        path = "../data/footy2/"
        file_format = '.png'

    #getting in all the images from the inputted folder
    for image in sorted(os.listdir(path)):
        if image.endswith(file_format):
            filename = os.path.join(path,image)

            # change if you get an assertion error to 'rgb' depending on your dataset 
            frames = import_data(filename,"gray")
            OD = ObjectDetector(frames)
            OD.feature_extractor()

