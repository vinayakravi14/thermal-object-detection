import os
import cv2 


def import_data(path, type:str):
    img = cv2.imread(path)

    if type == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if type == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
        

def blob_params():
    params = cv2.SimpleBlobDetector_Params()
      
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;
        
        # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 7000
        
        # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
        
        # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.01
        
        # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    return params