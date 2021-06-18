#!/usr/bin/env python
import numpy as np
import cv2


def discretize_entry_points(dense_map, threshold=0.9):
    '''
    Ref: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    '''
    # Convert to binary map
    _, dense_map = cv2.threshold(dense_map,0.9 , 1. ,cv2.THRESH_BINARY)
    dense_map = (255.*dense_map).astype(np.uint8)
    # NOTE: Dilate in order to reduce posibility of zero-thickness clusters
    kernel = np.ones((3,3), np.uint8) 
    dense_map = cv2.dilate(dense_map, kernel, iterations=1) 

    # Find separated clusters    
    #_, contours, _ = cv2.findContours(dense_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(dense_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    entry_pnt_list = []
    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)
        # Calculate x,y coordinate of center
        c_x = int(M["m10"] / M["m00"])
        c_y = int(M["m01"] / M["m00"])

        entry_pnt_list.append((c_x, c_y))

    return entry_pnt_list
