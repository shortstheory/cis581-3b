import numpy as np
import sys 
import cv2
from corner_detector import *
from anms import *
def getBoxFeatures(gray,box,n):
    # res = corner_detector(gray)
    # x, y, rmax = anms(res,n)
    corners = cv2.goodFeaturesToTrack(gray,n,0.01,10)
    corners = np.int0(corners)
    x = corners[:,0,0]
    y = corners[:,0,1]
    return x+box[0],y+box[1]
