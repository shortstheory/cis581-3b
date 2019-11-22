import numpy as np
import sys 
import cv2
from corner_detector import *
from anms import *
def getBoxFeatures(gray,box,n):
    res = corner_detector(gray)
    x, y, rmax = anms(res,n)
    return x+box[0],y+box[1]