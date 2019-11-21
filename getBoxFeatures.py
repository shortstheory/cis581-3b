import numpy as np
import sys 
import cv2
from corner_detector import *
from anms import *
def getBoxFeatures(gray,box):
    res = corner_detector(gray)
    x, y, rmax = anms(res,10)
    return x+box[0],y+box[1]