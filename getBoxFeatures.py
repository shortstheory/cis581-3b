import numpy as np
import sys 
import cv2
from corner_detector import *
from anms import *
def getBoxFeatures(gray,box,n):
    res = corner_detector(gray)
    x, y, rmax = anms(res,n)
    # x=np.append(x,0)
    # y=np.append(y,0)
    # x=np.append(x,gray.shape[1])
    # y=np.append(y,0)
    # x=np.append(x,0)
    # y=np.append(y,gray.shape[0])
    # x=np.append(x,gray.shape[1])
    # y=np.append(y,gray.shape[0])
    return x+box[0],y+box[1]