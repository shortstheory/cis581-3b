import numpy as np
import cv2
from utils import *

def applyBoxTransform(startX, startY, newX, newY, box):
    H = est_homography(startX,startY,newX,newY)
    n = startX.shape[0]
    stk = np.vstack((startX,startY,np.ones(n)))
    res = H@stk
    res = res/res[2,:]
# [293, 187, 402, 266]
# xmin,ymin,xmax,ymax
    X = res[0,:]
    Y = res[1,:]
    xmin = box[0]
    xmax = box[2]
    ymin = box[1]
    ymax = box[3]
    boxStk = [[xmin,xmax],[ymin,ymax],[1,1]]
    newBox = H@boxStk
    newBox = newBox/newBox[2:]
    _box = np.array([newBox[0,0],newBox[1,0],newBox[0,1],newBox[1,1]])
    return X,Y,_box