import numpy as np
import cv2
from utils import *
from ransac_est_homography import ransac_est_homography

def applyBoxTransform(startX, startY, newX, newY, box):
    # u = np.mean(newX-startX)
    # v = np.mean(newY-startY)
    # box[0] += u
    # box[1] += v
    # box[2] += u
    # box[3] += v
    H = ransac_est_homography(startX,startY,newX,newY,0.001)
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
    boxStk = [[xmin,xmax,xmin,xmax],
              [ymin,ymin,ymax,ymax],
              [1,1,1,1]]
    boxPts = H@boxStk
    boxPts = boxPts/boxPts[2:]
    # print(boxPts)
    box = np.array([np.min(boxPts[0,:]),np.min(boxPts[1,:]),np.max(boxPts[0,:]),np.max(boxPts[1,:])])
    return newX,newY,box