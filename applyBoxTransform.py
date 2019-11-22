import numpy as np
import cv2
from utils import *
from ransac_est_homography import ransac_est_homography

def applyBoxTransform(startX, startY, newX, newY, box, valid):
    shiftedBox = np.array([0,0,0,0])
    newX = newX[valid]
    newY = newY[valid]
    startX = startX[valid]
    startY = startY[valid]
    H = ransac_est_homography(startX,startY,newX,newY,0.001)
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
    shiftedBox = np.array([np.min(boxPts[0,:]),np.min(boxPts[1,:]),np.max(boxPts[0,:]),np.max(boxPts[1,:])])
    # xmin,ymin,xmax,ymax

    return shiftedBox