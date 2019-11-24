from matplotlib import patches, pyplot, pyplot as plt
import cv2
import xml.etree.ElementTree as ET 
import numpy as np
import sys
from getBoundingBoxes import *
from getBoxFeatures import *
from corner_detector import *
from anms import *
from estimateAllTranslation import *
from applyBoxTransform import applyBoxTransform
from shapely.geometry.polygon import LinearRing, Polygon
from numpy import linalg
# white car - corner 3
# black car - corner 3
cap = cv2.VideoCapture('vids/Easy.mp4')
ret,firstFrame = cap.read()
boxes = getBoundingBoxes('first.xml')
gray = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)
boxesData = []

for box in boxes:
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    boximg = gray[ymin:ymax,xmin:xmax]    
    x,y = getBoxFeatures(boximg,box,10)
    boxCoords = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    boxData = {
        'coords': boxCoords,
        'x': x,
        'y': y,
        'valid': np.ones(x.shape[0],dtype='bool'),
        'prevBoxCoords': np.array([[0,0],[0,0],[0,0],[0,0]]),
        'displayHeight': 0,
        'displayWidth': 0,
        'displayCorner': np.array([0,0])
    }
    boxesData.append(boxData)
currentFrame = firstFrame
idx = 0
boxUpdateRate = 10

while(cap.isOpened()):
    ret,nextFrame = cap.read()
    if ret == False:
        break
    # do work
    gray = cv2.cvtColor(currentFrame,cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(nextFrame,cv2.COLOR_BGR2RGB))
    fig = plt.gcf()
    for boxData in boxesData:
        prevX = boxData['x']
        prevY = boxData['y']
        boxData['x'],boxData['y'],boxData['valid'] = estimateAllTranslation(boxData['x'],boxData['y'],boxData['valid'],currentFrame,nextFrame)
        ax1 = fig.add_subplot(111)
        ax1.scatter(boxData['x'][boxData['valid']],boxData['y'][boxData['valid']],c='r',s=1)
        if np.sum(boxData['valid']) >= 4:
            boxData['coords'] = applyBoxTransform(prevX,prevY,boxData['x'],boxData['y'],boxData['coords'],boxData['valid'])
            coords = boxData['coords']
            delta = linalg.norm(coords-boxData['prevBoxCoords'],axis=1)
            minDeltaIdx = np.argmin(delta)
            print(delta)
            print(minDeltaIdx)
            boxData['prevBoxCoords'] = coords
            height = min(coords[3,1]-coords[0,1],coords[2,1]-coords[1,1])
            width = min(coords[1,0]-coords[0,0],coords[2,0]-coords[3,1])
            minDeltaIdx = 1
            if minDeltaIdx == 0:
                height = height
                width = width
            elif minDeltaIdx == 1:
                height = height
                width = -width
            elif minDeltaIdx == 2:
                height = -height
                width = -width
            elif minDeltaIdx == 3:
                height = -height
                width = width
            if idx % 1 == 0:
                boxData['displayCorner'] = coords[minDeltaIdx]
                boxData['displayHeight'] = height
                boxData['displayWidth']= width
            rect = patches.Rectangle(boxData['displayCorner'],boxData['displayWidth'],boxData['displayHeight'],linewidth=1,edgecolor='r',facecolor='none')
            ax1.add_patch(rect)
            ring = LinearRing(coords)
            # x,y = ring.xy
            # ax1.plot(x, y)
    plt.savefig("outputs2/"+str(idx).zfill(4)+".png")
    fig.clf()
    idx = idx+1
    print(idx)
    currentFrame = nextFrame
