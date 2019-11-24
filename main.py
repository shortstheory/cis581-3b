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
from refreshFeatures import *
# white car - corner 3
# black car - corner 3
cap = cv2.VideoCapture('vids/Medium.mp4')
ret,firstFrame = cap.read()
boxes = getBoundingBoxes('first.xml')
gray = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)
boxesData = []
pts = 20
for box in boxes:
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    boximg = gray[ymin:ymax,xmin:xmax]    
    x,y = getBoxFeatures(boximg,box,pts)
    boxCoords = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    boxData = {
        'coords': boxCoords,
        'x': x,
        'y': y,
        'valid': np.ones(x.shape[0],dtype='bool'),
        'prevBoxCoords': np.array([[0,0],[0,0],[0,0],[0,0]]),
        'displayHeight': 0,
        'displayWidth': 0,
        'displayCorner': np.array([[0,0],[0,0],[0,0],[0,0]])
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
        print(np.sum(boxData['valid']),end=" ")
        validpts = np.sum(boxData['valid'])
        if validpts<pts:
            # h,w,corner = getMinBox(boxData['coords'])
            h,w,corner = getMinPointsBox(boxData['x'],boxData['y'])
            print(h,w,corner)
            if h == 0 or w == 0 or validpts <= 6:
                h = 100
                w = 100
            if h > 0 and w > 0 and corner[0] >= 0 and corner[1] >= 0:
                boximg = gray[corner[1]:corner[1]+h,corner[0]:corner[0]+w]
                boxData['x'],boxData['y'],boxData['valid'] = refreshFeatures(boximg,corner,boxData['x'],boxData['y'],boxData['valid'],pts)
                print("Feature refreshed")
                prevX = boxData['x']
                prevY = boxData['y']
            else:
                boxData['x'],boxData['y'],boxData['valid'] = estimateAllTranslation(boxData['x'],boxData['y'],boxData['valid'],currentFrame,nextFrame)
        else:
            boxData['x'],boxData['y'],boxData['valid'] = estimateAllTranslation(boxData['x'],boxData['y'],boxData['valid'],currentFrame,nextFrame)
        ax1 = fig.add_subplot(111)
        ax1.scatter(boxData['x'][boxData['valid']],boxData['y'][boxData['valid']],c='r',s=1)
        if np.sum(boxData['valid']) >= 4:
            boxData['coords'] = applyBoxTransform(prevX,prevY,boxData['x'],boxData['y'],boxData['coords'],boxData['valid'])
            coords = boxData['coords']
            delta = linalg.norm(coords-boxData['prevBoxCoords'],axis=1)
            minDeltaIdx = np.argmin(delta)

            boxData['prevBoxCoords'] = coords
            # h, w, corner = getMinBox(coords)
            h,w,corner = getMinPointsBox(boxData['x'],boxData['y'])
            if idx % 10 == 0:
                boxData['displayCorner'] = corner
                boxData['displayHeight'] = h
                boxData['displayWidth']  = w
            rect = patches.Rectangle(boxData['displayCorner'],boxData['displayWidth'],boxData['displayHeight'],linewidth=1,edgecolor='r',facecolor='none')
            ax1.add_patch(rect)
            ring = LinearRing(coords)
            x, y = ring.xy
            # ax1.plot(x, y)
    plt.savefig("outputs2/"+str(idx).zfill(4)+".png")
    fig.clf()
    idx = idx+1
    print(idx)
    currentFrame = nextFrame
