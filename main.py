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

cap = cv2.VideoCapture('vids/Easy.mp4')
ret,firstFrame = cap.read()
boxes = getBoundingBoxes()
gray = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)

boxesData = []

for box in boxes:
    boximg = gray[box[1]:box[3],box[0]:box[2]]
    x,y = getBoxFeatures(boximg,box,10)
    boxData = {
        'coords': box,
        'x': x,
        'y': y,
        'startX': x.copy(),
        'startY': y.copy(),
        'startCoords': box
    }
    boxesData.append(boxData)
print(boxesData)
currentFrame = firstFrame
idx = 0

while(cap.isOpened()):
    ret,nextFrame = cap.read()
    if ret == False:
        break
    # do work
    gray = cv2.cvtColor(currentFrame,cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(nextFrame,cv2.COLOR_BGR2RGB))
    fig = plt.gcf()

    for boxData in boxesData:
        X,Y = estimateAllTranslation(boxData['x'],boxData['y'],currentFrame,nextFrame)
        startX = boxData['startX']
        startY = boxData['startY']
        # boxData['x'],boxData['y'],boxData['coords'] = applyBoxTransform(boxData['x'],boxData['y'],X,Y,boxData['coords'])
        boxData['x'],boxData['y'],boxData['coords'] = applyBoxTransform(startX,startY,X,Y,boxData['startCoords'])
        ax1 = fig.add_subplot(111)
        ax1.scatter(boxData['x'],boxData['y'],c='r',s=1)
        coords = boxData['coords']
        # xmin,ymin,xmax,ymax
        rect = patches.Rectangle((coords[0],coords[1]),coords[2]-coords[0],coords[3]-coords[1],linewidth=1,edgecolor='b',facecolor='none')
        ax1.add_patch(rect)
    plt.savefig("outputs/"+str(idx).zfill(4)+".png")
    fig.clf()
    idx = idx+1
    print(idx)
    currentFrame = nextFrame
