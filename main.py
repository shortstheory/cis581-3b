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
    coords = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
    boxData = {
        'coords': coords,
        'x': x,
        'y': y,
        'startX': x.copy(),
        'startY': y.copy(),
        'startCoords': box,
        'valid': np.ones(x.shape[0],dtype='bool')
    }
    boxesData.append(boxData)
currentFrame = firstFrame
idx = 0
boxUpdateRate = 30
while(cap.isOpened()):
    ret,nextFrame = cap.read()
    if ret == False:
        break
    # do work
    gray = cv2.cvtColor(currentFrame,cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(nextFrame,cv2.COLOR_BGR2RGB))
    fig = plt.gcf()

    for boxData in boxesData:
        # boxData['startX'] = boxData['x']
        # boxData['startY'] = boxData['y']
        boxData['x'],boxData['y'],boxData['valid'] = estimateAllTranslation(boxData['x'],boxData['y'],boxData['valid'],currentFrame,nextFrame)
        
        # if np.sum(boxData['valid'])<8:
        # x,y = getBoxFeatures(boximg,box,20)
        # x = boxData['x']
        # y = boxData['y']
        # coords = boxData['coords']


        # if idx % 150 == 0:
        # boxData['startCoords'] = boxData['coords']
        startX = boxData['startX']
        startY = boxData['startY']
            # boxData = {
            #     'coords': coords,
            #     'x': x,
            #     'y': y,
            #     'startX': x.copy(),
            #     'startY': y.copy(),
            #     'startCoords': coords,
            #     'valid': np.ones(x.shape[0],dtype='bool')
            # }


        ax1 = fig.add_subplot(111)
        ax1.scatter(boxData['x'][boxData['valid']],boxData['y'][boxData['valid']],c='r',s=1)
        print(np.sum(boxData['valid']),end=" ")
        if np.sum(boxData['valid']) >= 4:
            # if idx%10==0:
            boxData['coords'] = applyBoxTransform(startX,startY,boxData['x'],boxData['y'],boxData['startCoords'],boxData['valid'])
            coords = boxData['coords']
            # xmin,ymin,xmax,ymax
            if coords[0] > 0 and coords[1] > 0 and coords[2] < currentFrame.shape[1] and coords[3] < currentFrame.shape[0]:
                rect = patches.Rectangle((coords[0],coords[1]),coords[2]-coords[0],coords[3]-coords[1],linewidth=1,edgecolor='b',facecolor='none')
                ax1.add_patch(rect)
    plt.savefig("outputs2/"+str(idx).zfill(4)+".png")
    fig.clf()
    idx = idx+1
    print(idx)
    currentFrame = nextFrame
