from matplotlib import pyplot as plt
import cv2
import xml.etree.ElementTree as ET 
import numpy as np
import sys
from getBoundingBoxes import *
from getBoxFeatures import *
from corner_detector import *
from anms import *
from estimateAllTranslation import *

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
        'y': y
    }
    boxesData.append(boxData)

currentFrame = firstFrame
idx = 0
while(cap.isOpened()):
    ret,nextFrame = cap.read()
    if ret == False:
        break
    # do work
    gray = cv2.cvtColor(currentFrame,cv2.COLOR_BGR2GRAY)

    plt.imshow(cv2.cvtColor(nextFrame,cv2.COLOR_BGR2GRAY))
    fig = plt.gcf()

    for boxData in boxesData:
        boxData['x'],boxData['y'] = estimateAllTranslation(boxData['x'],boxData['y'],currentFrame,nextFrame)
        ax1 = fig.add_subplot(111)
        ax1.scatter(boxData['x'],boxData['y'],c='r',s=1)
    plt.savefig("outputs/"+str(idx).zfill(4)+".png")
    idx = idx+1
    currentFrame = nextFrame
