from matplotlib import pyplot as plt
import cv2
import xml.etree.ElementTree as ET 
import numpy as np
import sys
from getBoundingBoxes import *
from getBoxFeatures import *
from corner_detector import *
from anms import *

cap = cv2.VideoCapture('vids/Easy.mp4')
ret,firstFrame = cap.read()
boxes = getBoundingBoxes()
gray = cv2.cvtColor(firstFrame,cv2.COLOR_BGR2GRAY)

boxesData = []

for box in boxes:
    boximg = gray[box[1]:box[3],box[0]:box[2]]
    x,y = getBoxFeatures(boximg,box)
    boxData = {
        'coords': box,
        '_x': x,
        '_y': y
    }
    boxesData.append(boxData)
print(boxesData)
currentFrame = firstFrame
while(cap.isOpened()):
    ret,nextFrame = cap.read()
    if ret == False:
        break
    # do work

    currentFrame = nextFrame
