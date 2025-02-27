import cv2
import xml.etree.ElementTree as ET
import numpy as np
import sys

def getBoundingBoxes(file):
    root = ET.parse(file).getroot()
    boxes = []
    for obj in root.findall('object'):
        for box in obj.findall('bndbox'):
            xmin = box.find('xmin').text
            xmax = box.find('xmax').text
            ymin = box.find('ymin').text
            ymax = box.find('ymax').text
        box = np.array([xmin,ymin,xmax,ymax])
        boxes.append(box)
    return np.array(boxes).astype('int')

