#run using python3 getFirstFrame.py vids/Easy.mp4
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(sys.argv[1])
ret,frame = cap.read()
cv2.imwrite('first.png',frame)
