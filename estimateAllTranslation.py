from scipy import signal
import numpy as np
from estimateFeatureTranslation import *
import cv2
from utils import *

def estimateAllTranslation(startXs,startYs,img1,img2):
    img1G = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2G = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    Imag1,Ix1,Iy1,Iori1 = findDerivatives(img1G)
    newXs = np.zeros(startXs.shape)
    newYs = np.zeros(startYs.shape)
    for i in range(startXs.shape[0]): #x-coord
        newXs[i],newYs[i] = estimateFeatureTranslation(startXs[i],startYs[i],Ix1,Iy1,img1,img2)
    return newXs,newYs
