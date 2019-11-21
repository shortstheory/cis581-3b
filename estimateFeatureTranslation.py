import numpy as np
import cv2

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    i=0
    img1G = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2G = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x2,y2 = startX,startY
    u,v=10,10
    nX = np.arange(startX-5,startX+6)
    nY = np.arange(startY-5,startY+6)
    nCx,nCy = np.meshgrid(nX,nY)
    nCX = nCx.flatten()
    nCY = nCy.flatten()
    nC1 = np.vstack((nCY,nCX)).astype('int')
    nCX2 = nCX.copy()
    nCY2 = nCY.copy()
    nC2 = np.vstack((nCY2,nCX2)).astype('int')
    # gray image deltas
    It = -img2G[nC1[0],nC1[1]]+img1G[nC2[0],nC2[1]]
    Ixp = Ix[nC1[0],nC1[1]].reshape(-1,1)
    Iyp = Iy[nC1[0],nC1[1]].reshape(-1,1)
    A = np.hstack([Ixp,Iyp])
    while ((u+v)>1 and i<15):
        It = -img2G[nC1[0],nC1[1]]+img1G[nC2[0],nC2[1]]
        u,v = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(It)
        nC2[0] = nC2[0]+v
        nC2[1] = nC2[1]+u
        x2 = x2+u
        y2 = y2+v
        i=i+1
    if (x2<0 or x2>img1.shape[1] or y2<0 or y2>img1.shape[1]):
        x2 = 0
        y2 = 0
    return x2,y2
