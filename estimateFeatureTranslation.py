import numpy as np
import cv2

def interp2(v, xq, yq):
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    # if dim_input == 2:
    #     return interp_val.reshape(q_h, q_w)
    return interp_val

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    i=0
    img1G = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2G = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x2,y2 = startX,startY
    u,v=10,10
    print(img1.shape)
    nX = np.arange(startX-5,startX+6)
    nY = np.arange(startY-5,startY+6)
    nCx,nCy = np.meshgrid(nX,nY)
    nCX1 = nCx.flatten().astype('int')
    nCY1 = nCy.flatten().astype('int')
    nCX2 = nCX1.copy()
    nCY2 = nCY1.copy()
    Ixp = Ix[nCY1,nCX1].reshape(-1,1)
    Iyp = Iy[nCY1,nCX1].reshape(-1,1)
    A = np.hstack([Ixp,Iyp])
    while (abs(u+v)>0.1 and i<15):
        # print('i values',i)
        It = -interp2(img2G,nCX2,nCY2)+interp2(img1G,nCX1,nCY1)
        # print((interp2(img1G,nCX1,nCY1)-img2G[nCY2,nCX2]).sum())
        u,v = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(It)
        nCY2 = nCY2+v
        nCX2 = nCX2+u
        x2 = x2+u
        y2 = y2+v
        i=i+1
        print('u',u)
        print('v',v)
        # print('u',u)
        # print('v',v)
        #if (x1<0 or x1>img1.shape[1] or y1<0 or y1>img1.shape[1]):
         #   x1 = 0
          #  y1 = 0
           # break
    return x2,y2
