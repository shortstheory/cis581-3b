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
    nX = np.arange(startX-5,startX+6)
    nY = np.arange(startY-5,startY+6)
    nCx,nCy = np.meshgrid(nX,nY)
    nCX = nCx.flatten().astype('int')
    nCY = nCy.flatten().astype('int')
    nCX2 = nCX.copy()
    nCY2 = nCY.copy()
    # gray image deltas
    Ixp = Ix[nCY,nCX].reshape(-1,1)
    Iyp = Iy[nCY,nCX].reshape(-1,1)
    A = np.hstack([Ixp,Iyp])
    while ((u+v)>1 and i<15):
        # we don't really need to use interp2 for img2 but it's easy way to get pts
        It = interp2(img1G,nCX,nCY)-interp2(img2G,nCX2,nCY2) # WTF!
        u,v = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(It)
        nCY2 = nCY2+v
        nCX2 = nCX2+u
        x2 = x2+u
        y2 = y2+v
        xDelete = np.argwhere(x2>img1.shape[1])
        x2 = np.delete(x2, xDelete)
        y2 = np.delete(y2, xDelete)
        yDelete = np.argwhere(y2>img1.shape[0])
        x2 = np.delete(x2, yDelete)
        y2 = np.delete(y2, yDelete)
        i=i+1
    if (x2<0 or x2>img1.shape[1] or y2<0 or y2>img1.shape[1]):
        x2 = 0
        y2 = 0
    return x2,y2
