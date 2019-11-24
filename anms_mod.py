import numpy as np

def anms_mod(x_old1,y_old1,valid,cimg, max_pts):
    x_old = x_old1[valid].reshape(-1,1)
    y_old = y_old1[valid].reshape(-1,1)
    keypt = np.asarray(np.nonzero(cimg)).T
    keypt_old = np.hstack([y_old,x_old])
    keypt = np.vstack([keypt,keypt_old])
    N = min(max_pts,keypt.shape[0])
    keyR = (cimg[np.nonzero(cimg)]).reshape(-1,1)
    keyR_old = (np.ones(x_old.shape)*2*max(keyR))
    keyR = np.vstack([keyR,keyR_old]).reshape(-1)
    keypts = np.zeros([keypt.shape[0],3])
    keypts[:,:2]=keypt
    keypts[:,2]= keyR
    distance = np.zeros(keypts.shape)
    distance[:,:2]=keypts[:,:2]
    maxdist = np.sqrt(cimg.shape[0]**2 + cimg.shape[1]**2)

    for i in range(keypts.shape[0]):
        condind = np.argwhere(np.logical_and(keyR[:]>keyR[i],keyR[:]<1.4*keyR[i])==1)
        condind = condind[:,0]
        condpts = keypts[condind,:]
        if (condpts.shape[0]!=0):
            dist = np.sqrt((condpts[:,0]-keypts[i,0])**2 + (condpts[:,1]-keypts[i,1])**2)
            minD = dist.min()
            distance[i,2]=minD
        else:
            distance[i,2] = maxdist

    distanceSorted = distance[distance[:,2].argsort(kind='mergesort')]
    distanceSorted = np.flip(distanceSorted,axis=0)
    topOld = distanceSorted[:N-x_old.shape[0],:2]
    topNew = distanceSorted[N-x_old:N.shape[0],:2]

    # rmax = distanceSorted[N-1,2]
    y = y_old1
    x = x_old1
    y[valid==0]=topNew[:,0]
    x[valid==0]=topOld[:,1]

    valid = np.ones(x.shape[0],dtype=bool)
    return x, y, valid
