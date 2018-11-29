#!/usr/bin/env python
# coding: utf-8

# In[39]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import time

for i in range (0,16):
    # location = "../images/dart.bmp"
    location = str("../images/dart") + str(i) + str(".jpg")
    imgcol = cv2.imread(location)
    img = np.array(cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY))

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    (M,N) = sobelx.shape
    grad = np.zeros((M,N))
    direc = np.zeros((M,N))
    thresh = 0.5*np.sum(img)/(N*M) #setting a threshold of half the average gradient, so circles are more complete

    for m in range (0,M): #m = y
        for n in range  (0,N): # n = x
            grad[m,n] = math.sqrt(sobelx[m,n]**2 + sobely[m,n]**2)
            if sobelx[m,n] == 0: # to prevent division by zero in direc
                sobelx[m,n]= 1*10**(-5) # set the value to very small which approximates the direc very well
            direc[m,n] = np.arctan(sobely[m,n]/sobelx[m,n])
            if grad[m,n]>thresh:
                grad[m,n] = 255
            else:
                grad[m,n] = 0

    saveloc1 = (str("edgedetected/dartedge" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc1,grad)
    print ("save edge img")

    # In[40]:


    #Hough Circle

    #assuming the full board has to be in the image and the board isn't distorted.
    maxrad = int(max(M,N)/10)
    nc =  maxrad #numcir

    stime = time.time()
    rad = np.zeros(nc)
    for a in range (10,nc): # not zero to stop detection of tiny dots
        rad[a] = (a+1)*maxrad/nc

    Hxyr = np.zeros ((M,N,nc))
    total= M*N

    #eventhough we start in the top right corner, flipping and rotating the space doesnt affect the result
    #if we use many values of theta and if there is a edge at that grad(x0,y0) then H(x0,y0,r) + 1???
    for m in range (0,M): #y
        for n in range (0,N): #x
            if grad[m,n] == 255:
                for r in rad:
                    y0 = int(np.round(m + r*math.sin(direc[m,n])))
                    y1 = int(np.round(m - r*math.sin(direc[m,n])))
                    x0 = int(np.round(n + r*math.cos(direc[m,n])))
                    x1 = int(np.round(n - r*math.cos(direc[m,n])))
                    radindex = int(nc*r/maxrad -1)
                    #case where all the points are in the grid (majority of the points so first)
                    if 0<=y0<M and 0<=x0<N and 0<=y1<M and 0<=x1<N:
                        Hxyr[y0,x0,radindex] += 1
                        Hxyr[y1,x0,radindex] += 1
                        Hxyr[y0,x1,radindex] += 1
                        Hxyr[y1,x1,radindex] += 1
                    else:
                        # other cases
                        if 0<=y0<M and 0<=x0<N:
                            Hxyr[y0,x0,radindex] += 1
                        if 0<=y1<M and 0<=x1<N:
                            Hxyr[y1,x1,radindex] += 1
                        if 0<=y0<M and 0<=x1<N:
                            Hxyr[y0,x1,radindex] += 1
                        if 0<=y1<M and 0<=x0<N:
                            Hxyr[y1,x0,radindex] += 1
    etime = time.time()
    print("runtime, Hough space: " + str(etime-stime))


    # In[41]:


    #Drawing most likely circles back onto image

    imgcol = cv2.imread(location)

    (a,b,c) = Hxyr.shape #to calculate the no of elements
    nth = 10#number of green circles drawn

    argord = Hxyr.argsort(axis=None) #lists the indeces in accending order of value (array is flattened)
    (y,x,r)=np.unravel_index(argord[a*b*c-nth], (a,b,c)) #unravels the flattened value back into a tuple for the nth highest
    thresh = Hxyr[y,x,r]

    stime = time.time()
    for m in range (0,M):
        for n in range (0,N):
            for radindex in range (0,nc):
                if Hxyr[m,n,radindex]>thresh:
                    r = (radindex+1)*maxrad/nc
                    cv2.circle(imgcol, (m,n), int(r), (0,255,0), 3, cv2.LINE_AA)
    etime = time.time()
    print("runtime, circ draw: " + str(etime-stime))
    saveloc2 = (str("circledetected/dartcirc" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc2,imgcol)
    print (str(i) + " image(s) done")
