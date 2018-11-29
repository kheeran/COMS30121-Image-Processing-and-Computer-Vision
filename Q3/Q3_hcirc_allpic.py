#!/usr/bin/env python
# coding: utf-8

# In[116]:


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
    thresh = 2*np.sum(img)/(N*M)

    stime = time.time()
    for m in range (0,M): #m = y
        for n in range  (0,N): # n = x
            grad[m,n] = math.sqrt(sobelx[m,n]**2 + sobely[m,n]**2)
            if sobelx[m,n] == 0: # to prevent division by zero in direc
                sobelx[m,n]= 1*10**(-5)
            direc[m,n] = np.arctan(sobely[m,n]/sobelx[m,n])
            if grad[m,n]>thresh:
                grad[m,n] = 255
            else:
                grad[m,n] = 0
    etime = time.time()
    print("runtime, edge detect: " + str(etime-stime))

    saveloc1 = (str("edgedetected/dartedge" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc1,grad)
    plt.imshow(grad, cmap='gray')


    # In[153]:


    #Hough Circle

    #assuming the full board has to be in the image and the board isn't distorted.
    maxrad = max(M,N)
    nc =  maxrad #numcir

    stime = time.time()
    rad = np.zeros(nc)
    for a in range (0,nc):
        rad[a] = (a+1)*maxrad/nc

    Hxyr = np.zeros ((M,N,nc))

    #eventhough we start in the top right corner, flipping and rotating the space doesnt affect the result
    #if we use many values of theta and if there is a edge at that grad(x0,y0) then H(x0,y0,r) + 1???
    for m in range (0,M): #y
        for n in range (0,N): #x
            if grad[m,n] == 255:
                for r in rad:
                    y0 = int(np.round(m + r*math.sin(direc[m,n])))
                    x0 = int(np.round(n + r*math.cos(direc[m,n])))
                    if 0<=y0<M and 0<=x0<N:
                        radindex = int(nc*r/maxrad -1)
                        Hxyr[y0,x0,radindex] += 1
                    else:
                        break
    etime = time.time()
    print("runtime, hough space: " + str(etime-stime))


    # In[ ]:


    #Drawing most likely circles back onto image

    def imshow(image):
        #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
        imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(imagergb)


    (a,b,c) = Hxyr.shape #for later to calculate the highest index of the flattened array
    nth = 40 #number of green circles drawn

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
    print("runtime, draw circle: " + str(etime-stime))

    saveloc2 = (str("circledetected/dartcirc" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc2,imgcol)
    print(str(i) + "th image done")
    # imshow(imgcol)
