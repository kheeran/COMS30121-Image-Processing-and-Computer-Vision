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

    #implement own sobel operator***********************************
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    (M,N) = sobelx.shape
    grad = np.zeros((M,N))
    direc = np.zeros((M,N))
    thresh = 2*np.sum(img)/(N*M)

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
    # grad = cv2.Canny(img, thresh*0.5, thresh*2, 3 )

    saveloc1 = (str("edgedetected/dartedge" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc1,grad)
    print("image saved")
    # In[136]:

    #Hough Circle

    #assuming the full board has to be in the image and the board isn't distorted.
    maxrad = 100 #int(max(M,N)) #rad of largest circle
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
                    # optimisation of the for loop to only cater for the top left quarter of a circle to its center
                    y0 = int(np.round(m + r*math.sin(direc[m,n])))
    #                 y1 = int(np.round(m - r*math.sin(direc[m,n])))
                    x0 = int(np.round(n + r*math.cos(direc[m,n])))
    #                 x1 = int(np.round(n - r*math.cos(direc[m,n])))

                    y1 = int(np.round(m + r*math.sin(direc[m,n]+math.pi/180)))
                    x1 = int(np.round(n + r*math.cos(direc[m,n]+math.pi/180)))
                    y2 = int(np.round(m + r*math.sin(direc[m,n]-math.pi/180)))
                    x2 = int(np.round(n + r*math.cos(direc[m,n]-math.pi/180)))


                    radindex = int(nc*r/maxrad -1)
                    if 0<=y1<M and 0<=x1<N:
                        Hxyr[y1,x1,radindex] += 1
                    if 0<=y2<M and 0<=x2<N:
                        Hxyr[y2,x2,radindex] += 1
                    if 0<=y0<M and 0<=x0<N:
                        Hxyr[y0,x0,radindex] += 1
                    else:
                        break #break to prevent unecessary calculation wiht larger r values

    etime = time.time()
    print("runtime: Hough space " + str(etime-stime))

    #Plotting Hough Space
    (a,b,c) = Hxyr.shape #to calculate the no of elements
    Hspace = np.zeros((a,b))
    stime = time.time()
    for y in range (0,a):
        for x in range (0,b):
            sum = 0
            for z in range (0,c):
                sum += Hxyr[y,x,z]
            Hspace[y,x] = sum
    #normalising the values
    norm = np.amax(Hspace)
    Hspace = Hspace*255/norm
    etime = time.time()
    print ("runtime Hough space: " + str(etime-stime))

    saveloc3 = (str("circledetected/dart2circ_HS" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc3,Hspace)
    print ("image saved")

    np.amax(Hspace)

    # (M,N) = img.shape
    imgcol = cv2.imread(location)
    (a,b) = Hspace.shape #for later to calculate the highest index of the flattened array
    argord = Hspace.argsort(axis=None) #lists the indices in accending order of value (array is flattened)
    (y,x)=np.unravel_index(argord[a*b-1], (a,b)) #unravels the flattened value back into a tuple for the nth highest

    a = Hxyr[y,x].shape
    argord = Hxyr[y,x].argsort(axis=None)
    rindex = argord[a[0]-1]
    r = Hxyr[y,x,rindex]
    # r = np.amax(Hxyr[y,x])
    print (Hspace[y,x])
    cv2.circle(imgcol, (x,y), int(r), (0,255,0), 1, cv2.LINE_AA)
    saveloc2 = (str("circledetected/dart2circ" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc2,imgcol)
    print("image saved")
    print (str(i) + " image(s) done")
