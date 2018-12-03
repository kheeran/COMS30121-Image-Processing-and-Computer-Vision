import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import cv2
import math
import time

pylab.rcParams['figure.figsize'] = (20,10)

## FUNCTION DEFINITIONS

# colour imshow since cv2 imshow doesnt work
def imshow(image):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imagergb)
    plt.show()

# Detect edges using the sobel operator
def EdgeDetect (img, threshavg=3):

    #Applying the Sobel Operator (try rewrite using own sobel)

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    (M,N) = sobelx.shape
    grad = np.zeros((M,N))
    direc = np.zeros((M,N))

    thresh = threshavg*np.sum(img)/(N*M) #the higher the quicker, but risk not representing edges

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
                direc[m,n] = 0
#     FaceRemove (img, grad, direc)
    return grad, direc


# Hough Transform using circles

def HTCircle (grad, direc, minrad, maxrad):

    nc =  maxrad-minrad #num circles
    (M,N) = grad.shape
    rad = np.zeros(nc-minrad)
    for a in range (0,nc-minrad):
        rad[a] = (a+minrad+1)*maxrad/nc

    Hxyr = np.zeros ((M,N,nc))
    total= M*N

    deg = math.pi/180
    for m in range (0,M): #y
        for n in range (0,N): #x
            if grad[m,n] == 255:
                for r in rad:

                    # optimisation of the for loop to only cater for the top left quarter of a circle to its center
                    y0 = int(np.round(m + r*math.sin(direc[m,n])))
                    x0 = int(np.round(n + r*math.cos(direc[m,n])))

                    # to cater for noise we +/- 1 degree
                    y1 = int(np.round(m + r*math.sin(direc[m,n]+deg)))
                    x1 = int(np.round(n + r*math.cos(direc[m,n]+deg)))
                    y2 = int(np.round(m + r*math.sin(direc[m,n]-deg)))
                    x2 = int(np.round(n + r*math.cos(direc[m,n]-deg)))

                    # removing 2 if statements to speed up the loop
                    dx = 2*abs(x2-x1)
                    dy = 2*abs(y2-y1)
                    radindex = int(nc*r/maxrad -(minrad+1))

                    if dy<=y0<M-dy and dx<=x0<N-dx:
                        Hxyr[y0,x0,radindex] += 1
                        Hxyr[y1,x1,radindex] += 1
                        Hxyr[y2,x2,radindex] += 1
                    else:
                        break #break to prevent unecessary calculation wiht larger r values
    return Hxyr

#Creating Hough Space

def HSpace (Hxyr):
    (a,b,c) = Hxyr.shape #to calculate the no of elements
    Hspace = np.zeros((a,b))

    for y in range (0,a):
        for x in range (0,b):
            sum = 0
            for z in range (0,c):
                sum += Hxyr[y,x,z]
            Hspace[y,x] = sum

    #normalising the values
    norm = np.amax(Hspace)
    Hspace = Hspace*255/norm
    return Hspace

# Finding the rectangle enclosing the most likely 3 circles, not in close proximity to each other (50 pixels)
def PlotCircle (imgcol, Hxyr, Hspace, minrad, maxrad):

    nc =  maxrad-minrad #num circles
    (a,b) = Hspace.shape #for later to calculate the highest index of the flattened array
    argordH = Hspace.argsort(axis=None) #lists the indices in accending order of value (array is flattened)

    index = 1
    #in order to detect images with multiple dart boards, we find the top highest Hspace values that dont have

#circle0
    (y0,x0)=np.unravel_index(argordH[a*b-index], (a,b)) #unravels the flattened value back into a tuple for the nth highest
    amax = Hxyr[y0,x0].shape
    argordxyr = Hxyr[y0,x0].argsort(axis=None)
    rindex = argordxyr[amax[0]*1-1]
    r = (rindex+minrad+1)*maxrad/nc
#     cv2.circle(imgcol, (x0,y0), int(r), (0,255,0), 3, cv2.LINE_AA)

    circle0 = (x0-int(r),y0-int(r),int(2*r),int(2*r))
#     circle0 = np.zeros((2*int(r), 2*int(r), 2))
#     for p in range (0,int(2*r)):
#         for q in range (0,int(2*r)):
#             circle0[p,q] = (y0-int(r)+p, x0-int(r)+q)

#circle1
    index += 1
    for c in range(index, a*b):
        (y,x)=np.unravel_index(argordH[a*b-c], (a,b)) #unravels the flattened value back into a tuple for the nth highest
        if math.sqrt((x0-x)**2 + (y0-y)**2)>20:
            index = c
            break

    (y1,x1)=np.unravel_index(argordH[a*b-index], (a,b)) #unravels the flattened value back into a tuple for the nth highest
    amax = Hxyr[y1,x1].shape
    argordxyr = Hxyr[y1,x1].argsort(axis=None)
    rindex = argordxyr[amax[0]*1-1]
    r = (rindex+minrad+1)*maxrad/nc
#     cv2.circle(imgcol, (x1,y1), int(r), (0,0,255), 3, cv2.LINE_AA)

    circle1 = (x1-int(r),y1-int(r),int(2*r),int(2*r))
#     circle1 = np.zeros((2*int(r), 2*int(r), 2))
#     for p in range (0,int(2*r)):
#         for q in range (0,int(2*r)):
#             circle1[p,q] = (y1-int(r)+p, x1-int(r)+q)

#circle2
    for c in range(index, a*b):
        (y,x)=np.unravel_index(argordH[a*b-c], (a,b)) #unravels the flattened value back into a tuple for the nth highest
        if math.sqrt((x0-x)**2 + (y0-y)**2)>20 and math.sqrt((x1-x)**2 + (y1-y)**2)>50:
            index = c
            break

    (y2,x2)=np.unravel_index(argordH[a*b-index], (a,b)) #unravels the flattened value back into a tuple for the nth highest
    amax = Hxyr[y2,x2].shape
    argordxyr = Hxyr[y2,x2].argsort(axis=None)
    rindex = argordxyr[amax[0]*1-1]
    r = (rindex+minrad+1)*maxrad/nc
#     cv2.circle(imgcol, (x2,y2), int(r), (255,0,0), 3, cv2.LINE_AA)

    circle2 = (x2-int(r),y2-int(r),int(2*r),int(2*r))
#     circle2 = np.zeros((2*int(r), 2*int(r), 2))
#     for p in range (0,int(2*r)):
#         for q in range (0,int(2*r)):
#             circle2[p,q] = (y2-int(r)+p, x2-int(r)+q)


    return circle0, circle1, circle2

# Viola Jones dartboard detector
def ViolaJones (i):
    obj_classifier = cv2.CascadeClassifier('cascade.xml')
    location = str("../images/dart") + str(i) + str(".jpg")
    image = cv2.imread(location)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #run classifier
    obj = obj_classifier.detectMultiScale(gray, 1.1, 10, 0, (20,20), (500,500))
    return obj

# Compare overlapping areas of 2 rectangles relative to A
def Compare (A,B):
    Area = np.zeros((len(A),len(B)))

    for ai in range (len(A)):
        for bi in range (len(B)):
            (x0,y0,w0,h0) = A[ai]
            (x1,y1,w1,h1) = B[bi]
            TotalA = w0*h0
            if x0<=x1 and y0<=y1:
                for xs in range(x0,x0+w0):
                    for ys in range (y0, y0+h0):
                        if xs==x1 and ys==y1:
                            w = w0-(x1-x0)
                            h = h0-(y1-y0)
                            if w>w1:
                                w = w1
                            if h>h1:
                                h = h1
                            Area[ai,bi] = w*h/TotalA
                            break
            elif x0>x1 and y0>y1:
                for xs in range (x1, x1+w1):
                    for ys in range(y1,y1+h1):
                        if xs==x0 and ys==y0:
                            w = w1-(x0-x1)
                            h = h1-(y0-y1)
                            if w>w0:
                                w = w0
                            if h>h0:
                                h = h0
                            Area[ai,bi] = w*h/TotalA
                            break
            elif x0<=x1:
                for xs in range(x0, x0+w0):
                    if xs==x1:
                            w = w0 - (x1-x0)
                            h = h1 - (y0-y1)
                            if w>w1:
                                w = w1
                            if h>h0:
                                h = h0
                            Area[ai,bi] = w*h/TotalA
                            break
            elif y0<=y1:
                for xs in range(x1, x1+w1):
                    if xs == x0:
                            w = w1 - (x0-x1)
                            h = h0 - (y1-y0)
                            if w>w0:
                                w = w0
                            if h>h1:
                                h = h1
                            Area[ai,bi] = w*h/TotalA
                            break
            else:
                print('missing case/condition')
    return Area

# Determine if the overlap of 2 rectangles is large enough
def Eval (A,B,imgcol,thresh=0.5): #Geometric F1 Score
    judge = np.zeros((len(A),len(B)))
    Area = Compare (A,B)
    Areainv = Compare (B,A)
    for a in range (len(A)):
        for b in range (len(B)):
            p = Area[a,b]
            r = Areainv[b,a]
            if (p+r) ==0:
                judge[a,b] = 0
            elif 2*p*r/(p+r)>thresh:
                judge[a,b]= 1
                (x,y,w,h) = A[a]
                cv2.rectangle(imgcol, (x,y) , (x+w,y+h),(0,255,0), 3)

    return judge

## PROGRAMME STARTS

start = time.time()
for i in range (0,16):

    # Loading a given image
    location = str("../images/dart") + str(i) + str(".jpg")
    imgcol = cv2.imread(location)
    img = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)

    # Finding the edges of the image above a threshold
    stime = time.time()
    grad, direc = EdgeDetect (img, threshavg=3)
    print ("EdgeDetect runtime: " + str(time.time() - stime) )

    # Saving the edge image
    saveloc = (str("detected/dart" + str(i) + str("edge.jpg")))
    cv2.imwrite(saveloc,grad)
    print ("Edge image saved")

    # Setting the max and min radius of a detected circle in HT
    minrad = 10
    maxrad = 100

    #Running the Hough Transform for circles
    stime = time.time()
    Hxyr = HTCircle(grad, direc, minrad, maxrad)
    etime = time.time()
    print("runtime: Hough Transform " + str(etime-stime))


    # Finding the Hough Space for the Hough Transform
    stime = time.time()
    Hspace = HSpace(Hxyr)

    #Saving the Hough Space image
    saveloc = (str("detected/dart" + str(i) + str("HS.jpg")))
    cv2.imwrite(saveloc,Hspace)
    print ("Hough Space Runtime: " + str(time.time()-stime))
    print ("Hough image saved")

    # Finding the rectangle that encloses the 3 most likely circles
    rect0, rect1, rect2 = PlotCircle(imgcol, Hxyr, Hspace, minrad, maxrad)
    dart_HT = np.array([rect0,rect1,rect2])

    # Plotting the HT detection on the coloured image
    for (x,y,w,h) in dart_HT:
            cv2.rectangle(imgcol, (x,y), (x+w,y+h), (255,165,0), 3)

    # Saving the HT detection image
    saveloc = (str("detected/dart" + str(i) + str("HS_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Hough transform image saved")

    #Reloading a fresh coloured image
    imgcol = cv2.imread(location)

    # Finding the detected dartboards for Viola-Jones
    dart_VJ = ViolaJones(i)
    for (x,y,w,h) in dart_VJ:
            cv2.rectangle(imgcol, (x,y), (x+w,y+h), (0,165,255), 3)

    # Saving VJ detected image
    saveloc = (str("detected/dart" + str(i) + str("VJ_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Viola-Jones image saved")

    # Reload the coloured image
    imgcol = cv2.imread(location)

    # Combining Viola-Jones and Hough Transform by finding the overlapping classifications and plotting the corresponding VJ rectangle
    judge = Eval (dart_VJ,dart_HT, imgcol)

    # Saving the detection of combined VJ and HT
    saveloc = (str("detected/dart" + str(i) + str("HSVJ_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Joing HT & VJ image saved")
    print ("dart" + str(i) + ".jpg done")
print ("Total runtime: " + str(time.time()-start))
