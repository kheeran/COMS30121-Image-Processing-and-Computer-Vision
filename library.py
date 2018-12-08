import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import cv2
import math
import time

## FUNCTION DEFINITIONS

# colour imshow since cv2 imshow doesnt work
def imshow(image, s=""):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(s + " Click to close.", fontsize=14)
    plt.imshow(imagergb)
    plt.waitforbuttonpress()
    plt.close()

def drawshow(s):
    plt.title(s, fontsize=11)
    plt.draw()

# Annotate image(s)
def annotate (image, amount):
    store = np.zeros((amount,2,2), dtype=int)
    img = np.copy(image)

    for k in range(0,amount):
        plt.imshow(img)
        drawshow('You will annotate the image, click to continue.')
        plt.waitforbuttonpress()

        while True:
            img2 = np.copy(img)
            pts = []

            while len(pts) < 2:
                drawshow('For each object, draw the boundary by selecting 2 opposite corners with your right click, then press enter.')
                pts = plt.ginput(2, timeout=-1, show_clicks=True)
                if len(pts) < 2:
                    drawshow('Too few points, starting over.')
                    time.sleep(1)  # Wait a second

            cv2.rectangle(img, (int(pts[0][0]),int(pts[0][1])), (int(pts[1][0]),int(pts[1][1])), (128,0,128), 5)
            store[k] = pts
            plt.imshow(img)
            drawshow('Done? Enter to continue, mouse click to redo this object. ')
            if plt.waitforbuttonpress(timeout=-1):
                break
            img = img2
            plt.close()
            plt.imshow(img)
    plt.close()
    ground = np.zeros((len(store),4),dtype=int)

    for k in range (len(store)):
        s = store[k][1] - store[k][0]
        ground[k] = (store[k][0][0], store[k][0][1], s[0],s[1])

    return ground

# Viola Jones dartboard detector
def ViolaJones (i, image, obj_classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #run classifier
    obj = obj_classifier.detectMultiScale(gray, 1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50,50), (500,500))
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
# using a geometric f1 score
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

# get True positive, False Positive and False Negative from judgement
def getinfo (judge, obj):
    TP = 0
    FN = 0

    for k in range (len(judge)):
        sums = np.sum(judge[k])
        if sums == 0:
            FN += 1
        elif sums >= 1:
            TP += 1
        else:
            print ('error: missing condition')
    FP = len(obj) - TP
    detection = [TP, FP, FN]
    print (judge)
    print (detection)
    return detection

# Calculate the F1-score from [TP, FP, FN]
def f1score(detection):
    tp = detection[0]
    fp = detection[1]
    fn = detection[2]
    if tp == 0:
        return 0
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1score = 2*((precision*recall)/(precision+recall))
    return f1score

# Calculate the true positive rate from [TP, FP, FN]
def tpr(detection):
    tp = detection[0]
    fn = detection[2]
    recall = tp/(tp + fn)
    return recall

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
    print ("Performing Hough Troansform - Circles:")
    for m in range (0,M): #y
        if m % np.round(M/30) == 0:
            print (str(int(np.round(100*(m)/(M)))) + "%" )
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
def PlotRectangle (imgcol, Hxyr, Hspace, minrad, maxrad, prox=50):

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
        if math.sqrt((x0-x)**2 + (y0-y)**2)>prox:
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
        if math.sqrt((x0-x)**2 + (y0-y)**2)>prox and math.sqrt((x1-x)**2 + (y1-y)**2)>prox:
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
