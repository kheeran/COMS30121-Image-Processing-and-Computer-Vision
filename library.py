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

def imresize (image):
    hh, ww, dd = image.shape
    if ww < 1000:
        return image
    imscale = 500/ww
    newX, newY = image.shape[1]*imscale, image.shape[0]*imscale
    newimage = cv2.resize(image, (int(newX), int(newY)))
    return newimage

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
def Eval (A,B,imgcol,thresh): #Geometric F1 Score
    judge = np.zeros((len(A),len(B)))
    Area = Compare (A,B)
    Areainv = Compare (B,A)
    boxes = []
    for a in range (len(A)):
        for b in range (len(B)):
            p = Area[a,b]
            r = Areainv[b,a]
            if (p+r) ==0:
                judge[a,b] = 0
            elif 2*p*r/(p+r)>thresh:
                judge[a,b]= 1
                boxes.append(A[a])
                (x,y,w,h) = A[a]
                cv2.rectangle(imgcol, (x,y) , (x+w,y+h),(0,255,0), 3)

    return judge, boxes

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
    if tp+fn == 0:
        return 1.0
    elif tp == 0:
        return 0
    recall = tp/(tp + fn)
    return recall

# Calculate the precision
def ppv(detection):
    tp = detection[0]
    fp = detection[1]
    if tp+fp==0:
        return 1.0
    elif tp ==0:
        return 0
    precision = tp/(tp + fp)
    return precision

# Plot a barchart of the results
def f1bar(result1, result2, whichimgs):
    image_labels = []
    for i in whichimgs:
        each_image = 'dart'+str(i)+'jpg.'
        image_labels.append(each_image)

    indices = np.arange(len(image_labels))
    width = 0.35

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bar1 = ax.bar(indices, result1, width, color = 'royalblue', label = 'VJ F1-Score')
    bar2 = ax.bar(indices+width, result2, width, color = 'seagreen', label = 'VJHT F1-Score')
    plt.xticks(indices+width/2, image_labels, rotation = 'vertical')
    plt.legend(loc = 'lower left', bbox_to_anchor=(0,1.02,1,0.2), mode = 'expand', ncol = 2)
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
            if direc[m,n] < 0:
                    direc[m,n] += 2*math.pi
            if grad[m,n]>thresh:
                grad[m,n] = 255
            else:
                grad[m,n] = 0
                direc[m,n] = 0
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
    print ("Performing Hough Transform - Circles:")
    for m in range (0,M): #y
        if m % np.round(M/30) == 0:
            print ("progress - " + str(int(np.round(100*(m)/(M)))) + "%" )
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

    circle0 = (x0-int(r),y0-int(r),int(2*r),int(2*r))

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
    circle1 = (x1-int(r),y1-int(r),int(2*r),int(2*r))

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

    circle2 = (x2-int(r),y2-int(r),int(2*r),int(2*r))

    return circle0, circle1, circle2

def Q3(whichdartimgs = [1], minrad=10, maxrad=100, proximity=50, edgethresh=3, judgethresh=0.5):
    start = time.time()
    for i in whichdartimgs:
        # Loading a given image
        location = str("./images/dart") + str(i) + str(".jpg")
        imgcol = cv2.imread(location)
        img = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)
        print ("dart" + str(i) + ".jpg loaded")
        if len(whichdartimgs) == 1:
            imshow(imgcol, "Original Image.")

        # Finding the edges of the image above a threshold
        stime = time.time()
        grad, direc = EdgeDetect (img, threshavg=edgethresh)
        print ("EdgeDetect runtime: " + str(time.time() - stime) )

        if len(whichdartimgs) == 1:
            plt.title ("Edge image. Click to close")
            plt.imshow (grad, cmap='gray')
            plt.waitforbuttonpress()
            plt.close()

        #Running the Hough Transform for circles
        stime = time.time()
        Hxyr = HTCircle(grad, direc, minrad, maxrad)
        etime = time.time()
        print("runtime: Hough Transform " + str(etime-stime))


        # Finding the Hough Space for the Hough Transform
        stime = time.time()
        Hspace = HSpace(Hxyr)
        print ("Hough Space Runtime: " + str(time.time()-stime))

        if len(whichdartimgs) == 1:
            plt.title ("Hough Space. Click to close")
            plt.imshow(Hspace, cmap='gray')
            plt.waitforbuttonpress()
            plt.close()

        # Finding the rectangle that encloses the 3 most likely circles
        rect0, rect1, rect2 = PlotRectangle(imgcol, Hxyr, Hspace, minrad, maxrad, prox=proximity)
        dart_HT = np.array([rect0,rect1,rect2])

        # Plotting the HT detection on the coloured image
        for (x,y,w,h) in dart_HT:
                cv2.rectangle(imgcol, (x,y), (x+w,y+h), (255,165,0), 3)

        if len(whichdartimgs) == 1:
            imshow(imgcol, "Hough Transform detection.")

        #Reloading a fresh coloured image
        imgcol = cv2.imread(location)

        # Finding abd labeling the detected dartboards for Viola-Jones
        stime = time.time()
        classifier = cv2.CascadeClassifier('./Subtask2/classifier/dartcascade/cascade.xml')
        dart_VJ = ViolaJones(i, imgcol, classifier)
        print ("Viola-Jones Runtime: " + str(time.time()-stime))
        for (x,y,w,h) in dart_VJ:
                cv2.rectangle(imgcol, (x,y), (x+w,y+h), (0,165,255), 3)

        if len(whichdartimgs) == 1:
            imshow(imgcol, "Viola-Jones detection.")

        # Reload the coloured image
        imgcol = cv2.imread(location)

        # Combining Viola-Jones and Hough Transform by finding the overlapping classifications and plotting the corresponding VJ rectangle
        judgement, dart_VJHT = Eval (dart_VJ,dart_HT, imgcol, thresh=judgethresh)
        # Note: the judgement array is unused here

        if len(whichdartimgs) == 1:
            imshow(imgcol, "Joint HT & VJ detection.")
        print ("dart" + str(i) + ".jpg done")
    print ("Total runtime: " + str(time.time()-start))
    return dart_VJHT
