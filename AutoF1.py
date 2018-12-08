#!/usr/bin/env python
# coding: utf-8

# # IPCV Coursework
# ## Subtask 1 (and 2):
# ### a) Image annotation
# We automated the process of annotating a test image with the ground truth labeled as purple boundaries by altering the matplotlib example code at https://matplotlib.org/examples/pylab_examples/ginput_manual_clabel.html

# In[ ]:


import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(image):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imagergb)
    plt.show()

def drawshow(s):
    print(s)
    plt.title(s, fontsize=11)
    plt.draw()

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

def Eval (A,B,imgcol,thresh=0.4): #Geometric F1 Score
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
    return judge

## Programme Starts HERE

print ("Which images to load? (eg. 1 4 6 for images 1,4 and 6) ")
whichimgs = [int(x) for x in input().split() if 0<=int(x)<16]
print (whichimgs)

dart = bool(input("Detect faces(0) or dart(1)? "))

for i in whichimgs:

    image = cv2.imread('./images/dart'+str(i)+ '.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print ("dart" + str(i) + ".jpg loaded")

    plt.imshow(img)
    plt.waitforbuttonpress()
    plt.close()

    amount = int(input("How many objects do you want to annotate?: "))

    store = np.zeros((amount,2,2), dtype=int)

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


        print(pts)
    plt.close()
    print(store)

    ground = np.zeros((len(store),4),dtype=int)

    for k in range (len(store)):
        s = store[k][1] - store[k][0]
        ground[k] = (store[k][0][0], store[k][0][1], s[0],s[1])
        (x,y,w,h) = ground[k]
        cv2.rectangle(image, (x,y), (x+w,y+h), (128,0,128),3)
    plt.imshow(img)

    print ("DONE")

    # ### b) Automated TPR and F1-score calculation
    # Using the co-ordinated of the ground truths previously annotated, we run the Viola-Jones face detector and calculate the true positive rate (TPR) and the F1- score of the classification.

    # In[ ]:


    #Viola-Jones detector
    if dart:
        classifier = cv2.CascadeClassifier('./Subtask2/classifier/dartcascade/cascade.xml')
    else:
        classifier = cv2.CascadeClassifier('./Subtask1/frontalface.xml')


    img = cv2.imread('./images/dart' + str(i) + '.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if dart:
        obj = classifier.detectMultiScale(gray, 1.1, 10, 0, (20,20), (500,500))
    else:
        obj = classifier.detectMultiScale(gray, 1.1, 1, 0, (50,50), (500,500))

    if obj is ():
        print('No objects found')

    # Draw box by iteration
    for (x,y,w,h) in obj:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)

    imshow(image)
    saveloc = (str("dart" + str(i) + str("classified.jpg")))
    cv2.imwrite(saveloc,image)

    # Use the evaluator to determine if a classification is a true positive or a false positive.
    judge = Eval(ground, obj, image, thresh=0.5)

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

    def f1score(detection):
        tp = detection[0]
        fp = detection[1]
        fn = detection[2]
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)
        f1score = 2*((precision*recall)/(precision+recall))
        return f1score

    def tpr(detection):
        tp = detection[0]
        fn = detection[2]
        recall = tp/(tp + fn)
        return recall


    VJgraph = [TP, FP, FN, f1score(detection)]



    print("True Positive Rate of dart" + str(i) + ": ", tpr(detection))
    print("F1-score of dart" + str(i) + ": ", f1score(detection))
