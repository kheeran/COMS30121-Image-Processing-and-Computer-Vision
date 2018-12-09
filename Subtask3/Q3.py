import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import cv2
import math
import time
import library as lib
pylab.rcParams['figure.figsize'] = (20,10)

## PROGRAMME STARTS

# Pre-annotated ground truths
ground = {0: np.array([[435,   6, 167, 194]]), 1: np.array([[198, 143, 191, 173]]), 2:
np.array([[ 97, 101,  98,  80]]), 3: np.array([[322, 150,  69,  69]]), 4:
np.array([[175, 104, 172, 180]]), 5: np.array([[426, 141,  92,  97]]), 6:
np.array([[210, 120,  63,  57]]), 7: np.array([[241, 174, 127, 131]]), 8:
np.array([[840, 225, 123, 104],[ 67, 255,  59,  84]]), 9:
np.array([[188,  36, 257, 253]]), 10: np.array([[ 78, 101, 121, 113],
[578, 127,  60,  91],[916, 149,  37,  69]]), 11:
np.array([[170, 107,  68,  50]]), 12: np.array([[153,  74,  65, 146]]), 13:
np.array([[269, 126, 135, 130]]), 14:
np.array([[117, 106, 134, 113],[981, 101, 128, 113]]), 15:
np.array([[151,  56, 136, 131]])}

# Setting the thresholds
edgethresh = 2.2
judgethresh = 0.5
# Setting the max and min radius of a detected circle in HT
minrad = 10
maxrad = 100
# Set the min proximity of any 2 HT circles`
proximity = 70

whichdartimgs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# whichdartimgs = [9]

F1VJ = {}
F1VJHT = {}
PrecisionVJ = {}
PrecisionVJHT = {}
RecallVJ = {}
RecallVJHT = {}

start = time.time()
for i in whichdartimgs:

    # Loading a given image
    location = str("../images/dart") + str(i) + str(".jpg")
    imgcol = cv2.imread(location)
    img = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)
    print ("dart" + str(i) + ".jpg loaded")
    if len(whichdartimgs) == 1:
        lib.imshow(imgcol, "Original Image.")


    # Finding the edges of the image above a threshold
    stime = time.time()
    grad, direc = lib.EdgeDetect (img, threshavg=edgethresh)
    print ("EdgeDetect runtime: " + str(time.time() - stime) )

    # Saving the edge image
    saveloc = (str("detected/dart" + str(i) + str("edge.jpg")))
    cv2.imwrite(saveloc,grad)
    print ("Edge image saved")
    if len(whichdartimgs) == 1:
        plt.title ("Edge image. Click to close")
        plt.imshow (grad, cmap='gray')
        plt.waitforbuttonpress()
        plt.close()

    #Running the Hough Transform for circles
    stime = time.time()
    Hxyr = lib.HTCircle(grad, direc, minrad, maxrad)
    etime = time.time()
    print("runtime: Hough Transform " + str(etime-stime))


    # Finding the Hough Space for the Hough Transform
    stime = time.time()
    Hspace = lib.HSpace(Hxyr)

    #Saving the Hough Space image
    saveloc = (str("detected/dart" + str(i) + str("HS.jpg")))
    cv2.imwrite(saveloc,Hspace)
    print ("Hough Space Runtime: " + str(time.time()-stime))
    print ("Hough image saved")
    if len(whichdartimgs) == 1:
        plt.title ("Hough Space. Click to close")
        plt.imshow(Hspace, cmap='gray')
        plt.waitforbuttonpress()
        plt.close()

    # Finding the rectangle that encloses the 3 most likely circles
    rect0, rect1, rect2 = lib.PlotRectangle(imgcol, Hxyr, Hspace, minrad, maxrad, prox=proximity)
    dart_HT = np.array([rect0,rect1,rect2])

    # Plotting the HT detection on the coloured image
    for (x,y,w,h) in dart_HT:
            cv2.rectangle(imgcol, (x,y), (x+w,y+h), (255,165,0), 3)

    # Saving the HT detected colour image
    saveloc = (str("detected/dart" + str(i) + str("HS_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Hough transform image saved")
    if len(whichdartimgs) == 1:
        lib.imshow(imgcol, "Hough Transform detection.")

    #Reloading a fresh coloured image
    imgcol = cv2.imread(location)

    # Finding abd labeling the detected dartboards for Viola-Jones
    stime = time.time()
    classifier = cv2.CascadeClassifier('cascade.xml')
    dart_VJ = lib.ViolaJones(i, imgcol, classifier)
    for (x,y,w,h) in dart_VJ:
            cv2.rectangle(imgcol, (x,y), (x+w,y+h), (0,165,255), 3)

    # Saving VJ detected colour image
    saveloc = (str("detected/dart" + str(i) + str("VJ_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Viola-Jones Runtime: " + str(time.time()-stime))
    print ("Viola-Jones image saved")
    if len(whichdartimgs) == 1:
        lib.imshow(imgcol, "Viola-Jones detection.")

    # Reload the coloured image
    imgcol = cv2.imread(location)

    # Combining Viola-Jones and Hough Transform by finding the overlapping classifications and plotting the corresponding VJ rectangle
    _, dart_VJHT = lib.Eval(dart_VJ,dart_HT, imgcol, thresh=judgethresh)
    # Note: the judgement array is unused

    # Saving the detection of combined VJ and HT
    saveloc = (str("detected/dart" + str(i) + str("VJHS_detect.jpg")))
    cv2.imwrite(saveloc,imgcol)
    print ("Joint HT & VJ image saved")
    if len(whichdartimgs) == 1:
        lib.imshow(imgcol, "Joint HT & VJ detection.")

    # Reload the coloured image
    imgcol = cv2.imread(location)
    judgementVJHT, _ =lib.Eval(ground[i], dart_VJHT, imgcol, thresh=judgethresh)
    imgcol = cv2.imread(location)
    judgementVJ, _ = lib.Eval(ground[i], dart_VJ,imgcol, thresh=judgethresh)
    detectionVJHT = lib.getinfo(judgementVJHT,dart_VJHT)
    detectionVJ = lib.getinfo(judgementVJ,dart_VJ)
    F1VJHT[i] = lib.f1score(detectionVJHT)
    F1VJ[i] = lib.f1score(detectionVJ)
    PrecisionVJ[i] = lib.ppv(detectionVJ)
    PrecisionVJHT[i] = lib.ppv(detectionVJHT)
    RecallVJ[i] = lib.tpr(detectionVJ)
    RecallVJHT[i] = lib.tpr(detectionVJHT)
    print ("dart" + str(i) + ".jpg done")
print ("Total runtime: " + str(time.time()-start))
print ("F1-VJ:")
print (F1VJ)
print ("F1-VJHT:")
print (F1VJHT)

avgVJ_P = sum(PrecisionVJ.values())/len(PrecisionVJ)
avgVJ_R = sum(RecallVJ.values())/len(RecallVJ)
avgVJ_F1 = 2*avgVJ_P*avgVJ_R/(avgVJ_P + avgVJ_R)
print ("VJ - Mean F1: " + str(avgVJ_F1))

avgVJHT_R = sum(RecallVJHT.values())/len(RecallVJHT)
avgVJHT_P = sum(PrecisionVJHT.values())/len(PrecisionVJHT)
avgVJ_F1 = 2*avgVJHT_R*avgVJHT_P/(avgVJHT_P + avgVJHT_R)
print ("VJHT - Mean F1: " + str(avgVJHT_F1))

lib.f1bar(F1VJ, F1VJHT, whichdartimgs)
