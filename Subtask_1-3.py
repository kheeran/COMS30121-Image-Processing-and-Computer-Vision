import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import library as lib


# User selects which image to load.
print ("Which image(s) to load? (eg. 1 4 6 for images 1,4 and 6) ")
whichimgs = [int(x) for x in input().split() if 0<=int(x)<16]

# If only 1 image selected, show intermediary results.

if len(whichimgs)==0:
    whichimgs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

show = False
if len(whichimgs) == 1:
    show = True

dart = bool(int(input("Detect faces(0) or dart(1)? ")))
print (dart)
if dart:
    annotations = bool(input("Load annotations? (0 for no, 1 for yes)"))
else:
    annotations = False


if annotations:
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
else:
    ground = {}
f1 = {}
tpr = {}

# Case for annotating all images at the start
if not show and not annotations:
    for i in whichimgs:
        # Load dart(i).jpg and visually imspect the image
        location = './images/dart'+str(i)+ '.jpg'
        image = cv2.imread(location)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print ("dart" + str(i) + ".jpg loaded")
        if dart:
            plt.title("Note the number of dartboards to annotate. Click to continue.")
        else:
            plt.title("Note the number of faces to annotate. Click to continue.")
        plt.imshow(img)
        plt.waitforbuttonpress()
        plt.close()

        # Annotate the Ground truth
        amount = int(input("How many objects do you want to annotate?: "))
        ground[i] = lib.annotate(img, amount)

for i in whichimgs:
    # Load dart(i).jpg and visually imspect the image
    location = './images/dart'+str(i)+ '.jpg'
    image = cv2.imread(location)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print ("dart" + str(i) + ".jpg loaded")
    if dart:
        plt.title("Note the number of dartboards to annotate. Click to continue.")
    else:
        plt.title("Note the number of faces to annotate. Click to continue.")

    if show and not annotations:
        plt.imshow(img)
        plt.waitforbuttonpress()
        plt.close()

        # Annotate the Ground truth
        amount = int(input("How many objects do you want to annotate?: "))
        ground[i] = lib.annotate(img, amount)
        for (x,y,w,h) in ground[i]:
                cv2.rectangle(image, (x,y), (x+w,y+h), (128,0,128), 3)
        lib.imshow(image, "Ground Truth.")

    # Run Viola-Jones
    if dart:
        classifier = cv2.CascadeClassifier('./Subtask2/classifier/dartcascade/cascade.xml')
    else:
        classifier = cv2.CascadeClassifier('./Subtask1/frontalface.xml')

    obj = lib.ViolaJones(i, image, classifier)
    if show:
        tempimg = np.copy(image)
        for (x,y,w,h) in obj:
                cv2.rectangle(tempimg, (x,y), (x+w,y+h), (0,255,0), 3)
        lib.imshow(tempimg, "Viola-Jones detector.")

    # Compare VJ with the Ground Truth
    judgement, _ = lib.Eval(ground[i], obj, image, thresh=0.5)

    # Calsulate F1-score and True Positive Rate
    detection = lib.getinfo(judgement, obj)
    f1[i] = lib.f1score(detection)
    tpr[i] = lib.tpr(detection)


    print("True Positive Rate of dart" + str(i) + ": ", tpr[i])
    print("Precision of dart" + str(i) + ": ", lib.ppv(detection))
    print("F1-score of dart" + str(i) + ": ", f1[i])

plt.close()
resultf1 = []
resulttpr = []
for i in whichimgs:
    resultf1.append(f1[i])
    resulttpr.append(tpr[i])

if not dart:
    quit() # Since Q3 works only for dart boards.
## Starting Q3

# Setting the thresholds
edgethresh = 3
judgethresh = 0.4
# Setting the max and min radius of a detected circle in HT
minrad = 10
maxrad = 100
# Set the min proximity of any 2 HT circles`
proximity = 50

#Performing Hough Transform with circles on the images
dart_VJHT = lib.Q3(whichimgs, minrad,maxrad,proximity,edgethresh,judgethresh)

for i in whichimgs:
    # Storing f1-scores to plot in a bar chart
    image = cv2.imread(location)
    judgement, _ = lib.Eval(ground[i], dart_VJHT, image, thresh=0.5 )
    detection = lib.getinfo(judgement, dart_VJHT)
    f1[i] = lib.f1score(detection)

    image = cv2.imread(location)
    judgement, _ = lib.Eval(ground[i], obj, image, thresh=0.5 )
    detection = lib.getinfo(judgement, obj)
    tpr[i] = lib.tpr(detection)
    #not actually tpr, but F1-score for VJ
    print("VJ F1-score of dart" + str(i) + ": ", tpr[i])
    print("VJHT F1-score of dart" + str(i) + ": ", f1[i])

plt.close()

resultf1 = []
resulttpr = []
#Putting reslts in appropriate format for the function
for i in whichimgs:
    resultf1.append(f1[i])
    resulttpr.append(tpr[i])

# Plot a bar chart comparing f1 scores of VJ with and without Hough Transform
lib.f1bar(resulttpr, resultf1, whichimgs)
plt.waitforbuttonpress()
plt.close()
