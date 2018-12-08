import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import library as lib


# User selects which image to load.
print ("Which image(s) to load? (eg. 1 4 6 for images 1,4 and 6) ")
whichimgs = [int(x) for x in input().split() if 0<=int(x)<16]

# If only 1 image selected, show intermediary results.
show = False
if len(whichimgs) ==1:
    show = True

dart = bool(input("Detect faces(0) or dart(1)? "))
ground = {}
f1 = {}
tpr = {}

# Case for annotating all images at the start
if not show:
    for i in whichimgs:
        # Load dart(i).jpg and visually imspect the image
        image = cv2.imread('./images/dart'+str(i)+ '.jpg')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print ("dart" + str(i) + ".jpg loaded")
        if dart:
            plt.title("Note the number of dartboards to annotate.")
        else:
            plt.title("Note the number of faces to annotate.")
        plt.imshow(img)
        plt.waitforbuttonpress()
        plt.close()

        # Annotate the Ground truth
        amount = int(input("How many objects do you want to annotate?: "))
        ground[i] = lib.annotate(img, amount)

for i in whichimgs:
    # Load dart(i).jpg and visually imspect the image
    image = cv2.imread('./images/dart'+str(i)+ '.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print ("dart" + str(i) + ".jpg loaded")
    if dart:
        plt.title("Note the number of dartboards to annotate. Click to continue.")
    else:
        plt.title("Note the number of faces to annotate. Click to continue.")

    if show:
        plt.imshow(img)
        plt.waitforbuttonpress()
        plt.close()

        # Annotate the Ground truth
        amount = int(input("How many objects do you want to annotate?: "))
        ground[i] = lib.annotate(img, amount)

    # Run Viola-Jones
    if dart:
        classifier = cv2.CascadeClassifier('./Subtask2/classifier/dartcascade/cascade.xml')
    else:
        classifier = cv2.CascadeClassifier('./Subtask1/frontalface.xml')
    obj = lib.ViolaJones(i, image, classifier)

    # Compare VJ with the Ground Truth
    judgement = lib.Eval(ground[i], obj, image, thresh=0.5)

    # Calsulate F1-score and True Positive Rate
    detection = lib.getinfo(judgement, obj)
    f1[i] = lib.f1score(detection)
    tpr[i] = lib.tpr(detection)

    print("True Positive Rate of dart" + str(i) + ": ", tpr[i])
    print("F1-score of dart" + str(i) + ": ", f1[i])
