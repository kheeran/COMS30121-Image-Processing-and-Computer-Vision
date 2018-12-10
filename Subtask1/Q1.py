import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib.pylab as pylab

pylab.rcParams['figure.figsize'] = (20,10)


def imshow(image):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imagergb)

obj_classifier = cv2.CascadeClassifier('frontalface.xml')

fig = plt.figure()

#number of test images
N=16

for i in range (0,N):
    #load image
    location = str("../images/dart") + str(i) + str(".jpg")
    image = cv2.imread(location)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #run classifier
    obj = obj_classifier.detectMultiScale(gray, 1.1, 1, 0|cv2.CASCADE_SCALE_IMAGE, (50,50), (500,500))

    #check if emppty
    if obj is ():
        print('No objects found in ' + str("dart") + str(i) + str(".jpg"))

    # Visualise classifier: Draw box by iteration
    for (x,y,w,h) in obj:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)

    saveloc = (str("faceclassified/dartclass" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc,image)
#     plot figures
    cols = 4
    rows = math.ceil(N/cols)
    ax = fig.add_subplot(rows, cols, i+1)
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(imagergb)
