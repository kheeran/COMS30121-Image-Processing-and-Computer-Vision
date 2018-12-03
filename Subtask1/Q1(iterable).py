import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import time

def imshow(image):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imagergb)
    plt.show()

face_classifier = cv2.CascadeClassifier('frontalface.xml')
# face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

for i in range (0,16):

    location = str("../images/dart") + str(i) + str(".jpg")
    imgcol = cv2.imread(location)
    img = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(img, 1.1, 1, 0, (30,30), (200,200))


    if faces is ():
        print('No faces found')
    # Draw box by iteration

    for (x,y,w,h) in faces:
        cv2.rectangle(imgcol, (x,y), (x+w,y+h), (0,255,0), 2)
    # imshow (imgcol)



    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

    (M,N) = sobelx.shape
    grad = np.zeros((M,N))
    direc = np.zeros((M,N))
    thresh = 2*np.sum(img)/(N*M) #the higher the quicker, but risk not representing edges

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


    for (X,Y,W,H) in faces:
        for y in range (Y,Y+H):
            for x in range (X,X+W):
                grad[y,x] = 0

    saveloc = (str("nofaceedge/dartnofaceedge" + str(i) + str(".jpg")))
    cv2.imwrite(saveloc,grad)
    plt.imshow(grad, cmap='gray')
    print (str(i+1) + "image(s) done")
