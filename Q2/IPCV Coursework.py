#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import cv2

#face_classifier = cv2.CascadeClassifier('frontalface.xml')
face_classifier = cv2.CascadeClassifier('./Haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread('./images/dart14.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.1, 10, 0, (50,50), (500,500))


if faces is ():
    print('No faces found')

'''
def draw(img, obj):
    for (x,y,w,h) in obj:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        detected = cv2.imshow('Face Detection', img)
        cv2.waitKey(0)
        return detected

draw(image, faces)  
'''
    
#'''
# Draw box by iteration
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    detected = cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
#'''    

#cv2.imwrite('detected.jpg', detected)
cv2.destroyAllWindows()


# In[ ]:


help(face_classifier.detectMultiScale)

