import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(image):
    #OpenCV stores images in BGR so we have to convert to RGB to display it using matplotlib
    imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(imagergb)
    plt.show()

'''

#PART1

#Q0
#Importing image from file
img1 = cv2.imread('lena.png')
# imshow(img1)

#Q1
#Creating a 256x256 black grid
img2 = np.zeros((256,256,3), dtype=np.uint8)
#Adding "Hello World!" to the grid
cv2.putText(img2, "Hello World!", (70,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,255),1, cv2.LINE_AA)
#saving the image
cv2.imwrite("myimage.jpg",img2)

#Q2
# Creating a 256x256 red grid
img3 = np.full((256,256,3), np.array([0,0,255]), dtype=np.uint8)
#Rest done already

#Q3
img4 = np.zeros((256,256,3), dtype=np.uint8)
#Traversing the image collumns  i and rows j
for y in range(len(img4)):
    for x in range(len(img4[0])):
        img4[y][x][0]=x #setting the blue
        img4[y][x][1]=y #setting the green
        img4[y][x][2]=255-img4[y][x][1]
#imshow(img4)



#PART2


#Tresholding in greyscale 

img5 = cv2.imread('mandrill.jpg')
img6 = img5


for x in range (len(img5)):
    for y in range (len(img5[0])):
        if np.all(img5[x][y] > 128):
            img5[x][y] = np.array([225,225,225])
        else:
            img5[x][y] = np.array([0,0,0])
cv2.imwrite("tresholding128.jpg",img5)
imshow(img5)


#Default Thresholding func

print ((np.array(img6)).dtype)
_,img7 = cv2.threshold(img6,10,255,cv2.THRESH_BINARY)
imshow(img7)

'''

#Thresholding in colour (optional)

