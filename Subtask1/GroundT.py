import numpy as np
import matplotlib.pyplot as plt
import cv2

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}'.format(x, y)

name = input("Which image? (0,1,...,15) ")
type(name)
image = cv2.imread('../images/dart'+str(name)+ '.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots()
ax.set_title('Label the Ground Truth')
im = ax.imshow(gray, interpolation='none', cmap='gray')
ax.format_coord = Formatter(im)
plt.show()
#
# import time
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def tellme(s):
#     print(s)
#     plt.title(s, fontsize=16)
#     plt.draw()
#
# name = input("Which image? (0,1,...,15) ")
# image = cv2.imread('../images/dart'+str(name)+ '.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.clf()
# plt.imshow(gray, cmap='gray')
#
# tellme('You will define a square, click to begin')
#
# plt.waitforbuttonpress()
#
# while True:
#     pts = []
#     while len(pts) < 4:
#         tellme('Select 4 corners with mouse')
#         pts = np.array(plt.ginput(4, show_clicks=True))
#         if len(pts) < 4:
#             tellme('Too few points, starting over')
#             time.sleep(1)  # Wait a second
#
#
#     tellme('Happy? Key click for yes, mouse click for no')
#
#     if plt.waitforbuttonpress():
#         break
#
#     # Get rid of fill
#     for p in ph:
#         p.remove()
#
# print(pts)
