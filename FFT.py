from scipy.fftpack import ifftn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

N = 30
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

xf = np.zeros((N,N))
for i in range (0,N):
    xf[15, i] = 1

Z = ifftn(xf)
ax1.imshow(xf, cmap=cm.Reds)
ax4.imshow(np.real(Z), cmap=cm.gray)

xf = np.zeros((N, N))
for i in range (0,10):
    xf[15, i] =1
for i in range (11, 20):
    xf[10,i] = 1
for i in range (21, 30):
    xf[15,i] = 1
for i in range (10,16):
    xf[i,10] = 1
    xf[i,20] = 1


Z = ifftn(xf)
ax2.imshow(xf, cmap=cm.Reds)
ax5.imshow(np.real(Z), cmap=cm.gray)

xf = np.zeros((N, N))
xf[5, 10] = 1
xf[N-5, N-10] = 1
Z = ifftn(xf)
ax3.imshow(xf, cmap=cm.Reds)
ax6.imshow(np.real(Z), cmap=cm.gray)
plt.show()
