from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

f = loadmat('FRIS_999_whatever.mat')
x = f['MUA']['coordinates'][0][0][:,0]
y = f['MUA']['coordinates'][0][0][:,1]
connectivity = f['MUA']['connectivity'][0][0]-1
h = f['h'][:,0]
plt.tricontourf(x,y,connectivity,h)
plt.show()
