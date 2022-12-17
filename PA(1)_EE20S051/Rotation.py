#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as iread
import matplotlib.cm as cm
import math

img = iread.imread('/Users/rashmipathak/Documents/2ndSemester/EE5175/EE5175_ISP_Lab1/pisa_rotate.png')

target = img  #declaring target array to store the image after translation

# angle by which image coordinates need to be rotated
theta = math.pi/45 #taking angle as 4 degree


# In[93]:


#Finding the center of image
x_0 = 241 
y_0 = 103
i = 0
j = 0
while i<482:
    i = i+1
    while j<207:
        # doing translation along center then rotation then translation
        x_s = math.cos(theta)*(i - x_0) - math.sin(theta)*(j - y_0) + x_0
        y_s = math.sin(theta)*(i - x_0) + math.cos(theta)*(j - y_0) + y_0
        x_s_new = math.floor(x_s)
        y_s_new = math.floor(y_s)
        b = y_s - y_s_new
        a = x_s - x_s_new
        #assigining target pixel intensity using bilinear interpolation
        target[i,j] = (1-a)*(1-b)*img[x_s_new,y_s_new] + (1-a)*(b)*img[x_s_new,y_s_new + 1] + (a)*(1-b)*img[x_s_new+1,y_s_new]+(a)*(b)*img[x_s_new + 1 , y_s_new + 1]
        j = j+1
#plt.imshow(target,cmap = cm.gray)
plt.imshow(img,cmap = cm.gray)


# In[ ]:





# In[ ]:





# In[ ]:




