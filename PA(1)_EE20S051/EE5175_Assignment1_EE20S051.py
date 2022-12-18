#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as iread
import matplotlib.cm as cm
import math


# # Q1 Translation of the given image

# In[2]:


img = iread.imread('lena_translate.png')
h,w = img.shape
target = np.zeros((h,w))  #declaring target array to store the image after translation
plt.imshow(img, cmap=cm.gray) # testing the original image whether it displayed
print('original image')


# In[3]:


# value by which each pixel needs to be tranlated
t_x = 3.75
t_y = 4.3
i = 5
j = 5
while i<w:
    j = 5
    while j<h:
        x_s = i - t_x
        y_s = j - t_y
        x_s_new = math.floor(x_s)
        y_s_new = math.floor(y_s)
        b = y_s - y_s_new
        a = x_s - x_s_new
        #assigining target pixel intensity using bilinear interpolation
        target[i,j] = (1-a)*(1-b)*img[x_s_new,y_s_new] + (1-a)*(b)*img[x_s_new,y_s_new + 1] + (a)*(1-b)*img[x_s_new+1,y_s_new]+(a)*(b)*img[x_s_new + 1 , y_s_new + 1]
        j = j+1
    i = i+1
plt.imshow(target,cmap = cm.gray)
print('translated image')


# # Q2 Rotation of pisa tower about center

# In[4]:


img = iread.imread('pisa_rotate.png')
h,w = img.shape
#Finding the center of image
plt.imshow(img, cmap=cm.gray) # testing the original image whether it displayed
print('original image')
x_0 = math.floor(h/2) 
y_0 = math.floor(w/2)


# In[5]:


# angle by which image coordinates need to be rotated
theta = math.radians(-4) #taking angle as -4 degree
cosine = math.cos(theta)
sine = math.sin(theta)
#Initializing i and j which represents target coordinates
target = np.zeros((h,w))  #declaring target array to store the image after translation
i = 0
j = 0


# In[6]:


while i<h:
    j = 0
    while j<w:
        #Moving the pixels w.r.t center
        x =   x_0 - i
        y =   y_0 - j 
        #  rotation 
        x_s = (cosine * x) - (sine * y) 
        y_s = (cosine * y)+(sine * x) 
        x_s =  x_0 - x_s
        y_s =  y_0 - y_s 
        x_s_new = math.floor(x_s) 
        y_s_new = math.floor(y_s) 
        a = x_s - x_s_new
        b = y_s - y_s_new
        #assigining target pixel intensity using bilinear interpolation
        if x_s_new>=0 and x_s_new<(h-1) and y_s_new>=0 and y_s_new<(w-1):
            target[i,j] = (1-a)*(1-b)*img[x_s_new,y_s_new] + (1-a)*(b)*img[x_s_new,y_s_new + 1] + (a)*(1-b)*img[x_s_new+1,y_s_new]+(a)*(b)*img[x_s_new + 1 , y_s_new + 1]
        j = j+1
    i = i+1


# In[7]:


plt.imshow(target,cmap = cm.gray)


# # Q3 Scaling of the given image

# In[8]:


img = iread.imread('pisa_rotate.png') #Unable to read cells_scale image using matplotlib so using another image instead
h,w = img.shape
target = np.zeros((h,w))  #declaring target array to store the image after scaling
#plt.imshow(img, cmap=cm.gray) # testing the original image whether it displayed


# In[9]:


# value by which each pixel coordinates needs to be scaled
t_x = 0.8
t_y = 1.3
print('testing')
i = 2
j = 7
while i<h:
    j = 7
    while j<w:
        x_s = i/t_x
        y_s = j/t_y
        x_s_new = math.floor(x_s)
        y_s_new = math.floor(y_s)
        b = y_s - y_s_new
        a = x_s - x_s_new
        if(x_s_new<h-1 and y_s_new<w-1):
           #assigining target pixel intensity using bilinear interpolation
           target[i,j] = (1-a)*(1-b)*img[x_s_new,y_s_new] + (1-a)*(b)*img[x_s_new,y_s_new + 1] + (a)*(1-b)*img[x_s_new+1,y_s_new]+(a)*(b)*img[x_s_new + 1 , y_s_new + 1]
        j = j+1
    i = i+1
plt.imshow(target,cmap = cm.gray)
print('scaling the image')


# In[ ]:




