import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as iread
import matplotlib.cm as cm
import math

img = iread.imread('/Users/rashmipathak/Documents/2ndSemester/EE5175/EE5175_ISP_Lab1/lena_translate.png')

target = img  #declaring target array to store the image after translation

plt.imshow(img, cmap=cm.gray) # testing the original image whether it displayed

# value by which each pixel needs to be tranlated
t_x = 3.75
t_y = 4.3
print('testing')
i = 3
j = 4
while i<482:
    i = i+1
    while j<207:
        x_s = i - t_x
        y_s = j - t_y
        x_s_new = math.floor(x_s)
        y_s_new = math.floor(y_s)
        b = y_s - y_s_new
        a = x_s - x_s_new
        #assigining target pixel intensity using bilinear interpolation
        target[i,j] = (1-a)*(1-b)*img[x_s_new,y_s_new] + (1-a)*(b)*img[x_s_new,y_s_new + 1] + (a)*(1-b)*img[x_s_new+1,y_s_new]+(a)*(b)*img[x_s_new + 1 , y_s_new + 1]
        j = j+1
plt.imshow(target,cmap = cm.gray)
