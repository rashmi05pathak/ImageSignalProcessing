import cv2
import numpy as np
import matplotlib.pyplot as plt
# Code to read images
img = cv2.imread('flower.png')
code = cv2.COLOR_BGR2RGB # Converting the BGR to RGB format
img = cv2.cvtColor(img, code)
plt.imshow(img)
plt.show()
#euclidean distance between a and b
def euclideanDist(a,b):
    return (((a[0]-b[0])**2) + ((a[1]-b[1])**2) + ((a[2]-b[2])**2))
def min(a,b,c):
    min = 0
    if (a<b):
        min = a
    else :
        min = b
    if (c<min):
        min = c
    else:
        min = min
    return min
#Checking the equality of two vectors 
def isEqual(a,b):
    if((a[0] == b[0]) and (a[1] == b[1]) and (a[2] == b[2])):
        return True
    else:
        return False
#Initial Cluster means
c1_init = [255, 0, 0]
c2_init = [0,0,0]
c3_init = [255,255,255]
clust1 = []
clust2 = []
clust3 = []
height = img.shape[0]
width = img.shape[1]
for val in range(5):
    for i in range(0,height):
       for j in range(0,width):
           #distance of the pixel from each cluster
           d1 = euclideanDist(img[i][j],c1_init) 
           d2 = euclideanDist(img[i][j],c2_init)
           d3 = euclideanDist(img[i][j],c3_init)
           m =  min(d1,d2,d3)
           if (m == d1):
              clust1.append((i,j))
           if (m == d2):
              clust2.append((i,j))
           if (m == d3):
              clust3.append((i,j))

    # Again finding the mean of each clusters
    len1 = len(clust1)
    len2 = len(clust2)
    len3 = len(clust3)
#************Mean of cluster 1******************
    sum = [0,0,0]
    for x in range(len1):
       ind = clust1[x]
       sum = sum + img[ind]
    if (len1 != 0):
       c1_init = sum/len1
#************Mean of cluster 2*******************
    sum = [0,0,0]
    for x in range(len2):
       ind = clust2[x]
       sum = sum + img[ind]
    if (len2 != 0):
       c2_init = sum/len2
#************Mean of cluster 3*********************
    sum = [0,0,0]
    for x in range(len3):
       ind = clust3[x]
       sum = sum + img[ind]
    if (len3 != 0):
       c3_init = sum/len3
    if(val == 4): #Replace initial image with c_init intensities 
        for x in range(len1):
            ind = clust1[x]
            img[ind] = c1_init
        for x in range(len2):
            ind = clust2[x]
            img[ind] = c2_init
        for x in range(len3):
            ind = clust3[x]
            img[ind] = c3_init
    clust1.clear()
    clust2.clear()
    clust3.clear()
plt.imshow(img)
plt.show()
