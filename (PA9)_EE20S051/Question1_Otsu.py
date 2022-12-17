import cv2
import numpy as np

from matplotlib import pyplot as plt

img1 = cv2.imread('palmleaf1.pgm',0)
img2 = cv2.imread('palmleaf2.pgm',0)
def mean(arr,start,end,N):
        tot = 0
        for i in range(start,end):
            tot = tot + (i*arr[i])  # start and end are start index and end index between which mean needs to be find
        return tot/N

def count(arr,start,end):
    val = 0
    for i in range(start,end):
        val = val + arr[i]
    return val
# ***************Otsu's Thresholding******************
def Otsu(img):
    #Step 1 - Find total Mean
    histr = cv2.calcHist([img],[0],None,[256],[0,256]) # Array with Frequency count of intensity
    img1 = np.array(img)
    N = img1.size
    total = 0
    for i in range(0,255):
       total = total + (i * histr[i])
    Mean_T = total/N

    #Step 2 - Assume each intensity as the threshold and For each intensisy calculate between class variance
    # Mu1 is class-1 mean and Mu2 is class2 mean
    max = 0
    max_sigma  = 0
    for t in range(1,254):
        N1 = count(histr,0,t)
        N2 = count(histr,t+1,255)
        Mu1 = mean(histr,0,t,N1)
        Mu2 = mean(histr,t+1,255,N2)
        temp_sigma = (((Mu1 - Mean_T)*(Mu1 - Mean_T))*(N1/N)) + (((Mu2 - Mean_T)*(Mu2 - Mean_T))*(N2/N))
        if(temp_sigma>max_sigma):
           max = t
           max_sigma = temp_sigma
#************Now Thresholding the image using given t i.e. max *****************
    len = img1.shape[0]
    width = img1.shape[1]
    for i in range(0,len):
       for j in range(0,width):
          if(img1[i,j]<max):
             img1[i,j] = 0
          else:
             img1[i,j] = 1

    plt.imshow(img1, cmap = 'gray')
    plt.show()

Otsu(img1)
Otsu(img2)
