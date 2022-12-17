import numpy as np
from PIL import Image
import cv2
import math

#img = Image.open("Globe.png")
img = Image.open("Globe.png")

arr = np.array(img)

def conv(X, H):
    # make sure both X and H are 2-D
    assert( X.ndim == 2)
    assert( H.ndim == 2)

    # get the horizontal and vertical size of X and H
    imageColumns = X.shape[1]
    imageRows = X.shape[0]
    kernelColumns = H.shape[1]
    kernelRows = H.shape[0]

    # calculate the horizontal and vertical size of Y (assume "full" convolution)
    newRows = imageRows + kernelRows - 1
    newColumns = imageColumns + kernelColumns - 1

    # create an empty output array
    Y = np.zeros((newRows,newColumns))


    # go over output locations
    for m in range(newRows):
        for n in range(newColumns):

    # go over input locations
          for i in range(kernelRows):
              for j in range(kernelColumns):
                if (m-i >= 0) and (m-i < imageRows ) and (n-j >= 0) and (n-j < imageColumns):
                      Y[m,n] = Y[m,n] + H[i,j]*X[m-i,n-j]
        # make sure kernel is within bounds

        # calculate the convolution sum

    return Y

"********creating Gaussian kernel with given sigma**************"
#********creating Gaussian kernel with given sigma**************"
def GaussianKernal_spaceInvariant(sigma):
    s = 6*sigma + 1
    filter_size = math.ceil(s)
    b = filter_size%2
    if b == 0:
        filter_size = filter_size+1
    h = np.zeros((filter_size, filter_size), dtype = float)
    m = filter_size//2
    n = filter_size//2
    sum = 0
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            if sigma == 0.0:
                x1 = 1
                sigma = 1
            else:
                x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            h[x+m, y+n] = (1/x1)*x2
            sum = sum + h[x+m,y+n]
    #print(sum)
    h = h/sum
    
    return h
#Space Variant blurring - defining sigma for different pixels of image
def GaussianKernal_SigmaMap(arr):
    # get the horizontal and vertical size of image
    N = arr.shape[0]
    A = 2.0
    B = N*N
    B = B/10.596
    sigmaMap = np.zeros([N,N], dtype = float)
    sigmaMap[0,0] = 0.01
    #Defining sigma Map for various pixels of image
    for i in range(N-1):
        for j in range(N-1):
            sigmaMap[i,j] = A * np.exp(-((i-N/2)*(i-N/2) + (j - N/2)*(j - N/2))/B)
    return sigmaMap



def spaceVariantBlurring(arr):
#h = GaussianKernal(1.2)
#h = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
#h = np.array([[1,0,-1],[0,0,0],[-1,0,1]]) //Edge detection
#h = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]) #Edge detection
#h = np.array([[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]) #Gaussian Blurr
#####Getting Sigma Map matrix for the given input image
    sigmaMap = GaussianKernal_SigmaMap(arr)
    
    #print(sigmaMap)
    N = arr.shape[0]

    #******Nautical.png blurring experiment 2
    #sigmaMap = np.zeros([N,N], dtype = float)
   # for i in range(N-1):
       # for j in range(N-1):
          #  sigmaMap[i,j] = 1.0
    # create an empty output array
    Y = np.zeros((N,N), dtype = float)

    for i in range(N-1):
       for j in range(N-1):
           sigma = sigmaMap[i,j] # Ek Ek Sigma se Kernel banaana hai, Fir usko blurring ke liye use karna hai
           kernel = GaussianKernal_spaceInvariant(sigma)
            #### Use this kernel for blurring the arr[i,j] pixel of the image*************
           kernel = arr[i,j] * kernel #Kernel ko pixel intensity se multiply karna hai
           k_size = kernel.shape[0] ##Alag alag Sigma value ke liye kernel ka size alag alag hoga
           x = k_size//2 #center of kernel, it needs to be placed at arr[i,j]
           for m in range(-x,x):
               for n in range(-x,x):
                   if (i+m >= 0) and (i+m < N ) and (j+n >= 0) and (j+n < N):
                      Y[i+m,j+n] = Y[i+m,j+n] + kernel[m+x,n+x]
    return Y

#h = GaussianKernal_spaceInvariant(1.0) #*******Used for space Invariant Blurring
#s = conv(arr,h) #*******Used for space Invariant Blurring
##Space Variant Blurring of the input image
Y = spaceVariantBlurring(arr)
img1 = Image.fromarray(Y)
img1.show()
