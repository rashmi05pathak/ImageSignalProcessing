import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mat = scipy.io.loadmat('stack.mat')
#************Definition of convolution (X is image array and H is Kernel array)***************#####
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
#**********Sum Modified Laplacian operator for a point i,j******************#########
def SML(arr,N,i,j):
    len = arr.shape[0]
    val = 0
    for x in range(i-N,i+N):
        for y in range(j-N,j+N):
            if ( i-N>=0 ) and ( i+N<len ) and ( j-N>=0 ) and ( j+N<len ):
                val = val + arr[x,y]

    return val

#******** Laplacian kernel/filter**************#
fxx = np.array([[0,0,0],[1,-2,1],[0,0,0]]) #discrete approximation for fxx
fyy = np.array([[0,1,0],[0,-2,0],[0,1,0]]) #discrete approximation for fyy
#*********Convolution with fxx of entire image stack*********************
i = 1
y = []
arr1 = [y for j in range(100)]
for x in range(100):
    if i<10:
       s = 'frame00' + str(i)
       arr1[x] = conv(mat[s],fxx)
       i = i + 1
    elif i<100:
        s = 'frame0'+str(i)
        arr1[x] = conv(mat[s],fxx)
        i = i + 1
    else:
         arr1[x] = conv(mat['frame100'],fxx)
arr1 = np.abs(arr1)
#*************Convolution with fyy of entire image stack*********************
i = 1
y = []
arr2 = [y for j in range(100)]
for x in range(100):
    if i<10:
       s = 'frame00' + str(i)
       arr2[x] = conv(mat[s],fyy)
       i = i + 1
    elif i<100:
        s = 'frame0'+str(i)
        arr2[x] = conv(mat[s],fyy)
        i = i + 1
    else:
         arr2[x] = conv(mat['frame100'],fyy)
arr2 = np.abs(arr2)
ML = arr1 + arr2 #*******Modified Laplacian*************#######
#print(ML)
y = []
sum_mod_lap = [y for j in range(100)] 
#*******Focus measure i.e SUM of Modified Laplacian*************
for x in range(100):
    arr = ML[x]
    len = arr.shape[0]
    Y = np.zeros((len,len)) #Output array
    for i in range(len):
        for j in range(len):
            Y[i,j] = SML(arr,2,i,j) #N = 0,1,2 and focus measure at point (i,j)
    sum_mod_lap[x] = Y  

    
Output = np.zeros((117,117)) #For storing the depth map of each pixel     
delta_d = 50.50  #*******
for i in range(len):
    for j in range(len):
        Fm_minus = 0
        Fm = 0
        Fm_plus = 0
        dm = 0
        for k in range(100): #****Saare frames mein same image pixel ko compare karna hai           
            F = sum_mod_lap[k] #*********For ease of calculation assigining the list  
            if (Fm < F[i,j]):
               Fm_minus = sum_mod_lap[k-1][i,j]
               Fm = F[i,j]
               if (k<99):
                  Fm_plus = sum_mod_lap[k+1][i,j]
               dm = k * delta_d
               t_k = k
            if (k == 99 and Fm > 0 and Fm_minus > 0 and Fm_plus > 0): #***** Have explored all the frames, will calculate the d-bar value here only
                dm_plus = dm + delta_d
                dm_minus = dm - delta_d
                d_bar = ((math.log(Fm) - math.log(Fm_minus))*(dm_plus * dm_plus - dm * dm) - (math.log(Fm)-math.log(Fm_plus))*(dm_minus*dm_minus - dm*dm))//(2*delta_d*(2*math.log(Fm) - math.log(Fm_plus)-math.log(Fm_minus)))
                Output[i,j] = d_bar

np. savetxt("file2.txt", Output)


   