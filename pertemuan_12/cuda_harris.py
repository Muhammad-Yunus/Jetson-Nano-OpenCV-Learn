import cv2
import numpy as np
import matplotlib.pyplot as plt


# EXAMPLE CUDA Harris Corner Detection

# load image
img = cv2.imread('chessboard.png')
h, w, c = img.shape

# GPU memory initialization
img_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img_GpuMat.create((w, h), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray_GpuMat.create((w, h), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel
dst_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
dst_GpuMat.create((w, h), cv2.CV_32FC1) # cv2.CV_32FC1 -> 32 bit float image 1 channel

# create CUDA Harris COrner Detector object
HarrisCorner = cv2.cuda.createHarrisCorner(srcType=cv2.CV_8UC1, blockSize=2, ksize=3, k=0.04)

# upload to GPU memory
img_GpuMat.upload(img)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img_GpuMat, cv2.COLOR_BGR2GRAY, gray_GpuMat)

# apply CUDA Harris Corner Detector
HarrisCorner.compute(gray_GpuMat, dst_GpuMat)

# download to host memory
dst = dst_GpuMat.download() 

# -----------------------------------------------------------------------------------
# build mask image to store local maxima coordinate
mask = np.zeros((h,w), np.uint8)
mask[dst>0.05*dst.max()] = 255

# find contour from detected image corner 
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# draw cicle on detected contour
for cnt in contours :
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.circle(img, (x + w//2, y + h//2), int(0.02*img.shape[0]), (0, 255, 0), 2)

# show result
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.imshow(img[:,:,::-1])
plt.title("Detected Corner")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Corner Haris Image after Dillating (type CV_32FU1)")

plt.show()