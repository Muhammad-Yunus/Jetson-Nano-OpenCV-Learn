import cv2
import numpy as np
import matplotlib.pyplot as plt


# EXAMPLE CUDA Shi-Tomasi Corner Detection

# load image
img = cv2.imread('chessboard.png')
h, w, c = img.shape

# GPU memory initialization
img_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img_GpuMat.create((w, h), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray_GpuMat.create((w, h), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel

# create CUDA Shi-Tomasi
GoodFeature = cv2.cuda.createGoodFeaturesToTrackDetector(srcType=cv2.CV_8UC1, maxCorners=100, qualityLevel=0.001, minDistance=20)

# upload to GPU memory
img_GpuMat.upload(img)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img_GpuMat, cv2.COLOR_BGR2GRAY, gray_GpuMat)

# apply CUDA Shi-Tomasi Corner Detector
corners_GpuMat = GoodFeature.detect(gray_GpuMat)

# download to host memory
corners = corners_GpuMat.download() 

# -----------------------------------------------------------------------------------
# convert to int 64
corners = corners.astype(np.int0)

# draw circel for all detected corners (1, n, 2)
for x, y in corners[0, :]:
    cv2.circle(img, (x,y), int(0.02*img.shape[0]), (0, 0, 255), 2)

# show result
plt.figure(figsize=(14,7))
plt.imshow(img[:,:,::-1])

plt.show()