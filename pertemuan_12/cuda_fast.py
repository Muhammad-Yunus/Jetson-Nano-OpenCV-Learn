import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE CUDA FAST (Features from Accelerated Segment Test) 

# load image
img = cv2.imread('butterfly.jpg')
h, w, c = img.shape

# GPU memory initialization
img_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img_GpuMat.create((w, h), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray_GpuMat.create((w, h), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel

# create CUDA FAST (Features from Accelerated Segment Test) object
FAST_Detector = cv2.cuda.FastFeatureDetector_create(threshold=40)

# upload to GPU memory
img_GpuMat.upload(img)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img_GpuMat, cv2.COLOR_BGR2GRAY, gray_GpuMat)

# apply CUDA FAST (Features from Accelerated Segment Test) to find keypoint and descriptor
kp_GpuMat, des_GpuMat = FAST_Detector.detectAndCompute(gray_GpuMat, None)

# download to host memory
# Keypoint GPU Mat need to use `.convert()` from FAST object, 
# because it needs to deserialize a GPUMat int a std::vector<KeyPoint>
kp = FAST_Detector.convert(kp_GpuMat)
des = des_GpuMat.download()
# ----------------------------------------------------------------------------

# dray keypoints (detected corners by SIFT)
img = cv2.drawKeypoints(img, kp, img, (255,0,0), 4)

print("descriptor shape ", des.shape)

# show result
plt.figure(figsize=(14,7))
plt.imshow(img[:,:,::-1])

plt.show()