import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE CUDA SURF (Speeded-Up Robust Features)

# load image
img = cv2.imread('butterfly.jpg')
h, w, c = img.shape

# GPU memory initialization
img_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img_GpuMat.create((w, h), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray_GpuMat.create((w, h), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel

# create CUDA SURF (Speeded-Up Robust Features) object
SURF_Detector = cv2.cuda.SURF_CUDA_create(_hessianThreshold=40000, _upright=True)

# upload to GPU memory
img_GpuMat.upload(img)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img_GpuMat, cv2.COLOR_BGR2GRAY, gray_GpuMat)

# apply CUDA SURF (Speeded-Up Robust Features) to find keypoint and descriptor
kp_GpuMat, des_GpuMat = SURF_Detector.detectWithDescriptors(gray_GpuMat, None)

# download to host memory
# Keypoint GPU Mat need to use `.downloadKeypoints()` from SURF object, 
# because it needs to deserialize a GPUMat int a std::vector<KeyPoint>
kp = SURF_Detector.downloadKeypoints(kp_GpuMat)
des = des_GpuMat.download()
# ----------------------------------------------------------------------------

# dray keypoints (detected corners by SURF)
img = cv2.drawKeypoints(img, kp, img, (255,0,0), 4)

print("descriptor shape ", des.shape)

# show result
plt.figure(figsize=(14,7))
plt.imshow(img[:,:,::-1])

plt.show()