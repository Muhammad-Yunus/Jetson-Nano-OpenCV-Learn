import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE CUDA SURF + CUDA Brute Force Matcher (BFMatcher)

# load image
img1 = cv2.imread('box.png')
img2 = cv2.imread('box_in_scene.png')
h1, w1, c1 = img1.shape
h2, w2, c2 = img2.shape

# GPU memory initialization
img1_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img1_GpuMat.create((w1, h1), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
img2_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img2_GpuMat.create((w2, h2), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray1_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray1_GpuMat.create((w1, h1), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel
gray2_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray2_GpuMat.create((w2, h2), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel

# create CUDA SURF (Speeded-Up Robust Features) object
SURF_Detector = cv2.cuda.SURF_CUDA_create(_hessianThreshold=700, _upright=True)

# create CUDA BF Matcher object
BFMatcher = cv2.cuda.DescriptorMatcher_createBFMatcher()

# upload to GPU memory
img1_GpuMat.upload(img1)
img2_GpuMat.upload(img2)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img1_GpuMat, cv2.COLOR_BGR2GRAY, gray1_GpuMat)
cv2.cuda.cvtColor(img2_GpuMat, cv2.COLOR_BGR2GRAY, gray2_GpuMat)

# apply CUDA SURF (Speeded-Up Robust Features) to find keypoint and descriptor
kp1_GpuMat, des1_GpuMat = SURF_Detector.detectWithDescriptors(gray1_GpuMat, None)
kp2_GpuMat, des2_GpuMat = SURF_Detector.detectWithDescriptors(gray2_GpuMat, None)

# apply BF Matcher via KNN (output is list data in host memory, doesn't need to do .download() from device memory)
matches = BFMatcher.knnMatch(des1_GpuMat, des2_GpuMat, k=2)

# download to host memory
# Keypoint GPU Mat need to use `.downloadKeypoints()` from SURF object, 
# because it needs to deserialize a GPUMat int a std::vector<KeyPoint>
kp1 = SURF_Detector.downloadKeypoints(kp1_GpuMat)
kp2 = SURF_Detector.downloadKeypoints(kp2_GpuMat)
# ----------------------------------------------------------------------------

print("number of keypoint 1:", len(kp1))
print("number of keypoint 2:", len(kp2))

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#show result
plt.figure(figsize=(14,7))
plt.imshow(result)
plt.show()