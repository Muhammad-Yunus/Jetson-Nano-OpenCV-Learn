import cv2
import numpy as np
import matplotlib.pyplot as plt


# EXAMPLE (CUDA) VIDEO STREAM - DETECT OBJECT USING BF MATHCER + SURF with HOMOGRAPHY
# + GSreamer (NVIDIA Accelerated element)
# + OpenGL (Optimized Image Rendering)

# create window with OpenGL enable
window_name = "Detected Object"
#cv2.namedWindow(window_name, flags=cv2.WINDOW_OPENGL)    # with OpenGL
#cv2.namedWindow(window_name)         # without OpenGL


# load GStreamer File Loader 
from gst_file import gst_file_loader

# load video file using GStreamer
cap = cv2.VideoCapture(gst_file_loader("nemo_video.mp4"), cv2.CAP_GSTREAMER)    
#cap = cv2.VideoCapture("nemo_video.mp4")

# define minimum of match found
MIN_MATCH_COUNT = 10

while cap.isOpened():
    # load image queryImage
    ret, img1 = cap.read()
    if not ret : 
        break
    bbox = cv2.selectROI(window_name, img1, False)
    x, y, w, h = np.int0(bbox)
    img1 = img1[y:y+h, x:x+w]
    if img1.shape[0] != 0 and img1.shape[1] !=0:
        break

img1 = cv2.resize(img1, (0,0), fx=2, fy=2)
h1, w1, c1 = img1.shape 

__, img2 = cap.read()
h2, w2, c2 = img2.shape
h2, w2 = h2//2, w2//2

# GPU memory initialization
img1_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img1_GpuMat.create((w1, h1), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
img2_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img2_GpuMat.create((w2, h2), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
img2_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
img2_GpuMat.create((w2, h2), cv2.CV_8UC3) # cv2.CV_8UC3 -> 8 bit image 3 channel
gray1_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray1_GpuMat.create((w1, h1), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel
gray2_GpuMat = cv2.cuda_GpuMat() # Create GpuMat object 
gray2_GpuMat.create((w2, h2), cv2.CV_8UC1) # cv2.CV_8UC1 -> 8 bit image 1 channel


# create CUDA SURF (Speeded-Up Robust Features) object
SURF_Detector = cv2.cuda.SURF_CUDA_create(_hessianThreshold=200, _upright=True)

# create CUDA BF Matcher object
BFMatcher = cv2.cuda.DescriptorMatcher_createBFMatcher()


# upload to GPU memory
img1_GpuMat.upload(img1)

# convert to grayscale using CUDA
cv2.cuda.cvtColor(img1_GpuMat, cv2.COLOR_BGR2GRAY, gray1_GpuMat)

# apply CUDA SURF (Speeded-Up Robust Features) to find keypoint and descriptor
kp1_GpuMat, des1_GpuMat = SURF_Detector.detectWithDescriptors(gray1_GpuMat, None)

# download to host memory
kp1 = SURF_Detector.downloadKeypoints(kp1_GpuMat)

cap = cv2.VideoCapture(gst_file_loader("nemo_video.mp4"), cv2.CAP_GSTREAMER)   

times =[]
while cap.isOpened() :
    e1 = cv2.getTickCount()
    ret, img2 = cap.read() # trainImage
    if not ret : 
        break
    img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    img2_GpuMat.upload(img2)
    
    cv2.cuda.cvtColor(img2_GpuMat, cv2.COLOR_BGR2GRAY, gray2_GpuMat)

    # apply CUDA SURF (Speeded-Up Robust Features) to find keypoint and descriptor
    kp2_GpuMat, des2_GpuMat = SURF_Detector.detectWithDescriptors(gray2_GpuMat, None)

    # apply BF Matcher via KNN (output is list data in host memory, doesn't need to do .download() from device memory)
    matches = BFMatcher.knnMatch(des1_GpuMat, des2_GpuMat, k=2)

    # download to host memory
    kp2 = SURF_Detector.downloadKeypoints(kp2_GpuMat)


    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # do a HOMOGRAPHY transform for all good keypoint 
    try :
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            # find Homography Matrix with method RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            
            # apply perspective transform
            h,w,d = img1.shape
            pts = np.float32([[0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2) #tl, bl, br, tr
            dst = cv2.perspectiveTransform(pts,M) # object box 
            
            # draw object box (red color)
            img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 2)
            #print( "Matches found - %d/%d" % (len(good), MIN_MATCH_COUNT) )

        else:
            #print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT) )
            matchesMask = None
    except Exception as e:
        print(e)

    # show frame
    cv2.imshow(window_name, img2)

    if (cv2.waitKey(1) == ord("q")):
        break

    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

avg_time = np.array(times).mean()
print("Average processing time : %.4fs" % avg_time)
print("Average FPS : %.2f" % (1/avg_time))
cv2.destroyAllWindows()