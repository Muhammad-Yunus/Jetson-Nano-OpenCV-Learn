import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE DETECT OBJECT USING FLANN MATHCER + SIFT with HOMOGRAPHY

# define minimum of match found
MIN_MATCH_COUNT = 10

# load image
img1 = cv2.imread('box.png')          # queryImage
img2 = cv2.imread('box_in_scene.png') # trainImage

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

print("number of keypoint 1:", len(kp1))
print("number of keypoint 2:", len(kp2))

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# do a HOMOGRAPHY transform for all good keypoint 
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # find Homography Matrix with method RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    
    # apply perspective transform
    h,w,d = img1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1, h-1], [w-1,0] ]).reshape(-1,1,2) #tl, bl, br, tr
    dst = cv2.perspectiveTransform(pts,M)
    
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 2)

else:
    print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT) )
    matchesMask = None


# draw matches keypoint and homography area
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

plt.figure(figsize=(14,7))
plt.imshow(img3[:,:,::-1])
plt.show()