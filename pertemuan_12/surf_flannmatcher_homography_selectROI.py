import cv2
import numpy as np
import matplotlib.pyplot as plt

# EXAMPLE VIDEO STREAM - DETECT OBJECT USING FLANN MATHCER + SURF with HOMOGRAPHY

# define minimum of match found
MIN_MATCH_COUNT = 4

# Initiate SURF detector
surf = cv2.xfeatures2d.SURF_create(200)

# FLANN parameters & Object
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture("nemo_video.mp4")

while cap.isOpened():
    # load image queryImage
    ret, img1 = cap.read()
    if not ret : 
        break
    bbox = cv2.selectROI("detected object", img1, False)
    x, y, w, h = np.int0(bbox)
    img1 = img1[y:y+h, x:x+w]
    if img1.shape[0] != 0 :
        break
    
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# find the keypoints and descriptors with SURF
kp1, des1 = surf.detectAndCompute(gray1, None)

cap = cv2.VideoCapture("nemo_video.mp4")


times = []
while cap.isOpened() :
    e1 = cv2.getTickCount()
    ret, img2 = cap.read() # trainImage
    if not ret : 
        break
    img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find the keypoints and descriptors with SURF
    kp2, des2 = surf.detectAndCompute(gray2, None)

    # apply FLANN Matcher
    matches = flann.knnMatch(des1, des2, k=2)

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
    cv2.imshow("detected object", img2)

    if (cv2.waitKey(1) == ord("q")):
        break
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

avg_time = np.array(times).mean()
print("Average processing time : %.4fs" % avg_time)
print("Average FPS : %.2f" % (1/avg_time))
cv2.destroyAllWindows()