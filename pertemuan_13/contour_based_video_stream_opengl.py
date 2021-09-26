import cv2
import numpy as np
import matplotlib.pyplot as plt

from gst_file import gst_file_loader
from draw_utils import draw_ped


# EXAMPLE 3.2.1 | Apply Contour Based Visual Inspection Engine for Video Stream

# Uncomment this if using OpenGL for image rendering
window_name_1 = "Detected Object"
window_name_2 = "Detected Contour"
cv2.namedWindow(window_name_1, flags=cv2.WINDOW_OPENGL)    # Window with OpenGL
cv2.namedWindow(window_name_2, flags=cv2.WINDOW_OPENGL)    # Window with OpenGL

# define range of gray color in HSV 
lower_gray = np.array([0, 0, 20])       # (part_a.jpg : [0, 0, 20] | part_b.jpg : [0, 0, 10])
upper_gray = np.array([180, 100, 150])  # (part_a.jpg : [180, 100, 150]| part_b.jpg : [180, 100, 170])

THRESHOLD_COUNT = 22 # min number of child contour  (part_a.jpg : 22 | part_b.jpg : 2)
MIN_AREA = 100 # minimum number of pixel to be counted to reject small contour

# Contour Property parameter for parent contour
MAX_ASPECT_RATIO = 0.3 # (part_a.jpg : 0.3 | part_b.jpg : 0.5)
MIN_ASPECT_RATIO = 0.1 # (part_a.jpg : 0.1 | part_b.jpg : 0.3)
MIN_EXTENT = 0.4

# Contour Property parameter for child contour
MAX_ASPECT_RATIO_CHILD = 1.5
MIN_ASPECT_RATIO_CHILD = 0.5
MIN_EXTENT_CHILD = 0.4


# load video file using GStreamer
cap = cv2.VideoCapture(gst_file_loader("video_a.mp4", useRotate90=True), cv2.CAP_GSTREAMER)  # backend GSTREAMER 

# cap = cv2.VideoCapture("video_a.mp4", cv2.CAP_FFMPEG)  # backend FFMPEG
# cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) # applicable for backend FFMPEG only

times = []
while cap.isOpened() :
    object_contour = {}
    object_count = {}
    object_id = 0 

    e1 = cv2.getTickCount()
    ret, img = cap.read()
    if not ret : 
        break 

    #convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image to get only gray colors
    mask = cv2.inRange(hsv.copy(), lower_gray, upper_gray)

    # apply eroding into mask
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 2)

    # apply bitwise operation (for background removal), if needed.
    res = cv2.bitwise_and(img, img, mask=erosion)

    # find contour from range thresholding
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt, hrcy in zip(contours, hierarchy[0]):
        # find contour Area & boungin Rect
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # calculate aspectRatio & extent
        aspectRatio = float(w)/h 
        rect_area = w*h
        extent = float(area)/rect_area
        
        # filter a small contour
        if area <= MIN_AREA:
            continue 
        
        # Find All Extreme Outer Contour [BLUE]
        if hrcy[3] == -1 :   
            if aspectRatio < MAX_ASPECT_RATIO and aspectRatio > MIN_ASPECT_RATIO and extent > MIN_EXTENT:      
                cv2.drawContours(res, cnt, -1, (255,0,0), 2)
                
                object_contour["object_%d" % object_id] = cnt # insert parent contour
                object_count["object_%d" % object_id] = 0 # set initinal count 0
                object_id += 1

        # Find All child contour [GREEN]
        if hrcy[3] != -1 :  
            if aspectRatio < MAX_ASPECT_RATIO_CHILD and aspectRatio > MIN_ASPECT_RATIO_CHILD and extent > MIN_EXTENT_CHILD:    
                cv2.drawContours(res, cnt, -1, (0,255,0), 2)

                for obj_name in object_contour:
                    # find the child contour on wich parrent contour
                    if cv2.pointPolygonTest(object_contour[obj_name], (x, y), measureDist=True) > 0 :
                        object_count[obj_name] += 1


    for obj_name in object_count:
        x, y, w, h = cv2.boundingRect(object_contour[obj_name])
        # check if number of child contour inside parrent less than threshold count 
        if object_count[obj_name] < THRESHOLD_COUNT :
            img = draw_ped(img, "%s (%d)" % (obj_name, object_count[obj_name])  , x, y, x+w, y+h, 
                        font_size=0.4, alpha=0.6, bg_color=(0,0,255), ouline_color=(0,0,255), text_color=(0,0,0))
        else :
            img = draw_ped(img, "%s (%d)" % (obj_name, object_count[obj_name])  , x, y, x+w, y+h, 
                        font_size=0.4, alpha=0.6, bg_color=(0,255,0), ouline_color=(0,255,0), text_color=(0,0,0)) 

    cv2.imshow(window_name_1, img)
    cv2.imshow(window_name_2, res)
    if cv2.waitKey(1) == ord("q"):
        break
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()
print("Average execution time : %.4fs" % time_avg)
print("Average FPS : %.2f" % (1/time_avg))

cv2.destroyAllWindows()