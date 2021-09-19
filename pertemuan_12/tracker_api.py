import cv2
import numpy as np
import matplotlib.pyplot as plt

# load GStreamer File Loader 
from gst_file import gst_file_loader

# load video file using GStreamer 
cap = cv2.VideoCapture(gst_file_loader("nemo_video.mp4"), cv2.CAP_GSTREAMER)  

# Choose tracker
tracker = cv2.TrackerCSRT_create()
#tracker = cv2.TrackerKCF_create()

___, img = cap.read()
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

# create initial bounding box
bbox = cv2.selectROI("Tracking",img,False)

tracker.init(img, bbox)

while cap.isOpened():
    e1 = cv2.getTickCount()
    ret, img = cap.read()
    
    if not ret : 
        break

    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    success, bbox = tracker.update(img)

    if success:
        # draw bounding box
        x ,y ,w ,h = np.int0(bbox)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 3, 1)
        cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
    else:
        cv2.putText(img,"Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
    
    e2 = cv2.getTickCount()
    fps = cv2.getTickFrequency()/(e2-e1)
    
    cv2.putText(img,"%d FPS " % fps, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
    cv2.imshow("Tracking",img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()