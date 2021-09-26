import cv2
import numpy as np
import matplotlib.pyplot as plt

from gst_file import gst_file_loader

# EXAMPLE Play Video Stream + OpenGL Image Rendering

window_name = "Window"
cv2.namedWindow(window_name, flags=cv2.WINDOW_OPENGL)    # Window with OpenGL

# load video file using GStreaqmer
cap = cv2.VideoCapture(gst_file_loader("video_a.mp4"), cv2.CAP_GSTREAMER)  # backend GSTREAMER

# cap = cv2.VideoCapture("video_a.mp4", cv2.CAP_FFMPEG)  # backend FFMPEG
# cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1) # applicable for backend FFMPEG only

times = []
while cap.isOpened() : 
    e1 = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret : 
        break 

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == ord("q"):
        break
    e2 = cv2.getTickCount()
    times.append((e2 - e1)/ cv2.getTickFrequency())

time_avg = np.array(times).mean()
print("Average execution time : %.4fs" % time_avg)
print("Average FPS : %.2f" % (1/time_avg))

cv2.destroyAllWindows()