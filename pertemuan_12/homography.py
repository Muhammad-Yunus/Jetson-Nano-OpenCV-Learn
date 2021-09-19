import cv2
import numpy as np
import matplotlib.pyplot as plt
# EXAMPLE HOMOGRAPHY TRANSFORM

# Read source image.
im_src = cv2.imread('book2.jpg')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])

# Read destination image.
im_dst = cv2.imread('book1.jpg')

# Four corners of the book in destination image.
pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))


#show result
plt.figure(figsize=(14,7))
plt.subplot(1,3,1)
plt.imshow(im_src)
plt.title("Source Image")

plt.subplot(1,3,2)
plt.imshow(im_dst)
plt.title("Destination Image")

plt.subplot(1,3,3)
plt.imshow(im_out)
plt.title("Warped Source Image")

plt.show()
