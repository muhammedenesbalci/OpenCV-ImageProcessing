import cv2
import numpy as np

# Trransfer In
img = cv2.imread("lenna.png")
cv2.imshow("Original", img)

# Horizontal
hor = np.hstack((img, img))
cv2.imshow("Horizontal", hor)

# Vertical
ver = np.vstack((img, img))
cv2.imshow("Vertical", ver)
