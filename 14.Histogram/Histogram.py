"""
* It is a graphical representation of the tonal distribution in an image.
* Contains the number of pixels for each tonal value.

A histogram is a representation of frequency distribution. It is the basis for numerous spatial domain processing
techniques. Histogram manipulation can be used for image enhancement. Contrast is defined as the difference in
intensity between two objects in an image. If the contrast is too low, it is impossible to distinguish between
two objects, and they are seen as a single object.  Histogram equalization is a widely used contrast-enhancement
technique in image processing because of its high eﬃciency and simplicity. It is one of the sophisticated methods for
modifying the dynamic range and contrast of an image by altering that image such that its intensity histogram has the
desired shape. It can be classified into two branches as per the transformation function is used.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Transfer img in
img = cv2.imread("red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img_vis), plt.title("red_blue RGB")

print(img.shape)

# Histogram
img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist), plt.title("red_blue RGB")

# Colorful Histogram
color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):  # i = index, c = value of index
    hist = cv2.calcHist([img], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=c)

# Golden Gate
golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(golden_gate_vis)

# Mask
mask = np.zeros(golden_gate.shape[:2], np.uint8)

print("info ")
print(golden_gate.shape)
print(golden_gate.shape[:2])

plt.figure(), plt.imshow(mask, cmap="gray")

mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap="gray")

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask=mask)
plt.figure(), plt.imshow(masked_img_vis, cmap="gray")

masked_img = cv2.bitwise_and(golden_gate, golden_gate, mask=mask)
masked_img_hist = cv2.calcHist([golden_gate], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
plt.figure(), plt.plot(masked_img_hist)

# Histogram equalization
img = cv2.imread("hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray")

img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure(), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap="gray")

eq_img_hist = cv2.calcHist([eq_hist], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure(), plt.plot(eq_img_hist)

# Last example
img = cv2.imread("img1.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray"), plt.title("img in gray scale")

img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure(), plt.plot(img_hist), plt.title("img gray histogram graph")

# img equalize histogram

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap="gray"), plt.title("img equalized hist")

eq_img_hist = cv2.calcHist([eq_hist], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.figure(), plt.plot(eq_img_hist), plt.title("img equalized hist graphic")

plt.show()
