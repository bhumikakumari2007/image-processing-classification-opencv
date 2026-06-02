import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:/Users/priya/Downloads/WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found. Check the path.")

kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
