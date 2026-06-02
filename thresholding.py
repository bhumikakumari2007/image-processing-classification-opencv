import cv2
import matplotlib.pyplot as plt

image_path = r"C:/Users/priya/Downloads/WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check the path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(thresh, cmap='gray')
plt.title('Threshold Image')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
