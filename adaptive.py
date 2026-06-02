import cv2
import matplotlib.pyplot as plt

image_path = r"C:/Users/priya/Downloads/WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image not found. Check the path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptive_mean = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

adaptive_gaussian = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(adaptive_mean, cmap='gray')
plt.title('Adaptive Mean')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
