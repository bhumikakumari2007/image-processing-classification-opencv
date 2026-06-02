import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_path = r"C:/Users/priya/Downloads/WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg"
img = cv2.imread(image_path)

# Check if image loaded
if img is None:
    raise ValueError("Image not found. Check the file path.")

# Convert to grayscale (Canny works better on grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Convert original image to RGB for display
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(7, 4))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')

axs[1].imshow(edges, cmap='gray')
axs[1].set_title('Edge Detected Image')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
