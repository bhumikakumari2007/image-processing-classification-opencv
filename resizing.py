import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image from given path
image_path = r"C:/Users/priya/Downloads/WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise ValueError("Image not found. Check the file path.")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Scaling factors
scale_factor_1 = 3.0  
scale_factor_2 = 1/3.0

# Original dimensions
height, width = image_rgb.shape[:2]

# Zoom (enlarge)
new_height = int(height * scale_factor_1)
new_width = int(width * scale_factor_1)

zoomed_image = cv2.resize(
    image_rgb,
    (new_width, new_height),
    interpolation=cv2.INTER_CUBIC
)

# Scale down
new_height1 = int(height * scale_factor_2)
new_width1 = int(width * scale_factor_2)

scaled_image = cv2.resize(
    image_rgb,
    (new_width1, new_height1),
    interpolation=cv2.INTER_AREA
)

# Plot images
fig, axs = plt.subplots(1, 3, figsize=(10, 4))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Shape: ' + str(image_rgb.shape))

axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Shape: ' + str(zoomed_image.shape))

axs[2].imshow(scaled_image)
axs[2].set_title('Scaled Shape: ' + str(scaled_image.shape))

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
