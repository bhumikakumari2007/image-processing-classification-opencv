import cv2
import os

video_path = r"C:\Users\priya\Downloads\WhatsApp Video 2026-05-31 at 8.22.36 PM.mp4"
output_folder = "extracted_frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Could not open video file.")

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_path = os.path.join(
        output_folder,
        f"frame_{frame_count:04d}.jpg"
    )

    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()

print(f"{frame_count} frames extracted successfully.")
