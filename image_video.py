import cv2

images = [
    r"C:\Users\priya\Downloads\WhatsApp Image 2026-05-20 at 2.30.48 PM.jpeg",
    r"C:\Users\priya\Downloads\WhatsApp Image 2026-05-31 at 8.19.59 PM.jpeg",
    r"C:\Users\priya\Downloads\WhatsApp Image 2026-05-31 at 8.19.58 PM.jpeg"
]

frames = []

for image in images:
    img = cv2.imread(image)
    if img is not None:
        frames.append(img)

if not frames:
    raise ValueError("No valid images found.")

height, width = frames[0].shape[:2]
frames = [cv2.resize(img, (width, height)) for img in frames]

fps = 30

video = cv2.VideoWriter(
    "image_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for i in range(len(frames) - 1):
    current = frames[i]
    next_frame = frames[i + 1]

    for _ in range(fps * 2):
        video.write(current)

    for j in range(30):
        alpha = j / 30.0
        blended = cv2.addWeighted(current, 1 - alpha, next_frame, alpha, 0)
        video.write(blended)

for _ in range(fps * 2):
    video.write(frames[-1])

video.release()

print("Video saved as image_video.mp4")
