import cv2
import glob
import os

frames_folder = os.path.join(os.getcwd(), "output")
frames = glob.glob(os.path.join(frames_folder, "*.jpg"))
frames.sort()
output_video_path = os.path.join("output.mp4")

frame = cv2.imread(frames[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))

for frame in frames:
    video.write(cv2.imread(frame))

video.release()
