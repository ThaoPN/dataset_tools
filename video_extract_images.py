import cv2
import os
from tqdm import tqdm, trange

INPUT_PATH = '/home/thaopn/Desktop/demo/IMG_0076_Trim1_poses_estimate.mp4'
OUTPUT_PATH = '/home/thaopn/Desktop/demo/'
EVERY_N_FRAMES = 5


cap = cv2.VideoCapture(INPUT_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)

video_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for count in trange(n_frames):
    ret, frame = cap.read()
    if count % EVERY_N_FRAMES != 0:
        continue
    frame_id = "{:05d}".format(count)
    fname = "{}_{}.jpg".format(video_name, frame_id)
    cv2.imwrite(os.path.join(OUTPUT_PATH, fname), frame)

cap.release()
