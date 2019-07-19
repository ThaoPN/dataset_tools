import cv2
from tqdm import tqdm, trange
import imutils

INPUT_PATH = '/extHDD2/gg_project/daudo/AutoLens_DauDoXe.MP4'
OUTPUT_PATH = '/extHDD2/gg_project/daudo/AutoLens_DauDoXe_resized2.MP4'
EVERY_N_FRAMES = 1


cap = cv2.VideoCapture(INPUT_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ret, frame = cap.read()
# print(frame.shape)
# frame = imutils.resize(frame, width=1280)
# print(width, height)
# print(frame.shape)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (1280, 720))


# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if frame:
#         out.write(frame)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for count in trange(n_frames):
    # if count < 150:
    #     continue
    if count > 600:
        break
    ret, frame = cap.read()
    if frame is not None:
        frame = imutils.resize(frame, width=1280)
        if count % EVERY_N_FRAMES != 0:
            continue
        if count >= 150:
            out.write(frame)

cap.release()
out.release()
