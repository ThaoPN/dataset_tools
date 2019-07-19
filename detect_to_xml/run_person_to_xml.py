import os
import sys
import glob
import json
import argparse
import datetime

from typing import Dict, Optional

import cv2
import numpy as np

from tqdm import tqdm, trange

from make_xml import make_xml
import visualization_utils as vis_util

YOLO_V3_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "darknet",
)
print(YOLO_V3_BASE_PATH)

sys.path.append(os.path.join(YOLO_V3_BASE_PATH, "python/"))

import darknet as dn

YOLO_V3_CFG_PATH = os.path.join(YOLO_V3_BASE_PATH, "cfg/yolov3-spp.cfg")
YOLO_V3_WEIGHTS_PATH = os.path.join(YOLO_V3_BASE_PATH, "yolov3-spp.weights")
YOLO_META_PATH = os.path.join(YOLO_V3_BASE_PATH, "cfg/coco.data",)

PERSON_ID = 0


class BBOX(object):
    def __init__(self, bb: dn.BOX) -> None:
        self.x = bb.x
        self.y = bb.y
        self.w = bb.w
        self.h = bb.h
        self.left = None
        self.right = None
        self.top = None
        self.bot = None

    def __str__(self) -> str:
        return "x: {}, y: {}, w: {}, h: {}".format(
            self.x, self.y, self.w, self.h
        )

    def set_lrtb(self, im_w: int, im_h: int) -> None:
        self.left = int(self.x - self.w/2.)
        self.right = int(self.x + self.w/2.)
        self.top = int(self.y - self.h/2.)
        self.bottom = int(self.y + self.h/2.)
        if self.left < 0:
            self.left = 0
        if self.right > im_w-1:
            self.right = im_w-1
        if self.top < 0:
            self.top = 0
        if self.bottom > im_h-1:
            self.bottom = im_h-1


class PersonDetector(object):
    def __init__(self) -> None:
        self.net = dn.load_net(
            YOLO_V3_CFG_PATH.encode(),
            YOLO_V3_WEIGHTS_PATH.encode(),
            0
        )
        self.meta = dn.load_meta(YOLO_META_PATH.encode())

    def detect(self,
               image: str,
               thresh: float = .5,
               hier_thresh: float = .5,
               nms: float = .45) -> list:
        if isinstance(image, str):
            im = dn.load_image(image.encode(), 0, 0)
        elif image is None:
            return []
        else:
            arr = image.transpose(2, 0, 1)
            c, h, w = arr.shape
            arr = (arr/255.0).flatten()
            data = dn.c_array(dn.c_float, arr)
            im = dn.IMAGE(w, h, c, data)
        num = dn.c_int(0)
        pnum = dn.pointer(num)
        dn.predict_image(self.net, im)
        dets = dn.get_network_boxes(
            self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms):
            dn.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            if dets[j].prob[PERSON_ID] > 0:
                bb = dets[j].bbox
                res.append((dets[j].prob[PERSON_ID], BBOX(bb)))
        res = sorted(res, key=lambda x: -x[0])  # 0 is prob
        # dn.free_image(im)  # raise double free error
        dn.free_detections(dets, num)
        return res


def detect_from_video(pd: PersonDetector,
                      save_dir: str,
                      video_path: str,
                      detect_results,
                      every_n_frames=None):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(video_name)
    for count in trange(n_frames):
        ret, frame = cap.read()
        if every_n_frames is not None and count % every_n_frames != 0:
            continue

        results = pd.detect(frame)
        if len(results) > 0:
            frame_id = "{:04d}".format(count)
            fname = "{}_{}.jpg".format(video_name, frame_id)
            cv2.imwrite(os.path.join(save_dir, fname), frame)
            for n, (prob, bb) in enumerate(results):
                h, w, _ = frame.shape
                bb.set_lrtb(w, h)
                if bb.left is None:
                    continue

                person_id = "{:02d}".format(n)
                if fname not in detect_results:
                    detect_results[fname] = dict()

                detect_results[fname].update({
                    person_id: {
                        "top": bb.top,
                        "bottom": bb.bottom,
                        "left": bb.left,
                        "right": bb.right,
                        "prob": prob,
                    }
                })

    cap.release()
    return detect_results


def convert_to_datetime(s: str) -> datetime.datetime:
    f = '%Y%m%d%H%M%S'
    return datetime.datetime.strptime(s, f)


def detect_all(base_dir: str,
               save_base_dir: str,
               every_n_frames = None) -> None:
    pd = PersonDetector()
    print(base_dir)
    print('detect all')
    print(glob.glob(os.path.join(base_dir, "*")))
    for hour_dir in glob.glob(os.path.join(base_dir, "*")):
        print(hour_dir)
        hour_dir_name = os.path.split(hour_dir)[-1]
        save_dir = os.path.join(save_base_dir, hour_dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        images_dir = os.path.join(save_dir, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        json_path = os.path.join(save_dir, "result.json")
        if os.path.exists(json_path):
            continue

        detect_results = dict()
        files = glob.glob(os.path.join(hour_dir, "*.mkv"))
        print('num of video: ', len(files))
        print(files)
        for video_path in tqdm(files):
            detect_from_video(pd, images_dir, video_path, detect_results,
                              every_n_frames)
        with open(json_path, "w") as f:
            json.dump(detect_results, f, indent=2)

        make_xml(save_dir)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir", type=str,
        help="video folder path."
    )
    parser.add_argument(
        "save_dir", type=str,
        help="save directory path",
    )
    parser.add_argument(
        "--every_n_frames", type=int,
        help="every n frame will get 1 frame",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    # detect_all(args.base_dir, args.save_dir, args.every_n_frames)
    # pd = PersonDetector()
    # detect_from_video(pd, None, args.videos, None)
    detect_all(args.base_dir, args.save_dir, args.every_n_frames)
