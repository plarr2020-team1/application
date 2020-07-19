import cv2
import math
import torch
import argparse
import numpy as np

from PIL import Image
from monodepth2.infer import infer_depth
from yolact.infer import infer_segmentation
from tools import get_res

import sys 
sys.path.append('./tools')

from tracktor_utils import tracker_obj
from tracktor.utils import interpolate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Social Distancing App.')
    parser.add_argument('--video_source', default=0, type=str, help='It can be a video path or webcam id.')
    parser.add_argument('--depth_merger', default='mean', type=str, help='It can be mean or median')
    parser.add_argument('--inference', default='monodepth', choices=['monodepth', 'mannequin'], type=str,
                        help='It can be monodepth or mannequin')
    parser.add_argument('--with_tracker', action='store_true', help='Tracker or YOLACT.')
    args = parser.parse_args()

    tracker = None

    if args.with_tracker:
        tracker = tracker_obj("./tracking_wo_bnw")
        tracker.reset()

    depth_merger = args.depth_merger
    video_source = args.video_source
    inference = args.inference
    try:
        video_source = int(video_source)
    except:
        pass

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) / 2
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    out = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), fps, size)

    counter = 0
    while(cap.isOpened()):
        counter += 1
        if counter % 3 != 0:
            continue
        counter = 0
        ret, frame = cap.read()
        if not ret:
            break

        img, res_img = get_res(frame, inference, tracker, depth_merger)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

        out.write(res_img)
        # im_v = cv2.vconcat([img, res_img])
        # cv2.imshow('frame', im_v)
        cv2.imshow('frame', res_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
