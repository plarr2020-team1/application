import cv2
import math
import torch
import argparse
import numpy as np

from PIL import Image
from monodepth2.infer import infer_depth as monodepth_infer
from yolact.infer import infer_segmentation
from mannequinchallenge.infer import infer_depth as mannequin_infer

from tracktor.utils import interpolate
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage

def merge_masks(res_img, masks, masks_im, boxes, depth_merger, depth_map, inference):    
    def get_threshold(x):
        if x < 2: # less than 2m
            return 0
        elif x < 5: # less than 5m
            return 1
        elif x < 8: # less than 8m
            return 2
        return 3

    colors = {
        0: tuple([229, 20, 0]),
        1: tuple([250, 104, 0]),
        2: tuple([227, 200, 0]),
        3: tuple([27, 161, 226])
    }

    i = 0
    for m in masks:
        i+=1
        person_depth = depth_map * np.squeeze(m, -1)
        try:
            if depth_merger == 'mean':
                avg_depth = person_depth[np.where(person_depth != 0)].mean()
            elif depth_merger == 'median': 
                avg_depth = np.median(person_depth[np.where(person_depth != 0)])
            else:
                raise Exception("Undefined depth_merger error!")
            x, y = int(np.where(person_depth != 0)[0].mean()), int(np.where(person_depth != 0)[1].mean())
        except ValueError:
            #invalid avg_depth
            continue

        c = colors[get_threshold(avg_depth)]
        
        CENTER = (y, x)
        res_img = cv2.circle(res_img, CENTER, int(math.e ** (-avg_depth/2) * 100), tuple([int(x) for x in c]), -1)

        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.8 * (10 - avg_depth) / 10 if inference == 'monodepth' else 0.8
        TEXT_THICKNESS = 1
        TEXT = f"{avg_depth:.2f}m"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
        cv2.putText(res_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)

        res_img = cv2.addWeighted(res_img, 1, (np.array(c) * np.concatenate([m, m, m], -1)).astype(np.uint8), 0.3, 0)
        
    return res_img

def merge_boxes(res_img, results, depth_merger, depth_map, inference):
    def get_threshold(x):
        if x < 2: # less than 2m
            return 0
        elif x < 5: # less than 5m
            return 1
        elif x < 8: # less than 8m
            return 2
        return 3

    colors = {
        0: tuple([229, 20, 0]),
        1: tuple([250, 104, 0]),
        2: tuple([227, 200, 0]),
        3: tuple([27, 161, 226])
    }

    i = 0
    for _, r in results.items():
        x1, y1, x2, y2 = map(int, r[max(r, key=int)])
        m = np.zeros_like(depth_map)
        y1 = int(y1 * m.shape[0] / 749)
        y2 = int(y2 * m.shape[0] / 749)

        x1 = int(x1 * m.shape[1] / 1333)
        x2 = int(x2 * m.shape[1] / 1333)

        m[y1:y2, x1:x2] = 1
        person_depth = depth_map * m
        try:
            if depth_merger == 'mean':
                avg_depth = person_depth[np.where(person_depth != 0)].mean()
            elif depth_merger == 'median': 
                avg_depth = np.median(person_depth[np.where(person_depth != 0)])
            else:
                raise Exception("Undefined depth_merger error!")
            x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        except ValueError:
            #invalid avg_depth
            continue

        c = colors[get_threshold(avg_depth)]
        
        CENTER = (x, y)

        res_img = cv2.circle(res_img, CENTER, int(math.e ** (-avg_depth/2) * 100), tuple([int(x) for x in c]), -1)

        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.8 * (10 - avg_depth) / 10 if inference == 'monodepth' else 0.8
        TEXT_THICKNESS = 1
        TEXT = f"{avg_depth:.2f}m"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
        cv2.putText(res_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)

    return res_img

def get_res(img, inference, tracker, depth_merger='mean'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    masks, masks_im, boxes = None, None, None
    results = None
    
    with torch.no_grad():
        if inference == 'monodepth':
            depth_map, depth_im = monodepth_infer("mono+stereo_1024x320", img_pil)
            depth_map = depth_map[0, 0]
        else:  # mannequin
            depth_map, depth_im = mannequin_infer(img_pil)
            depth_map = (255 - depth_map) / 7

        if tracker == None:
            masks, masks_im, boxes = infer_segmentation("yolact_plus_resnet50_54_800000.pth", img)
        else:
            transforms = Compose([
                Resize((749, 1333)),
                ToTensor(),
            ])
            frame_batch = {
                'img': transforms(img_pil).unsqueeze(0)
            }
            tracker.step(frame_batch)
            results = tracker.get_results()
            results = interpolate(results)

    res_img = img.copy()
    
    if tracker == None:
        return img, merge_masks(res_img, masks, masks_im, boxes, depth_merger, depth_map, inference)

    return img, merge_boxes(res_img, results, depth_merger, depth_map, inference)
