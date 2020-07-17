import cv2
import math
import torch
import argparse
import numpy as np

from PIL import Image
from monodepth2.infer import infer_depth
from yolact.infer import infer_segmentation

def get_res(img, depth_merger='mean'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    with torch.no_grad():
        depth_map, depth_im = infer_depth("mono+stereo_1024x320", img_pil)
        masks, masks_im = infer_segmentation("yolact_plus_resnet50_54_800000.pth", img)
        depth_map = depth_map[0, 0]# * 5.4
        
    res_img = img.copy()
    
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
        TEXT_SCALE = 0.8 * (10 - avg_depth) / 10
        TEXT_THICKNESS = 1
        TEXT = f"{avg_depth:.2f}m"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
        cv2.putText(res_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)

        res_img = cv2.addWeighted(res_img, 1, (np.array(c) * np.concatenate([m, m, m], -1)).astype(np.uint8), 0.3, 0)
        
    return img, res_img