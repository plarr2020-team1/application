import cv2
import math
import torch
import argparse
import numpy as np

from PIL import Image
from statsmodels.nonparametric.kernel_regression import KernelReg
from monodepth2.infer import infer_depth as monodepth_infer
from yolact.infer import infer_segmentation
from mannequinchallenge.infer import infer_depth as mannequin_infer

from tracktor.utils import interpolate
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage

# reference: https://www.kalmanfilter.net/kalman1d.html
def kalmanfilter(x,p,z,r):
    # p - estimate unceratininty 
    # r - measurement unceratininty ( σ2 )  
    # z - Measured System State

    # Kalman gain calculation
    K =  p/(p+r)
    # estimate current state
    x1 = x + K*(z-x)
    # update current estimate uncertainity
    p1 = (1-K)*p

    return (x1,p1)

depth_tracks = {}
depth_tracks_smoothed = {}
depth_tracks_p = {}

def merge_masks(res_img, masks, masks_im, boxes, depth_merger, depth_map, inference, scale, given_K):    
    def get_threshold(x):
        if x < 1.5: # less than 2m
            return 0
        elif x < 3: # less than 5m
            return 1
        elif x < 4: # less than 8m
            return 2
        return 3

    colors = {
        0: tuple([229, 20, 0]),
        1: tuple([250, 104, 0]),
        2: tuple([227, 200, 0]),
        3: tuple([27, 161, 226])
    }

    h = len(res_img)
    avg_human_height = 1.7  # in m
    padding = 0

    # Find the scale
    new_sum_scales = 0
    new_num_human = 0
    human_depths = []
    for i, m in enumerate(masks):
        person_depth = depth_map * np.squeeze(m, -1)
        try:
            avg_depth = person_depth[np.where(person_depth != 0)].mean()
            human_depths.append(avg_depth)
        except ValueError:
            continue
        if given_K and boxes[i][0] > padding and boxes[i][2] < h - padding:
            # scale = K(1,1) * Y / v where Y = avg human height and v is vertical pixel difference
            new_sum_scales += avg_human_height * K[1][1] / (boxes[i][2] - boxes[i][0])
            new_num_human += 1

    # Accumulate scale across frames
    if given_K and new_num_human > 1:
        sum_scales = new_sum_scales + scale['avg'] * scale['num_human']
        # Check overflow
        if abs(sum_scales) != np.inf:
            scale['num_human'] += new_num_human
            scale['avg'] = float(sum_scales) / scale['num_human']

    for i, m in enumerate(masks):
        person_depth = depth_map * np.squeeze(m, -1)
        try:
            if depth_merger == 'mean':
                avg_depth = human_depths[i]
            elif depth_merger == 'median':
                avg_depth = np.median(human_depths[i])
            else:
                raise Exception("Undefined depth_merger error!")
            x, y = int(np.where(person_depth != 0)[0].mean()), int(np.where(person_depth != 0)[1].mean())
        except ValueError:
            #invalid x, y
            continue

        if np.isnan(avg_depth):
            continue
        if given_K:
            avg_depth = avg_depth * scale['avg']
        c = colors[get_threshold(avg_depth)]
        
        CENTER = (y, x)
        res_img = cv2.circle(res_img, CENTER, int(math.e ** (-avg_depth/2) * 45), tuple([int(x) for x in c]), -1)

        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.75 * (10 - avg_depth) / 10 if inference == 'monodepth' else 0.75
        TEXT_THICKNESS = 1
        TEXT = f"{avg_depth:.2f}m"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
        cv2.putText(res_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)

        res_img = cv2.addWeighted(res_img, 1, (np.array(c) * np.concatenate([m, m, m], -1)).astype(np.uint8), 0.3, 0)
        
    return res_img

def merge_boxes(res_img, results, depth_merger, depth_map, inference):
    global depth_tracks
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
    for t, r in results.items():
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

            if t not in depth_tracks:
                depth_tracks[t] = [avg_depth]
            else: 
                depth_tracks[t].append(avg_depth)

            avg_depth_s = avg_depth
            p = 1
            if len(depth_tracks[t]) > 1:
                avg_depth_s = depth_tracks_smoothed[t][-1]
                p = depth_tracks_p[t][-1]
                
            avg_depth_s, p = kalmanfilter(avg_depth_s, p, avg_depth, 1)
                
            if t not in depth_tracks_smoothed:
                depth_tracks_smoothed[t] = [avg_depth_s]
            else: 
                depth_tracks_smoothed[t].append(avg_depth_s)
                    
            if t not in depth_tracks_p:
                depth_tracks_p[t] = [p]
            else: 
                depth_tracks_p[t].append(p)
                
        except ValueError:
            #invalid avg_depth
            continue

        c = colors[get_threshold(avg_depth)]
        
        CENTER = (x, y)

        res_img = cv2.circle(res_img, CENTER, int(math.e ** (-avg_depth/2) * 100), tuple([int(x) for x in c]), -1)

        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.8 * (10 - avg_depth) / 10 if inference == 'monodepth' else 0.8
        TEXT_THICKNESS = 1
        TEXT = f"{avg_depth:.2f}m/{avg_depth_s:.2f}m"

        text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
        text_origin = (CENTER[0] - text_size[0] // 2, CENTER[1] + text_size[1] // 2)
        cv2.putText(res_img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (255,255,255), TEXT_THICKNESS, cv2.LINE_AA)

    return res_img
# Monodepth2 assumes during training that intrinsics of all views are identical. We will make the same assumption for
# Mannequin too.
K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def get_res(img, inference, scale, tracker, depth_merger='mean', given_K=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    masks, masks_im, boxes = None, None, None
    results = None
    
    with torch.no_grad():
        if inference['name'] == 'monodepth':
            depth_map, depth_im = monodepth_infer(inference['encoder'],
                                                  inference['depth_decoder'],
                                                  inference['input_size'],
                                                  img_pil)
            depth_map = depth_map[0, 0] # * 5.4
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
                'img': transforms(img_pil).unsqueeze(0)#.cuda()
            }
            tracker.step(frame_batch)
            results = tracker.get_results()
            results = interpolate(results)

    res_img = img.copy()
    
    if tracker == None:
        return img, merge_masks(res_img, masks, masks_im, boxes, depth_merger, depth_map, inference['name'], scale, given_K)
    
    return img, merge_boxes(res_img, results, depth_merger, depth_map, inference['name'])
