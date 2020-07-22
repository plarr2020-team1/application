from yolact.infer import infer_segmentation
from mannequinchallenge.infer import infer_depth as mannequin_infer
from human_depth_dataset.evaluate_depth import evaluateDepths, calculateTrueErrors
import numpy as np

def get_yolact_mask(dataset):
    all_masks = []
    for imgs in dataset:
        masks, merge_masks, boxes = infer_segmentation("yolact_plus_resnet50_54_800000.pth", np.array(imgs[0]))
        masks = masks[:, :, :, 0] if len(masks) > 0 else []
        all_masks.append(masks)
    return all_masks

def get_mannequin_depth(dataset):
    all_depth_maps = []
    for imgs in dataset:
        depth_map, depth_im = mannequin_infer(imgs[0])
        depth_map = (255 - depth_map) / 7
        all_depth_maps.append(depth_map)
    return all_depth_maps

""" It is assumed that segmentation masks and depth maps are from the same dataset. """
def get_segmentation_depth_map_stats(all_masks, all_depth_maps, dataset):
    stats = []
    for i, imgs in enumerate(dataset):
        depth_map = all_depth_maps[i]
        masks = all_masks[i]
        if len(masks) == 0:
            stats.append([])
            continue
        for mask in masks:
            predicted_depth = mask * depth_map
            gt_depth = mask * imgs[1]
            stats.append(evaluateDepths(predicted_depth, gt_depth))
    return stats