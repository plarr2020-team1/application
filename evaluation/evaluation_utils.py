from yolact.infer import infer_segmentation
from mannequinchallenge.infer import infer_depth as mannequin_infer
from monodepth2.infer import load_model, infer_depth
from human_depth_dataset.evaluate_depth import evaluateDepths, \
    calculateTrueErrors
import numpy as np


def get_yolact_mask(dataset):
    all_masks = {}
    for data in dataset:
        masks, merge_masks, boxes = infer_segmentation(
            "yolact_plus_resnet50_54_800000.pth", np.array(data['rgb']))
        masks = masks[:, :, :, 0] if len(masks) > 0 else []
        all_masks[data['index']] = masks
    return all_masks


def get_mannequin_depth(dataset):
    all_depth_maps = {}
    for data in dataset:
        depth_map, depth_im = mannequin_infer(data['rgb'])
        depth_map = (255 - depth_map) / 7
        all_depth_maps[data['index']] = depth_map
    return all_depth_maps


def get_monodepth_depth(dataset):
    model_name = 'mono+stereo_1024x320'
    encoder, depth_decoder, input_size = load_model(model_name)

    depth_maps = {}
    for data in dataset:
        depth_map, _ = infer_depth(encoder, depth_decoder, input_size,
                                   data['rgb'])
        depth_maps[data['index']] = depth_map

    return depth_maps


""" It is assumed that segmentation masks and depth maps are from the same dataset. """
def get_segmentation_depth_map_stats(all_masks, all_depth_maps, dataset):
    stats = []
    for i, data in enumerate(dataset):
        depth_map = np.squeeze(all_depth_maps[data['index']])
        masks = all_masks[data['index']]['mask']
        if len(masks) == 0:
            stats.append([])
            continue
        for mask in masks:
            mask = np.squeeze(mask)
            predicted_depth = mask * depth_map
            gt_depth = mask * data['depth']
            stats.append(evaluateDepths(predicted_depth, gt_depth))
    return stats
