from yolact.infer import infer_segmentation

def get_human_masks(rgb_dataset):
    masks_all_scenes = []
    for img in rgb_dataset:
        masks, _, _ = infer_segmentation("yolact_plus_resnet50_54_800000.pth", img)
        masks_all_scenes.append(masks)
    return masks_all_scenes