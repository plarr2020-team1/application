import cv2
import torch

from PIL import Image
from tracktor.utils import interpolate
from yolact.infer import infer_segmentation
from monodepth2.infer import infer_depth as monodepth_infer
from mannequinchallenge.infer import infer_depth as mannequin_infer
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage

def run_first_phase_model(img, inference={'name': None}):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    with torch.no_grad():
        if inference['name'] == 'monodepth':
            depth_map, depth_im = monodepth_infer(inference['encoder'],
                                                  inference['depth_decoder'],
                                                  inference['input_size'],
                                                  img_pil)
        else:
            depth_map, depth_im = mannequin_infer(img_pil)

def run_second_phase_model(img, tracker=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    with torch.no_grad():
        if tracker == None:
            masks, masks_im, boxes = infer_segmentation("yolact_plus_resnet50_54_800000.pth", img)
        else:
            transforms = Compose([
                Resize((749, 1333)),
                ToTensor(),
            ])
            frame_batch = {
                'img': transforms(img_pil).unsqueeze(0).cuda()
            }
            tracker.step(frame_batch)
            results = tracker.get_results()
            results = interpolate(results)