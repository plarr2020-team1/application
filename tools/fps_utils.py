import cv2
import torch

from PIL import Image
from tracktor.utils import interpolate
from yolact.infer import infer_segmentation
from torchvision.transforms import ToTensor, Compose, Resize, ToPILImage


def run_model(img, tracker=None):
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