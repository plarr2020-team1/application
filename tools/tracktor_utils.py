from os import path as osp

import torch

import yaml
from tqdm import tqdm
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50

def tracker_obj(base_dir):
    tracktor = yaml.safe_load(open(f'{base_dir}/experiments/cfgs/tracktor.yaml').read())['tracktor']
    reid = yaml.safe_load(open(f"{base_dir}/{tracktor['reid_config']}"))['reid']
    # set all seeds

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(f"{base_dir}/{tracktor['obj_detect_model']}",
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(f"{base_dir}/{tracktor['reid_weights']}",
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    return tracker
    