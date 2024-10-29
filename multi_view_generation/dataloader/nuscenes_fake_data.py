import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint

from multi_view_generation.bev_utils.util import Cameras, get_fake_stage_2_data
"""接收一个边界框 bbox，并从该框的范围内生成一个随机的子边界框"""
def random_bbox(bbox):
    v = [randint(0, v) for v in bbox]
    left = min(v[0], v[2])
    upper = min(v[1], v[3])
    right = max(v[0], v[2])
    lower = max(v[1], v[3])
    return [left, upper, right, lower]

"""这种伪造的数据结构可以用于调试，特别是在实现数据管道时，用于测试代码是否正常工作而无需加载实际数据。"""
class NuScenesDatasetFake(Dataset):
    def __init__(self, stage='stage_2', cam_h=256, cam_w=256, seg_channels=21, cam_names=None, **kwargs):
        self.stage = stage
        self.cam_h = cam_h
        self.cam_w = cam_w
        self.seg_channels = seg_channels
        self.cam_names = Cameras[cam_names]
        
    def __getitem__(self, index: int):
        if self.stage == 'stage_2':
            return get_fake_stage_2_data(self.cam_h, self.cam_w, self.seg_channels, self.cam_names)
        elif self.stage == 'stage_1':
            return {
                'image': torch.randn(([self.cam_h, self.cam_w, 3]), dtype=torch.float32),
                'segmentation': torch.randn(([256, 256, 3]), dtype=torch.float32), 
                'angle': torch.pi,
                'dataset': 'nuscenes',
            }#返回了一个包含图像数据、分割信息、角度、数据集名称的字典

    def __len__(self):
        return 100
