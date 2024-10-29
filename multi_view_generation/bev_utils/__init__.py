import os
from pathlib import Path
from multi_view_generation.bev_utils.util import Cameras, Dataset
from multi_view_generation.bev_utils.nuscenes_helper import CLASSES
from multi_view_generation.bev_utils.visualize import camera_bev_grid, batched_camera_bev_grid, viz_bev, argoverse_camera_bev_grid, raw_output_data_bev_grid, save_binary_as_image, return_binary_as_image

ARGOVERSE_DIR = Path(os.getenv('ARGOVERSE_DATA_DIR', 'datasets/av2')).expanduser().resolve()
NUSCENES_DIR = Path(os.getenv('NUSCENES_DATA_DIR', 'datasets/nuscenes')).expanduser().resolve()#获取nuScenes 数据集的绝对路径
SAVE_DATA_DIR = Path(os.getenv('SAVE_DATA_DIR', 'datasets')).expanduser().resolve()#保存数据的路径
