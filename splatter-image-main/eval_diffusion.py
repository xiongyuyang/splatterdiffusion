import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

import lpips as lpips_lib

import torch
import torchvision
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from scene.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn


bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

test_dataset = get_dataset("test", cfg.data)