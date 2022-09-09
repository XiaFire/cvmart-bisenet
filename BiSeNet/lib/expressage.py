#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset


class Expressage(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train', cache_images=False):
        super(Expressage, self).__init__(dataroot, annpath, trans_func=trans_func, mode=mode, cache_images=cache_images)
        self.n_cats = 2 
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), 
            std=(0.27469736, 0.27012361, 0.28515933),
        )
