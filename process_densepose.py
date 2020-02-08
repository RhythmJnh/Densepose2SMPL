import os
import pickle
import sys
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.structures import DensePoseResult
from densepose.vis.extractor import CompoundExtractor, create_extractor, DensePoseResultExtractor
from densepose.vis.base import MatrixVisualizer

import numpy as np
import cv2


class DenseposeProcesser(object):
    def __init__(self, gpu_id=0):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(r'../configs/densepose_rcnn_R_50_FPN_s1x.yaml')
        opts = []
        opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
        opts.append(str(0.8))
        opts.append('MODEL.ROI_HEADS.NMS_THRESH_TEST')
        opts.append(str(0.2))
        opts.append('MODEL.DEVICE')
        opts.append('cuda:{}'.format(gpu_id))
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = r'../models/densepose.pth'
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        self.extractor = DensePoseResultExtractor()
        self.mask_visualizer = MatrixVisualizer(
            inplace=True, cmap=cv2.COLORMAP_PARULA, val_scale=10.625, alpha=1.0
        )
        self.u_visualizer = MatrixVisualizer(
            inplace=True, cmap=cv2.COLORMAP_PARULA, val_scale=1.0, alpha=1.0
        )
        self.v_visualizer = MatrixVisualizer(
            inplace=True, cmap=cv2.COLORMAP_PARULA, val_scale=1.0, alpha=1.0
        )


    def forward_iuv(self, image):
        outputs = self.predictor(image)['instances']
        data = self.extractor(outputs)
        iuvs  = []
        boxes = []
        for i, result_encoded_w_shape in enumerate(data.results):
            iuv_arr = DensePoseResult.decode_png_data(*result_encoded_w_shape)
            x, y, w, h = [int(val) for val in data.boxes_xywh[i]]
            boxes.append([x, y, w, h])

            iuv_arr = iuv_arr.transpose((1, 2, 0))
            put_back_iuv = np.zeros_like(image)
            put_back_iuv[y:y+h, x:x+w, :] = iuv_arr
            iuvs.append(put_back_iuv)

        return iuvs, boxes

