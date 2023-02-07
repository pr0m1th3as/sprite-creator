## Copyright (C) 2023 Andreas Bertsatos <abertsatos@biol.uoa.gr>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

from yoloface import yolov5
from model import FullGenerator
from skimage import transform as trans
from torch.nn import functional as F

import torch
import cv2 
import numpy as np


def crop_with_ldmk(img, ldmk):
    std_ldmk = np.array([[193, 240], [319, 240],
                         [257, 314], [201, 371],
                         [313, 371]], dtype=np.float32) / 2
    tform = trans.SimilarityTransform()
    tform.estimate(ldmk, std_ldmk)
    M = tform.params[0:2, :]
    cropped = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
    return cropped


class Inference():
    def __init__(self, model_path, yolov5_path, device="cuda"):
        self.device = device
        self.G = FullGenerator(256, 512, 8, channel_multiplier=1, narrow=0.5, device=device).to(device)
        self.G.eval()
        self.G.load_state_dict(torch.load(model_path))
        self.yolonet = yolov5(yolov5_path, confThreshold=0.3, nmsThreshold=0.5, objThreshold=0.3)

    def inference(self, img_rgb):
        dets = self.yolonet.detect(img_rgb)
        dets = self.yolonet.postprocess(img_rgb, dets)
        [confidence, bbox, landmark] = dets[0]
        landmark = landmark.reshape([5, 2])
        aligned_img = crop_with_ldmk(img_rgb, landmark)
        with torch.no_grad():
            img_tensor = torch.tensor(aligned_img.copy(),
                dtype=torch.float32).to(self.device).permute(2, 0, 1)[None] / 127.5 - 1.0
            fake_img = self.G(img_tensor)
            res = (fake_img.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()[0] + 1.) * 127.5
        return res
