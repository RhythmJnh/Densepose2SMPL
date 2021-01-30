import colorsys
import cv2
import numpy as np
from enum import IntEnum
import math
import random
from utils.mathlib import umeyama

kpt_template = np.array([[237.065979, 139.186798],
                         [152.598572, 136.371216],
                         [217.356918, 265.887939],
                         [166.676483, 265.887939]])

kpt_template_idx = np.array([5, 6, 11, 12])

def get_transform_mat_middle(kpt, output_size, scale=0.8):
    middle_pt = [384 / 2, 512 / 2]
    dst_kpt = ((kpt_template.copy() + [0, -30] - middle_pt) * scale + middle_pt) * (output_size[1] / 512)
    kpt_M = umeyama(kpt[kpt_template_idx, :2], dst_kpt, True)[0:2]
    return kpt_M

def get_transform_mat_from_twopt(kpt, output_size_w, output_size_h, scale_ext=1.0):
    template_ = np.array(
        [
            [0.5, 0.3],
            [0.5, 0.45]
        ]
    )

    if len(kpt) == 25:
        assert kpt[2, 2] > 0 and kpt[5, 2] > 0 and kpt[9, 2] > 0 and kpt[12, 2] > 0
        src_pt1 = (kpt[2, :2] + kpt[5, :2]) / 2
        src_pt2 = (kpt[9, :2] + kpt[12, :2]) / 2
    elif len(kpt) == 17:
        assert kpt[6, 2] > 0 and kpt[5, 2] > 0 and kpt[12, 2] > 0 and kpt[11, 2] > 0
        src_pt1 = (kpt[6, :2] + kpt[5, :2]) / 2
        src_pt2 = (kpt[12, :2] + kpt[11, :2]) / 2

    template_[:, 0] *= output_size_w
    template_[:, 1] *= output_size_h

    len_dst = cv2.norm(template_[0] - template_[1])
    len_src = cv2.norm(src_pt1 - src_pt2)
    scale1 = len_dst / len_src

    valid_ = (kpt[:, 2] > 0.2)
    bbox = cv2.boundingRect(kpt[valid_, :2])
    scale2 = (output_size_w + output_size_h) * 0.5 * 0.4 / max(bbox[2], bbox[3])

    scale = min(scale1, scale2) * scale_ext

    rotate = (90 - math.degrees(math.atan2(src_pt2[1] - src_pt1[1], src_pt2[0] - src_pt1[0])) + 360) % 360
    center_src = (src_pt1 + src_pt2) / 2
    M = cv2.getRotationMatrix2D((center_src[0], center_src[1]), -rotate, scale)
    center_dst = (template_[0] + template_[1]) / 2
    M[:, 2] += (center_dst - center_src)
    return M, -rotate, scale

def get_transform_mat(person_keypoints, output_size):
    box = cv2.boundingRect(person_keypoints[np.newaxis, :, :])
    # left_shoulder = person_keypoints[5]
    # right_shoulder = person_keypoints[6]
    scale = output_size[1] * 0.7 / box[3]
    center_x = box[0] + box[2] / 2
    center_y = box[1] + box[3] / 2
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
    dst_x = output_size[0] / 2
    dst_y = output_size[1] / 2
    M[:, 2] += [dst_x - center_x, dst_y - center_y]

    return M

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def flip_kpt(person_keypoints, w_size):
    flip_kpts = np.copy(person_keypoints)
    flip_kpts[:, 0] = w_size - flip_kpts[:, 0]
    for i in range(1, 17, 2):
        left_ = np.copy(flip_kpts[i])
        right_ = np.copy(flip_kpts[i + 1])
        flip_kpts[i] = right_
        flip_kpts[i + 1] = left_
    return flip_kpts
