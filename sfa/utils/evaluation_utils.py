"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The utils for evaluation
# Refer from: https://github.com/xingyizhou/CenterNet
"""

from __future__ import division
import os
import sys

import torch
import numpy as np
import torch.nn.functional as F
import cv2

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf
from data_process.kitti_bev_utils import drawRotatedBox


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections


def get_yaw(direction):
    # 如果是CUDA张量，先转换到CPU再转为numpy
    if hasattr(direction, 'cpu'):
        direction = direction.cpu().numpy()
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def compute_bev_box_iou(box1, box2):
    """
    两个 BEV 框的伪 IoU（中心距离启发式 + 手工分段，非旋转矩形几何交叠）。
    box format: [score, x, y, z, h, w, l, yaw]
    竞赛调参产物：dist/size < 0.2 → 0.9；< 0.4 → 0.7；否则 1.2 - d。
    """
    # Extract centers and dimensions
    x1, y1, w1, l1, yaw1 = box1[1], box1[2], box1[5], box1[6], box1[7]
    x2, y2, w2, l2, yaw2 = box2[1], box2[2], box2[5], box2[6], box2[7]

    # Compute center distance (more sensitive detection)
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Compute more conservative average size (use smaller dimension)
    min_dim1 = min(w1, l1)
    min_dim2 = min(w2, l2)
    min_size = min(min_dim1, min_dim2)
    max_size = max(w1, l1, w2, l2)
    avg_size = (w1 + l1 + w2 + l2) / 4.0

    if avg_size < 1e-6:
        return 0.0

    # Enhanced distance-based IoU calculation
    normalized_dist = dist / avg_size

    # More aggressive overlap detection for cross-class NMS
    if normalized_dist > 1.0:  # Reduced from 1.5
        return 0.0
    elif normalized_dist < 0.2:  # Reduced from 0.3
        return 0.9  # Increased from 0.8
    elif normalized_dist < 0.4:
        return 0.7
    else:
        return max(0.0, 1.2 - normalized_dist)  # More aggressive


def apply_inter_class_nms(top_preds, num_classes=3, iou_thresh=0.3):
    """
    Apply NMS across different classes to remove duplicate detections.
    More aggressive with lower threshold for better suppression.
    """
    # Collect all detections with class info
    all_dets = []
    for cls_id in range(num_classes):
        if len(top_preds[cls_id]) > 0:
            for det in top_preds[cls_id]:
                all_dets.append((det, cls_id))

    if len(all_dets) == 0:
        return top_preds

    # Sort by score (descending)
    all_dets.sort(key=lambda x: x[0][0], reverse=True)

    # Keep track of suppressed detections
    keep = [True] * len(all_dets)

    for i in range(len(all_dets)):
        if not keep[i]:
            continue
        det_i, cls_i = all_dets[i]

        for j in range(i + 1, len(all_dets)):
            if not keep[j]:
                continue
            det_j, cls_j = all_dets[j]

            # Only apply NMS between different classes
            if cls_i != cls_j:
                iou = compute_bev_box_iou(det_i, det_j)

                # Multiple suppression criteria
                suppress = False

                # Criterion 1: High overlap (traditional NMS)
                if iou > iou_thresh:
                    suppress = True

                # Criterion 2: Very close proximity (regardless of IoU)
                x1, y1 = det_i[1], det_i[2]
                x2, y2 = det_j[1], det_j[2]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                # Get average box size for distance normalization
                avg_size = (det_i[5] + det_i[6] + det_j[5] + det_j[6]) / 4.0
                if avg_size > 0:
                    normalized_dist = dist / avg_size
                    if normalized_dist < 0.25:  # Very close proximity
                        suppress = True

                    # Criterion 3: Same location with different classes (aggressive)
                    if normalized_dist < 0.1:
                        suppress = True

                if suppress:
                    # Suppress the lower-score detection
                    keep[j] = False

    # Rebuild top_preds with kept detections
    new_top_preds = {j: [] for j in range(num_classes)}
    for idx, (det, cls_id) in enumerate(all_dets):
        if keep[idx]:
            new_top_preds[cls_id].append(det)

    # Convert lists back to numpy arrays
    for j in range(num_classes):
        if len(new_top_preds[j]) > 0:
            new_top_preds[j] = np.array(new_top_preds[j])
        else:
            new_top_preds[j] = np.array([]).reshape(0, 8)

    return new_top_preds


def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2, inter_class_nms=True, nms_thresh=0.5):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :param inter_class_nms: Apply NMS across different classes to remove duplicates
    :param nms_thresh: IoU threshold for inter-class NMS
    :return:
    """
    # 如果是CUDA张量，先转换到CPU再转为numpy
    if hasattr(detections, 'cpu'):
        detections = detections.cpu().numpy()
    
    # TODO: Need to consider rescale to the original scale: x, y

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]

        # Apply inter-class NMS to remove duplicate detections across classes
        if inter_class_nms:
            top_preds = apply_inter_class_nms(top_preds, num_classes, nms_thresh)

        ret.append(top_preds)

    return ret


def draw_predictions(img, detections, num_classes=3):
    for j in range(num_classes):
        if len(detections[j]) > 0:
            for det in detections[j]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)])

    return img


def convert_det_to_real_values(detections, num_classes=3):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                z = _z + cnf.boundary['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)


# ============================================================================
# NumPy-based functions for ONNX Runtime inference
# ============================================================================

def _nms_numpy(heat, kernel=3):
    """
    NumPy version of NMS (Non-Maximum Suppression) for heatmap.
    Uses max pooling to find local maxima.
    """
    from scipy.ndimage import maximum_filter
    pad = (kernel - 1) // 2
    hmax = maximum_filter(heat, size=kernel, mode='constant')
    keep = (hmax == heat).astype(np.float32)
    return heat * keep


def _topk_numpy(scores, K=40):
    """
    NumPy version of topk operation.
    Args:
        scores: [batch, num_classes, height, width]
        K: number of top scores to keep
    Returns:
        topk_score, topk_inds, topk_clses, topk_ys, topk_xs
    """
    batch, cat, height, width = scores.shape

    # Flatten spatial dimensions and get topk for each class
    scores_flat = scores.reshape(batch, cat, -1)
    topk_inds_per_class = np.argsort(-scores_flat, axis=2)[:, :, :K]  # [batch, cat, K]
    topk_scores_per_class = np.take_along_axis(scores_flat, topk_inds_per_class, axis=2)

    # Convert flat indices to y, x coordinates
    topk_inds_per_class = topk_inds_per_class % (height * width)
    topk_ys_per_class = (topk_inds_per_class // width).astype(np.float32)
    topk_xs_per_class = (topk_inds_per_class % width).astype(np.float32)

    # Get topk across all classes
    topk_scores_flat = topk_scores_per_class.reshape(batch, -1)  # [batch, cat*K]
    topk_ind_global = np.argsort(-topk_scores_flat, axis=1)[:, :K]  # [batch, K]
    topk_score = np.take_along_axis(topk_scores_flat, topk_ind_global, axis=1)  # [batch, K]

    # Determine which class each top detection belongs to
    topk_clses = topk_ind_global // K  # [batch, K]

    # Gather the corresponding inds, ys, xs
    topk_inds_flat = topk_inds_per_class.reshape(batch, -1, 1)
    topk_ys_flat = topk_ys_per_class.reshape(batch, -1, 1)
    topk_xs_flat = topk_xs_per_class.reshape(batch, -1, 1)

    batch_indices = np.arange(batch)[:, None, None]
    topk_ind_global_expanded = topk_ind_global[:, :, None]

    topk_inds = topk_inds_flat[batch_indices, topk_ind_global_expanded, 0]  # [batch, K]
    topk_ys = topk_ys_flat[batch_indices, topk_ind_global_expanded, 0]  # [batch, K]
    topk_xs = topk_xs_flat[batch_indices, topk_ind_global_expanded, 0]  # [batch, K]

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _transpose_and_gather_feat_numpy(feat, ind):
    """
    NumPy version of transpose and gather features.
    Args:
        feat: [batch, C, H, W]
        ind: [batch, K]
    Returns:
        gathered_feat: [batch, K, C]
    """
    batch, channels, height, width = feat.shape
    K = ind.shape[1]

    # Transpose: [batch, C, H, W] -> [batch, H, W, C]
    feat = np.transpose(feat, (0, 2, 3, 1))

    # Reshape: [batch, H, W, C] -> [batch, H*W, C]
    feat = feat.reshape(batch, -1, channels)

    # Gather along spatial dimension
    batch_indices = np.arange(batch)[:, None, None]
    ind_expanded = ind[:, :, None]
    gathered_feat = feat[batch_indices, ind_expanded, np.arange(channels)]

    return gathered_feat


def decode_numpy(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    """
    NumPy version of decode function for ONNX Runtime inference.
    Args:
        hm_cen: [batch, num_classes, H, W] - center heatmap
        cen_offset: [batch, 2, H, W] - center offset
        direction: [batch, 2, H, W] - direction vector
        z_coor: [batch, 1, H, W] - z coordinate
        dim: [batch, 3, H, W] - 3D dimensions (h, w, l)
        K: number of top detections to keep
    Returns:
        detections: [batch, K, 10] numpy array
    """
    batch_size, num_classes, height, width = hm_cen.shape

    # Apply NMS on heatmap
    hm_cen = _nms_numpy(hm_cen)

    # Get topk
    scores, inds, clses, ys, xs = _topk_numpy(hm_cen, K=K)

    # Gather center offset
    if cen_offset is not None:
        cen_offset_gathered = _transpose_and_gather_feat_numpy(cen_offset, inds)
        cen_offset_gathered = cen_offset_gathered.reshape(batch_size, K, 2)
        xs = xs.reshape(batch_size, K, 1) + cen_offset_gathered[:, :, 0:1]
        ys = ys.reshape(batch_size, K, 1) + cen_offset_gathered[:, :, 1:2]
    else:
        xs = xs.reshape(batch_size, K, 1) + 0.5
        ys = ys.reshape(batch_size, K, 1) + 0.5

    # Gather other features
    direction_gathered = _transpose_and_gather_feat_numpy(direction, inds)
    direction_gathered = direction_gathered.reshape(batch_size, K, 2)

    z_coor_gathered = _transpose_and_gather_feat_numpy(z_coor, inds)
    z_coor_gathered = z_coor_gathered.reshape(batch_size, K, 1)

    dim_gathered = _transpose_and_gather_feat_numpy(dim, inds)
    dim_gathered = dim_gathered.reshape(batch_size, K, 3)

    clses = clses.reshape(batch_size, K, 1).astype(np.float32)
    scores = scores.reshape(batch_size, K, 1)

    # Concatenate all features
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    detections = np.concatenate([scores, xs, ys, z_coor_gathered,
                                  dim_gathered, direction_gathered, clses], axis=2)

    return detections


def post_processing_numpy(detections, num_classes=3, down_ratio=4, peak_thresh=0.2,
                          inter_class_nms=True, nms_thresh=0.5):
    """
    NumPy version of post_processing for ONNX Runtime inference.
    Args:
        detections: [batch_size, K, 10] numpy array
        num_classes: number of object classes
        down_ratio: downsampling ratio
        peak_thresh: confidence threshold
        inter_class_nms: whether to apply inter-class NMS
        nms_thresh: IoU threshold for inter-class NMS
    Returns:
        List of dictionaries containing detections per class
    """
    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]

        for j in range(num_classes):
            inds = (classes == j)

            # Extract and process detections for this class
            # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
            direction_vec = detections[i, inds, 7:9]
            yaw = np.arctan2(direction_vec[:, 0:1], direction_vec[:, 1:2]).astype(np.float32)

            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],  # score
                detections[i, inds, 1:2] * down_ratio,  # x
                detections[i, inds, 2:3] * down_ratio,  # y
                detections[i, inds, 3:4],  # z
                detections[i, inds, 4:5],  # h
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,  # w
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,  # l
                yaw  # yaw
            ], axis=1)

            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]

        # Apply inter-class NMS
        if inter_class_nms:
            top_preds = apply_inter_class_nms(top_preds, num_classes, nms_thresh)

        ret.append(top_preds)

    return ret
