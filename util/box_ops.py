# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from os import stat
import torch
from torchvision.ops.boxes import box_area
import warnings


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh3d(x):
    z0, x0, y0, z1, x1, y1 = x.unbind(-1)
    b = [(z0 + z1)/ 2., (x0 + x1) / 2., (y0 + y1) / 2.,
         (z1 - z0), (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_zxyzxy_to_cdcxcydwh(x):
    z0, x0, y0, z1, x1, y1 = x.unbind(-1)
    b = [(z0 + z1)/ 2., (x0 + x1) / 2., (y0 + y1) / 2.,
         (z1 - z0), (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_zzxxyy_to_zxyzxy(x):
    z0, z1, x0, x1, y0, y1  = x.unbind(-1)
    return torch.stack([z0, x0, y0, z1, x1, y1], dim=-1)


    

def _bbox_xxyy_to_xyxy(bbox:torch.Tensor)->torch.Tensor:
    """
    Converts boxes from [[z1, z2], x1, x2, y1, y2] to [[z1], x1, y1, [z2], x2, y2] 
    """
    xxyy = bbox.unbind(-1)
    if len(xxyy) == 6:
        # this is a 3d box
        xyxy = [xxyy[0], xxyy[2], xxyy[4], xxyy[1], xxyy[3], xxyy[5]]
    elif len(xxyy) == 4:
        # this is a 2d box
        xyxy = [xxyy[0], xxyy[2], xxyy[1], xxyy[3]]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xyxy, -1)


def _bbox_xyxy_to_xxyy(bbox:torch.Tensor)->torch.Tensor:
    """
    Converts boxes from [[z1], x1, y1, [z2], x2, y2] to [[z1, z2], x1, x2, y1, y2]
    """
    xyxy = bbox.unbind(-1)
    if len(xyxy) == 6:
        # this is a 3d box
        xxyy = [xyxy[0], xyxy[3], xyxy[1], xyxy[4], xyxy[2], xyxy[5]]
    elif len(xyxy) == 4:
        # this is a 2d box
        xxyy = [xyxy[0], xyxy[2], xyxy[1], xyxy[3]]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xxyy, -1)

def _bbox_xyxy_to_ccwh(bbox:torch.Tensor)->torch.Tensor:
    """
    Converts boxes from [[z1], x1, y1, [z2], x2, y2] to [[cz], cx, cy, [d], w, h]
    """
    xyxy = bbox.unbind(-1)
    if len(xyxy) == 6:
        # this is a 3d box
        xywh = [
            (xyxy[3] + xyxy[0]) * 0.5, # cz
            (xyxy[4] + xyxy[1]) * 0.5, # cx
            (xyxy[5] + xyxy[2]) * 0.5, # cy   
            (xyxy[3] - xyxy[0]),      # depth
            (xyxy[4] - xyxy[1]),      # width
            (xyxy[5] - xyxy[2]),      # height
        ]
    elif len(xyxy) == 4:
        # this is a 2d box
        xywh = [
            (xyxy[2] + xyxy[0]) * 0.5, # cx
            (xyxy[3] + xyxy[1]) * 0.5, # cy
            (xyxy[2] - xyxy[0]),      # width 
            (xyxy[3] - xyxy[1])       # height
        ]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xywh, -1)

def _bbox_xyxy_to_xywh(bbox:torch.Tensor) -> torch.Tensor: 
    """
    Converts boxes from [[z1], x1, y1, [z2], x2, y2] to [[z1], x1, y1, [d], w, h]
    """
    xyxy = bbox.unbind(-1)
    if len(xyxy) == 6:
        # this is a 3d box
        xywh = [
            xyxy[0], xyxy[1], xyxy[2],  # z1, x1, y1   
            (xyxy[3] - xyxy[0]),        # depth
            (xyxy[4] - xyxy[1]),        # width
            (xyxy[5] - xyxy[2]),        # height
        ]
    elif len(xyxy) == 4:
        # this is a 2d box
        xywh = [
            xyxy[0], xyxy[1],           # x1, y1
            (xyxy[2] - xyxy[0]),        # width 
            (xyxy[3] - xyxy[1])         # height
        ]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xywh, -1)

def _bbox_xywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Converts boxes from [[z1], x1, y1, [d], w, h] to [[z1], x1, y1, [z2], x2, y2]
    """
    xywh = bbox.unbind(-1)
    if len(xywh) == 6:
        # this is a 3d box
        xyxy = [
            xywh[0], xywh[1], xywh[2],  # z1, y1, x1   
            (xywh[3] + xywh[0]),        # z2
            (xywh[4] + xywh[1]),        # y2
            (xywh[5] + xywh[2]),        # x2
        ]
    elif len(xywh) == 4:
        # this is a 2d box
        xyxy = [
            xywh[0], xywh[1],           # x1, y1
            (xywh[2] + xywh[0]),        # x2 
            (xywh[3] + xywh[1])         # y2
        ]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xyxy, -1)

def _bbox_ccwh_to_xyxy(bbox:torch.Tensor)->torch.Tensor:
    """
    Converts boxes from [[cz], cx, cy, [d], w, h] to [[z1], x1, y1, [z2], x2, y2]
    """

    ccwh = bbox.unbind(-1)
    if len(ccwh) == 6:
        # this is a 3d box
        xyxy = [
            ccwh[0] - ccwh[3] * 0.5, # z1
            ccwh[1] - ccwh[4] * 0.5, # y1
            ccwh[2] - ccwh[5] * 0.5, # x1   
            ccwh[0] + ccwh[3] * 0.5, # z2
            ccwh[1] + ccwh[4] * 0.5, # y2
            ccwh[2] + ccwh[5] * 0.5, # x2   
        ]
    elif len(ccwh) == 4:
        # this is a 2d box
        xyxy = [
            ccwh[0] - ccwh[2] * 0.5, # y1
            ccwh[1] - ccwh[3] * 0.5, # x1   
            ccwh[0] + ccwh[2] * 0.5, # y2
            ccwh[1] + ccwh[3] * 0.5, # x2          
        ]
    else: 
        raise ValueError("Not supported!")
    return torch.stack(xyxy, -1)


CONVERT = {
    "xxyy_to_xyxy": _bbox_xxyy_to_xyxy,
    "xyxy_to_xxyy": _bbox_xyxy_to_xxyy,
    "xyxy_to_ccwh": _bbox_xyxy_to_ccwh,
    "xyxy_to_xywh": _bbox_xyxy_to_xywh,
    "xywh_to_xyxy": _bbox_xywh_to_xyxy,
    "ccwh_to_xyxy": _bbox_ccwh_to_xyxy,
}

def convert(bbox:torch.Tensor, from_fmt:str, to_fmt:str)->torch.Tensor:
    assert {from_fmt, to_fmt}.issubset({"xyxy", "xxyy", "xywh", "ccwh"})
    formatter = CONVERT.get(f"{from_fmt}_to_{to_fmt}")
    
    if from_fmt == to_fmt:
        return bbox

    if formatter is None:
        bbox = convert(bbox, from_fmt, "xyxy")
        return convert(bbox, "xyxy", to_fmt)
    
    return formatter(bbox)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
