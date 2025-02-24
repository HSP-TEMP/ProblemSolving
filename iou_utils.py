import torch
from torch import Tensor
import math


def compute_iou(boxA, boxB) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes (box_A, box_B)
    Two bounding boxes follow the form of [x_min, y_min, x_max, y_max].
    Here, we are assuming normalized coordinates [0, 1]
    Args:
        box_A (list | np.array): Predicted bounding box [x_min, y_min, x_max, y_max]
        box_B (list | np.array): Ground truth bounding box [x_min, y_min, x_max, y_max]
    Returns:
        result (float): IoU value between two bounding boxes
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_width  = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height
    
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = inter_area / (boxA_area + boxB_area - inter_area + 1e-6)
    return iou


def generalized_iou_loss(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """
    Calculate generalized IoU loss between two normalized bounding boxes
    Args:
        pred_boxes (Tensor): Predicted bounding box, shape is [N, 4]
        gt_boxes (Tensor): Ground truth bounding box, shape is [N, 4]
    Returns:
        result (Tensor): mean GIoU loss
    """
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes.unbind(dim=1)
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes.unbind(dim=1)
    
    # Calculate intersection coordinates
    inter_xmin = torch.max(pred_xmin, gt_xmin)
    inter_ymin = torch.max(pred_ymin, gt_ymin)
    inter_xmax = torch.min(pred_xmax, gt_xmax)
    inter_ymax = torch.min(pred_ymax, gt_ymax)
    
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    area_inter = inter_w * inter_h
    
    # Region of each box
    area_pred = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    area_gt   = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    union = area_pred + area_gt - area_inter + 1e-6
    
    iou = area_inter / union
    
    # Enclosing box (minimum box including two boxes
    enc_xmin = torch.min(pred_xmin, gt_xmin)
    enc_ymin = torch.min(pred_ymin, gt_ymin)
    enc_xmax = torch.max(pred_xmax, gt_xmax)
    enc_ymax = torch.max(pred_ymax, gt_ymax)
    enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin) + 1e-6 # Avoid ZeroDivisionException
    
    giou = iou - (enc_area - union) / enc_area
    loss = 1 - giou # GIoU loss = 1 - GIoU
    return loss.mean()


def ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """
    Calculate CIoU loss between two normalized bounding boxes
    Args:
        pred_boxes (Tensor): Predicted bounding box, shape is [N, 4]
        gt_boxes (Tensor): ground truth bounding box, shape is [N, 4]
    Returns:
        result (Tensor): Mean CIoU loss
    """
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes.unbind(dim=1)
    gt_xmin,   gt_ymin,   gt_xmax,   gt_ymax   = gt_boxes.unbind(dim=1)
    
    inter_xmin = torch.max(pred_xmin, gt_xmin)
    inter_ymin = torch.max(pred_ymin, gt_ymin)
    inter_xmax = torch.min(pred_xmax, gt_xmax)
    inter_ymax = torch.min(pred_ymax, gt_ymax)
    
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    area_inter = inter_w * inter_h
    
    area_pred = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    area_gt   = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    union = area_pred + area_gt - area_inter + 1e-6
    
    iou = area_inter / union  # IoU
    
    # Enclosing box
    enc_xmin = torch.min(pred_xmin, gt_xmin)
    enc_ymin = torch.min(pred_ymin, gt_ymin)
    enc_xmax = torch.max(pred_xmax, gt_xmax)
    enc_ymax = torch.max(pred_ymax, gt_ymax)
    enc_w = (enc_xmax - enc_xmin)
    enc_h = (enc_ymax - enc_ymin)
    c2 = enc_w * enc_w + enc_h * enc_h + 1e-6  # Square of diagonal length
    
    # Center
    pred_cx = (pred_xmin + pred_xmax) / 2
    pred_cy = (pred_ymin + pred_ymax) / 2
    gt_cx = (gt_xmin + gt_xmax) / 2
    gt_cy = (gt_ymin + gt_ymax) / 2

    rho2 = (pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2
    
    # Normalized center distance
    distance_term = rho2 / c2
    
    # Ratio of width, height
    pred_w = pred_xmax - pred_xmin
    pred_h = pred_ymax - pred_ymin
    gt_w   = gt_xmax - gt_xmin
    gt_h   = gt_ymax - gt_ymin
    
    # arctan
    atan_pred = torch.atan(pred_w / (pred_h + 1e-6))
    atan_gt   = torch.atan(gt_w / (gt_h + 1e-6))
    v = (4 / (math.pi ** 2)) * (atan_gt - atan_pred)**2
    alpha = v / (1 - iou + v + 1e-6)
    
    ciou = iou - (distance_term + alpha * v)
    loss = 1 - ciou
    return loss.mean()
