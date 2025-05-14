import torch
import numpy as np
from sklearn.metrics import average_precision_score

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1 (torch.Tensor): First set of boxes [N, 4]
        boxes2 (torch.Tensor): Second set of boxes [M, 4]
        
    Returns:
        torch.Tensor: IoU matrix [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    
    return iou

def box_area(boxes):
    """
    Compute area of boxes
    
    Args:
        boxes (torch.Tensor): Boxes [N, 4]
        
    Returns:
        torch.Tensor: Area [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def generalized_box_iou(boxes1, boxes2):
    """
    Compute generalized IoU between two sets of boxes
    
    Args:
        boxes1 (torch.Tensor): First set of boxes [N, 4]
        boxes2 (torch.Tensor): Second set of boxes [M, 4]
        
    Returns:
        torch.Tensor: GIoU matrix [N, M]
    """
    # Calculate IoU
    iou = box_iou(boxes1, boxes2)
    
    # Calculate enclosing box area
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1]
    
    # Calculate union area
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2)
    
    # Calculate GIoU
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    
    return giou

def calculate_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) at a given IoU threshold
    
    Args:
        pred_boxes (torch.Tensor): Predicted boxes [N, 4]
        pred_scores (torch.Tensor): Prediction scores [N]
        gt_boxes (torch.Tensor): Ground truth boxes [M, 4]
        iou_threshold (float): IoU threshold
        
    Returns:
        float: AP value
    """
    if gt_boxes.shape[0] == 0:
        return 0.0
    
    if pred_boxes.shape[0] == 0:
        return 0.0
    
    # Calculate IoU between predictions and ground truth
    ious = box_iou(pred_boxes, gt_boxes)
    
    # For each prediction, find the best matching ground truth
    max_ious, _ = ious.max(dim=1)
    
    # Create binary labels: 1 if IoU > threshold, 0 otherwise
    labels = (max_ious > iou_threshold).float()
    
    # Sort by confidence
    sorted_indices = torch.argsort(pred_scores, descending=True)
    labels = labels[sorted_indices]
    
    # Calculate AP using scikit-learn
    if labels.sum() > 0:
        ap = average_precision_score(labels.cpu().numpy(), pred_scores[sorted_indices].cpu().numpy())
    else:
        ap = 0.0
    
    return ap

def calculate_position_matching_accuracy(pred_boxes, text_boxes, iou_threshold=0.5):
    """
    Calculate Position Matching Accuracy
    
    Args:
        pred_boxes (torch.Tensor): Predicted boxes [B, N, 4]
        text_boxes (torch.Tensor): Text-derived boxes [B, 4]
        iou_threshold (float): IoU threshold
        
    Returns:
        float: Position matching accuracy
    """
    batch_size = pred_boxes.shape[0]
    num_preds = pred_boxes.shape[1]
    
    # Expand text_boxes to match pred_boxes
    text_boxes = text_boxes.unsqueeze(1).expand(-1, num_preds, -1)  # [B, N, 4]
    
    # Calculate IoU between predicted boxes and text boxes
    total_matches = 0
    total_preds = 0
    
    for b in range(batch_size):
        ious = box_iou(pred_boxes[b], text_boxes[b])
        matches = (ious > iou_threshold).sum().item()
        total_matches += matches
        total_preds += num_preds
    
    # Calculate accuracy
    accuracy = total_matches / max(total_preds, 1)
    
    return accuracy

def calculate_metrics(all_preds, all_targets, config):
    """
    Calculate evaluation metrics
    
    Args:
        all_preds (list): List of predicted boxes tensors
        all_targets (list): List of target boxes tensors
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary of metrics
    """
    threshold = config['evaluation']['score_threshold']
    iou_thresholds = [0.3, 0.5, 0.7]
    
    # Initialize metrics
    metrics = {
        'mAP': 0.0,
        'AP50': 0.0,
        'AP70': 0.0,
        'P@5': 0.0,
        'R@5': 0.0,
        'ACS': 0.0  # Added Anatomical Consistency Score
    }
    
    # Unpack predictions and scores
    batch_size = len(all_preds)
    device = all_preds[0].device
    
    # Calculate metrics
    ap_sum = 0.0
    ap50_sum = 0.0
    ap70_sum = 0.0
    p_at_5_sum = 0.0
    r_at_5_sum = 0.0
    acs_sum = 0.0  # Sum of ACS scores
    
    for b in range(batch_size):
        pred_boxes = all_preds[b]
        
        # Handle different formats of all_preds
        if pred_boxes.dim() == 3:  # [N, num_queries, 4+] format
            # Extract ACS if available (should be in the last column)
            if pred_boxes.shape[-1] > 5:  # Has ACS
                acs_scores = pred_boxes[..., -1]
                pred_scores = pred_boxes[..., 0]
                pred_boxes = pred_boxes[..., 1:5]  # Use box coordinates
            else:
                pred_scores = pred_boxes[..., 0]
                pred_boxes = pred_boxes[..., 1:5]  # Use box coordinates
                acs_scores = None
        else:  # [N, 4] format
            pred_scores = torch.ones(pred_boxes.shape[0], device=device) * 0.9  # Default score
            acs_scores = None
        
        # Apply threshold
        mask = pred_scores > threshold
        pred_boxes_thresholded = pred_boxes[mask]
        pred_scores_thresholded = pred_scores[mask]
        
        # Get target boxes
        target_boxes = all_targets[b]
        
        # Calculate AP at different IoU thresholds
        ap50 = calculate_ap(pred_boxes_thresholded, pred_scores_thresholded, target_boxes, iou_threshold=0.5)
        ap70 = calculate_ap(pred_boxes_thresholded, pred_scores_thresholded, target_boxes, iou_threshold=0.7)
        
        # Calculate average AP across different IoU thresholds
        ap_values = []
        for iou_thresh in iou_thresholds:
            ap = calculate_ap(pred_boxes_thresholded, pred_scores_thresholded, target_boxes, iou_threshold=iou_thresh)
            ap_values.append(ap)
        
        mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0
        
        # Calculate Precision@5 and Recall@5
        top5_indices = torch.argsort(pred_scores, descending=True)[:5]
        top5_boxes = pred_boxes[top5_indices]
        
        if target_boxes.shape[0] > 0 and top5_boxes.shape[0] > 0:
            ious = box_iou(top5_boxes, target_boxes)
            max_ious, _ = ious.max(dim=1)
            matches = (max_ious > 0.5).sum().item()
            
            precision_at_5 = matches / 5.0
            recall_at_5 = matches / target_boxes.shape[0]
        else:
            precision_at_5 = 0.0
            recall_at_5 = 0.0
        
        # Calculate or retrieve ACS score
        if acs_scores is not None:
            # Use the ACS scores from predictions
            acs = acs_scores[mask].mean().item() if mask.sum() > 0 else 0.0
        else:
            # Default ACS if not available
            acs = 0.0
        
        # Accumulate metrics
        ap_sum += mean_ap
        ap50_sum += ap50
        ap70_sum += ap70
        p_at_5_sum += precision_at_5
        r_at_5_sum += recall_at_5
        acs_sum += acs
    
    # Calculate averages
    metrics['mAP'] = ap_sum / batch_size
    metrics['AP50'] = ap50_sum / batch_size
    metrics['AP70'] = ap70_sum / batch_size
    metrics['P@5'] = p_at_5_sum / batch_size
    metrics['R@5'] = r_at_5_sum / batch_size
    metrics['ACS'] = acs_sum / batch_size  # Average ACS
    
    return metrics 