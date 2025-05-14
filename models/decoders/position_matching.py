import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionMatchingModule(nn.Module):
    """
    Position Matching Optimization Module
    
    This module aligns the predicted bounding boxes with
    the text-derived position information.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Position matching parameters
        self.delta_r = config["loss"]["delta_r"]
        self.delta_theta = config["loss"]["delta_theta"]
        self.pixel_size = config["data"]["pixel_size"]
        
    def forward(self, pred_boxes, text_boxes):
        """
        Calculate position matching loss
        
        Args:
            pred_boxes (torch.Tensor): Predicted boxes [B, N, 4] in format [x_min, y_min, x_max, y_max]
            text_boxes (torch.Tensor): Text-derived boxes [B, 4] in format [x_min, y_min, x_max, y_max]
            
        Returns:
            torch.Tensor: Position matching loss
        """
        batch_size = pred_boxes.shape[0]
        num_queries = pred_boxes.shape[1]
        device = pred_boxes.device
        
        # Expand text_boxes to match pred_boxes shape
        text_boxes = text_boxes.unsqueeze(1).expand(-1, num_queries, -1)  # [B, N, 4]
        
        # Calculate GIoU between predicted boxes and text boxes
        giou_loss = self.generalized_box_iou_loss(pred_boxes, text_boxes)
        
        # Average loss over batch and queries
        position_loss = giou_loss.mean()
        
        return position_loss
    
    def generalized_box_iou_loss(self, boxes1, boxes2):
        """
        Calculate the Generalized IoU loss between two sets of boxes
        
        Args:
            boxes1 (torch.Tensor): First set of boxes [B, N, 4]
            boxes2 (torch.Tensor): Second set of boxes [B, N, 4]
            
        Returns:
            torch.Tensor: GIoU loss (1 - GIoU) [B, N]
        """
        # Calculate IoU
        iou, union = self.box_iou(boxes1, boxes2)
        
        # Find the smallest enclosing box
        x1_min = torch.min(boxes1[..., 0], boxes2[..., 0])
        y1_min = torch.min(boxes1[..., 1], boxes2[..., 1])
        x2_max = torch.max(boxes1[..., 2], boxes2[..., 2])
        y2_max = torch.max(boxes1[..., 3], boxes2[..., 3])
        
        # Calculate area of the smallest enclosing box
        enclosing_area = (x2_max - x1_min).clamp(min=0) * (y2_max - y1_min).clamp(min=0)
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-7)
        
        # Return loss (1 - GIoU)
        return 1 - giou
    
    def box_iou(self, boxes1, boxes2):
        """
        Calculate IoU between two sets of boxes
        
        Args:
            boxes1 (torch.Tensor): First set of boxes [B, N, 4]
            boxes2 (torch.Tensor): Second set of boxes [B, N, 4]
            
        Returns:
            tuple: (IoU [B, N], Union area [B, N])
        """
        # Calculate intersection areas
        x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
        
        # Calculate width and height of intersection
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        
        # Calculate intersection area
        intersection = w * h
        
        # Calculate area of each box
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        return iou, union
    
    def convert_polar_to_rect(self, r, theta, image_size):
        """
        Convert polar coordinates to rectangular region
        
        Args:
            r (float or torch.Tensor): Radius in pixels
            theta (float or torch.Tensor): Angle in radians
            image_size (tuple): (height, width) of the image
            
        Returns:
            torch.Tensor: [x_min, y_min, x_max, y_max] rectangular region
        """
        height, width = image_size
        center_x, center_y = width // 2, height // 2
        
        # Calculate delta r and theta in pixels
        delta_r = self.delta_r
        delta_theta_rad = math.radians(self.delta_theta)
        
        # Calculate center of the region in Cartesian coordinates
        # Adjust coordinate system: 0° at right, increasing counterclockwise
        x_center = center_x + r * torch.cos(theta)
        y_center = center_y + r * torch.sin(theta)
        
        # Calculate half-width and half-height of the region
        # Size is proportional to radius and angular uncertainty
        half_width = r * torch.sin(delta_theta_rad) + delta_r
        half_height = half_width  # Make it square for simplicity
        
        # Calculate rectangle corners
        x_min = x_center - half_width
        y_min = y_center - half_height
        x_max = x_center + half_width
        y_max = y_center + half_height
        
        # Ensure box is within image bounds
        x_min = torch.clamp(x_min, min=0, max=width-1)
        y_min = torch.clamp(y_min, min=0, max=height-1)
        x_max = torch.clamp(x_max, min=0, max=width-1)
        y_max = torch.clamp(y_max, min=0, max=height-1)
        
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    
    def position_matching_score(self, pred_boxes, text_boxes):
        """
        Calculate Position Matching Score (PMS)
        
        Args:
            pred_boxes (torch.Tensor): Predicted boxes [B, N, 4]
            text_boxes (torch.Tensor): Text-derived boxes [B, 4]
            
        Returns:
            torch.Tensor: PMS scores [B, N]
        """
        batch_size = pred_boxes.shape[0]
        num_queries = pred_boxes.shape[1]
        
        # Expand text_boxes to match pred_boxes shape
        text_boxes = text_boxes.unsqueeze(1).expand(-1, num_queries, -1)  # [B, N, 4]
        
        # Calculate IoU between predicted boxes and text-derived boxes
        iou, _ = self.box_iou(pred_boxes, text_boxes)
        
        # Calculate PMS (binary score based on IoU threshold)
        threshold = self.config["evaluation"]["pms_threshold"]
        pms = (iou > threshold).float()
        
        return pms 