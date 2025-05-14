import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import math

from .encoders.ptpe import PolarTextPositionEncoder
from .decoders.anatomy_constraint import AnatomicalConstraintModule
from .decoders.position_matching import PositionMatchingModule

class PolarDETR(nn.Module):
    """
    PolarDETR: Anatomical Entity Detection and Localization in Dental Images
    
    This model integrates DETR-based object detection with anatomical constraints
    and polar text-position encoding for precise localization based on
    textual descriptions.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Image encoder (backbone)
        self.build_image_encoder()
        
        # Text encoder and polar position encoding
        self.ptpe = PolarTextPositionEncoder(config)
        
        # DETR transformer decoder
        self.build_transformer_decoder()
        
        # Anatomical constraint module
        self.anatomy_module = AnatomicalConstraintModule(config)
        
        # Position matching module
        self.position_module = PositionMatchingModule(config)
        
        # Output heads
        hidden_dim = config["model"]["decoder"]["dim"]
        self.class_embed = nn.Linear(hidden_dim, 1)  # Binary detection for now
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4D box coordinates
        
        # Loss weights
        self.lambda_detr = config["loss"]["lambda_detr"]
        self.lambda_anatomy = config["loss"]["lambda_anatomy"]
        self.lambda_position = config["loss"]["lambda_position"]
    
    def build_image_encoder(self):
        """Build the image encoder (backbone)"""
        backbone_name = self.config["model"]["image_encoder"]["name"]
        pretrained = self.config["model"]["image_encoder"]["pretrained"]
        
        if backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048  # ResNet50 feature dimension
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048  # ResNet101 feature dimension
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if specified
        if self.config["model"]["image_encoder"]["freeze_backbone"]:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature projection
        hidden_dim = self.config["model"]["decoder"]["dim"]
        self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)
    
    def build_transformer_decoder(self):
        """Build the transformer decoder (DETR-style)"""
        hidden_dim = self.config["model"]["decoder"]["dim"]
        nheads = self.config["model"]["decoder"]["nhead"]
        num_decoder_layers = self.config["model"]["decoder"]["num_decoder_layers"]
        dim_feedforward = hidden_dim * 4  # Standard practice
        dropout = 0.1  # Default dropout rate
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Object queries
        num_queries = self.config["model"]["decoder"]["num_queries"]
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
    
    def forward(self, images, texts, target_boxes=None):
        """
        Forward pass
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            texts (list): List of text descriptions
            target_boxes (torch.Tensor, optional): Target boxes for training [B, M, 4]
            
        Returns:
            dict: Dictionary with outputs and losses
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Extract image features
        features = self.backbone(images)  # [B, C, H/32, W/32]
        features = self.input_proj(features)  # [B, hidden_dim, H/32, W/32]
        
        # Generate position encoding for transformer
        h, w = features.shape[-2:]
        pos_embed = self.positional_encoding(h, w, device)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, 1, C]
        
        # Flatten feature map
        features_flatten = features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Extract region features for anatomical constraint
        region_features = self.anatomy_module.extract_region_features(features, (h, w))
        
        # Generate queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Apply polar text-position encoding to enhance queries
        if self.config["model"]["ptpe"]["enabled"]:
            query_embed = self.ptpe(texts, query_embed)
        
        # Transformer decoder (DETR-style)
        # Convert to shape expected by transformer
        features_flatten = features_flatten.permute(1, 0, 2)  # [H*W, B, C]
        query_embed = query_embed.permute(1, 0, 2)  # [num_queries, B, C]
        
        # Decoder outputs
        decoder_output = self.transformer_decoder(
            query_embed,
            features_flatten,
            memory_key_padding_mask=None,
            pos=pos_embed
        )  # [num_queries, B, C]
        
        # Convert back to batch-first
        decoder_output = decoder_output.permute(1, 0, 2)  # [B, num_queries, C]
        
        # Apply anatomical constraint
        if self.config["model"]["anatomy"]["enabled"]:
            decoder_output, anatomy_loss = self.anatomy_module(region_features, decoder_output)
        else:
            anatomy_loss = torch.tensor(0.0, device=device)
        
        # Output heads
        outputs_class = self.class_embed(decoder_output).sigmoid()  # [B, num_queries, 1]
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4]
        
        # Calculate ACS even during training for monitoring
        acs = self.anatomy_module.anatomical_consistency_score(outputs_coord, (images.shape[2], images.shape[3]))
        
        # Calculate losses if training
        losses = {}
        if target_boxes is not None:
            # DETR loss (class + box)
            detr_loss = self.compute_detr_loss(outputs_class, outputs_coord, target_boxes)
            losses["detr_loss"] = detr_loss
            
            # Anatomical constraint loss
            losses["anatomy_loss"] = anatomy_loss
            
            # Position matching loss
            if self.config["model"]["ptpe"]["enabled"]:
                # Convert text to boxes
                text_boxes = []
                for i, text in enumerate(texts):
                    entities = self.ptpe.extract_entities(text)
                    r, theta = self.ptpe.map_to_polar_coordinates(entities)
                    text_box = self.ptpe.convert_polar_to_rect(r, theta, (images.shape[2], images.shape[3]))
                    text_boxes.append(text_box)
                
                text_boxes = torch.stack(text_boxes).to(device)  # [B, 4]
                position_loss = self.position_module(outputs_coord, text_boxes)
                losses["position_loss"] = position_loss
            else:
                losses["position_loss"] = torch.tensor(0.0, device=device)
            
            # Total loss
            losses["total_loss"] = (
                self.lambda_detr * losses["detr_loss"] +
                self.lambda_anatomy * losses["anatomy_loss"] +
                self.lambda_position * losses["position_loss"]
            )
        
        # Prepare output dictionary
        output = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "acs": acs,  # Include ACS in output during both training and inference
            "losses": losses
        }
        
        # Calculate position matching score during inference
        if not self.training and target_boxes is None:
            # Convert text to boxes for position matching score
            text_boxes = []
            for i, text in enumerate(texts):
                entities = self.ptpe.extract_entities(text)
                r, theta = self.ptpe.map_to_polar_coordinates(entities)
                text_box = self.ptpe.convert_polar_to_rect(r, theta, (images.shape[2], images.shape[3]))
                text_boxes.append(text_box)
            
            text_boxes = torch.stack(text_boxes).to(device)  # [B, 4]
            
            # Calculate position matching score
            pms = self.position_module.position_matching_score(outputs_coord, text_boxes)
            
            output["pms"] = pms
        
        return output
    
    def compute_detr_loss(self, outputs_class, outputs_coord, target_boxes):
        """
        Compute DETR loss (classification + box regression)
        
        Args:
            outputs_class (torch.Tensor): Class predictions [B, num_queries, 1]
            outputs_coord (torch.Tensor): Box predictions [B, num_queries, 4]
            target_boxes (torch.Tensor): Target boxes [B, M, 4]
            
        Returns:
            torch.Tensor: DETR loss
        """
        batch_size = outputs_class.shape[0]
        device = outputs_class.device
        
        # Create target class labels (all positive for now - binary detection)
        target_class = torch.ones_like(outputs_class)
        
        # Classification loss (focal loss)
        cls_loss = self.focal_loss(outputs_class, target_class)
        
        # Box loss (L1 + GIoU)
        # For each target, find the best matching prediction
        l1_loss = 0
        giou_loss = 0
        
        box_loss_weight = self.config["loss"]["box_loss_weight"]
        giou_loss_weight = self.config["loss"]["giou_loss_weight"]
        
        for i in range(batch_size):
            pred_boxes = outputs_coord[i]  # [num_queries, 4]
            tgt_boxes = target_boxes[i]    # [M, 4]
            
            # Skip if no target boxes
            if tgt_boxes.shape[0] == 0:
                continue
            
            # Calculate cost matrix
            cost_giou = torch.cdist(pred_boxes, tgt_boxes, p=1)  # [num_queries, M]
            
            # Assign predictions to targets (Hungarian matching)
            indices = self.hungarian_matching(cost_giou)
            pred_idx, tgt_idx = indices
            
            # Compute losses for matched pairs
            l1_loss += F.l1_loss(pred_boxes[pred_idx], tgt_boxes[tgt_idx], reduction='sum')
            
            # Compute GIoU
            pred_boxes_matched = pred_boxes[pred_idx]
            tgt_boxes_matched = tgt_boxes[tgt_idx]
            
            # Calculate IoU
            lt = torch.max(pred_boxes_matched[:, :2], tgt_boxes_matched[:, :2])
            rb = torch.min(pred_boxes_matched[:, 2:], tgt_boxes_matched[:, 2:])
            wh = (rb - lt).clamp(min=0)
            intersection = wh[:, 0] * wh[:, 1]
            area1 = (pred_boxes_matched[:, 2] - pred_boxes_matched[:, 0]) * (pred_boxes_matched[:, 3] - pred_boxes_matched[:, 1])
            area2 = (tgt_boxes_matched[:, 2] - tgt_boxes_matched[:, 0]) * (tgt_boxes_matched[:, 3] - tgt_boxes_matched[:, 1])
            union = area1 + area2 - intersection
            iou = intersection / (union + 1e-6)
            
            # Calculate GIoU
            lt_all = torch.min(pred_boxes_matched[:, :2], tgt_boxes_matched[:, :2])
            rb_all = torch.max(pred_boxes_matched[:, 2:], tgt_boxes_matched[:, 2:])
            wh_all = (rb_all - lt_all).clamp(min=0)
            area_all = wh_all[:, 0] * wh_all[:, 1]
            giou = iou - (area_all - union) / (area_all + 1e-6)
            
            giou_loss += (1 - giou).sum()
        
        # Normalize losses
        num_targets = sum([t.shape[0] for t in target_boxes])
        l1_loss = l1_loss / max(num_targets, 1)
        giou_loss = giou_loss / max(num_targets, 1)
        
        # Combine losses
        box_loss = box_loss_weight * l1_loss + giou_loss_weight * giou_loss
        
        # Total DETR loss
        detr_loss = cls_loss + box_loss
        
        return detr_loss
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal loss for classification
        
        Args:
            pred (torch.Tensor): Predictions [B, N, 1]
            target (torch.Tensor): Targets [B, N, 1]
            alpha (float): Balancing factor
            gamma (float): Focusing parameter
            
        Returns:
            torch.Tensor: Focal loss
        """
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal loss components
        pt = torch.exp(-bce)
        focal_weight = alpha * (1 - pt) ** gamma
        
        # Apply weight to BCE
        loss = focal_weight * bce
        
        return loss.mean()
    
    def hungarian_matching(self, cost_matrix):
        """
        Simple greedy matching implementation (substitute for Hungarian algorithm)
        
        Args:
            cost_matrix (torch.Tensor): Cost matrix [num_queries, num_targets]
            
        Returns:
            tuple: (pred_indices, tgt_indices)
        """
        pred_indices = []
        tgt_indices = []
        
        # Make a copy of cost matrix
        cost = cost_matrix.clone()
        
        # While we have targets to match
        while cost.numel() > 0:
            # Find minimum cost
            min_val, min_idx = cost.view(-1).min(0)
            
            # Convert flat index to 2D indices
            min_i = min_idx // cost.shape[1]
            min_j = min_idx % cost.shape[1]
            
            # Add to indices
            pred_indices.append(min_i.item())
            tgt_indices.append(min_j.item())
            
            # Remove matched rows and columns
            mask = torch.ones_like(cost, dtype=torch.bool)
            mask[min_i, :] = 0
            mask[:, min_j] = 0
            
            # Break if all targets are matched
            if mask.sum() == 0:
                break
                
            # Update cost matrix
            cost = cost[mask].view(cost.shape[0] - 1, cost.shape[1] - 1)
        
        return torch.tensor(pred_indices, device=cost_matrix.device), torch.tensor(tgt_indices, device=cost_matrix.device)
    
    def positional_encoding(self, height, width, device):
        """
        Generate 2D positional encodings for transformer
        
        Args:
            height (int): Feature map height
            width (int): Feature map width
            device: Device to place tensors on
            
        Returns:
            torch.Tensor: Positional encoding [1, C, H, W]
        """
        dim = self.config["model"]["decoder"]["dim"] // 2
        
        y_embed = torch.arange(height, device=device).float()
        x_embed = torch.arange(width, device=device).float()
        
        y_embed = y_embed / height
        x_embed = x_embed / width
        
        pos_y = y_embed.unsqueeze(1).unsqueeze(1).repeat(1, dim, 1)
        pos_x = x_embed.unsqueeze(1).unsqueeze(0).repeat(height, 1, dim)
        
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        
        pos_y[:, 0::2, 0] = torch.sin(pos_y[:, 0::2, 0] * div_term)
        pos_y[:, 1::2, 0] = torch.cos(pos_y[:, 1::2, 0] * div_term)
        
        pos_x[:, :, 0::2] = torch.sin(pos_x[:, :, 0::2] * div_term)
        pos_x[:, :, 1::2] = torch.cos(pos_x[:, :, 1::2] * div_term)
        
        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1).unsqueeze(0)
        
        return pos


class MLP(nn.Module):
    """
    Multi-layer perceptron for box regression
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x 