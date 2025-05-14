import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AnatomicalConstraintModule(nn.Module):
    """
    Anatomical Constraint Learning Module
    
    This module enforces anatomical constraints on the detection process
    by incorporating prior knowledge about anatomical regions.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Number of anatomical regions
        self.num_regions = config["model"]["anatomy"]["num_regions"]
        
        # Region names
        self.region_names = config["model"]["anatomy"]["regions"]
        
        # Feature dimension
        self.feature_dim = config["model"]["decoder"]["dim"]
        
        # Prior distribution
        prior_dist = config["model"]["anatomy"]["prior_distributions"]
        self.register_buffer('prior_distribution', 
                            torch.tensor([prior_dist.get(region, 0.1) 
                                         for region in self.region_names]))
        
        # Region embeddings (learnable)
        self.region_embeddings = nn.Parameter(
            torch.randn(self.num_regions, self.feature_dim)
        )
        
        # Region attention projection layer
        self.attention_projection = nn.Linear(self.feature_dim, self.feature_dim)
        
    def forward(self, region_features, query_vectors):
        """
        Compute anatomical attention scores and loss
        
        Args:
            region_features (torch.Tensor): Features of anatomical regions [B, K, D]
                                           K: number of regions, D: feature dimension
            query_vectors (torch.Tensor): Query vectors from DETR decoder [B, N, D]
                                         N: number of queries, D: feature dimension
                                  
        Returns:
            tuple: (enhanced_queries, anatomy_loss)
                  enhanced_queries: Query vectors enhanced with anatomical constraints
                  anatomy_loss: KL divergence loss between attention and prior
        """
        batch_size = query_vectors.shape[0]
        num_queries = query_vectors.shape[1]
        
        # If we don't have region features, use learnable embeddings
        if region_features is None:
            region_features = self.region_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project queries for attention computation
        projected_queries = self.attention_projection(query_vectors)
        
        # Calculate attention scores (B, N, K)
        attention_logits = torch.bmm(
            projected_queries,                            # [B, N, D]
            region_features.transpose(1, 2)               # [B, D, K]
        )                                                 # [B, N, K]
        
        # Apply softmax to get attention distribution over regions
        attention_scores = F.softmax(attention_logits, dim=-1)    # [B, N, K]
        
        # Average attention scores across all queries
        mean_attention = attention_scores.mean(dim=1)     # [B, K]
        
        # Calculate KL divergence loss against prior distribution
        prior = self.prior_distribution.unsqueeze(0).expand(batch_size, -1)  # [B, K]
        
        # Ensure prior is a proper distribution (sums to 1)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        
        # Compute KL divergence loss
        kl_loss = F.kl_div(
            mean_attention.log(),       # log probabilities (log q)
            prior,                      # target (p)
            reduction='batchmean'       # sum over batches, mean over elements
        )
        
        # Enhance query vectors with weighted anatomical region information
        region_context = torch.bmm(
            attention_scores,           # [B, N, K]
            region_features             # [B, K, D]
        )                               # [B, N, D]
        
        enhanced_queries = query_vectors + region_context
        
        return enhanced_queries, kl_loss
    
    def get_region_mask(self, image_size, device):
        """
        Generate anatomical region masks
        
        Args:
            image_size (tuple): (height, width) of the image
            device: Device to create tensors on
            
        Returns:
            torch.Tensor: Binary masks for each anatomical region [K, H, W]
        """
        height, width = image_size
        masks = torch.zeros((self.num_regions, height, width), device=device)
        
        # For demonstration, create simple region masks
        # In a real implementation, these would be derived from anatomical knowledge
        
        # Example: Mandibular canal (horizontal band in lower third)
        # This is a simplified example; real masks would be more anatomically accurate
        if "mandibular_canal" in self.region_names:
            idx = self.region_names.index("mandibular_canal")
            y_start = int(height * 0.7)
            y_end = int(height * 0.8)
            masks[idx, y_start:y_end, :] = 1.0
        
        # Example: Alveolar ridge (top area)
        if "alveolar_ridge" in self.region_names:
            idx = self.region_names.index("alveolar_ridge")
            y_end = int(height * 0.3)
            masks[idx, :y_end, :] = 1.0
        
        # Example: Mental foramen (small region lower left and right)
        if "mental_foramen" in self.region_names:
            idx = self.region_names.index("mental_foramen")
            # Left
            y_center = int(height * 0.7)
            x_center = int(width * 0.3)
            radius = int(width * 0.05)
            
            # Create circular mask
            y, x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device)
            )
            dist_left = ((y - y_center)**2 + (x - x_center)**2) < radius**2
            
            # Right
            x_center = int(width * 0.7)
            dist_right = ((y - y_center)**2 + (x - x_center)**2) < radius**2
            
            masks[idx] = (dist_left | dist_right).float()
        
        # Example: Apical lesion region (around tooth roots)
        if "apical_lesion_region" in self.region_names:
            idx = self.region_names.index("apical_lesion_region")
            y_center = int(height * 0.4)
            
            # Several positions for teeth
            for x_pos in [int(width * p) for p in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]]:
                radius = int(width * 0.03)
                y, x = torch.meshgrid(
                    torch.arange(height, device=device),
                    torch.arange(width, device=device)
                )
                dist = ((y - y_center)**2 + (x - x_pos)**2) < radius**2
                masks[idx] = masks[idx] | dist.float()
        
        # Other regions would be similarly defined
        # In practice, these masks could be loaded from pre-defined anatomical atlases
        
        return masks
    
    def extract_region_features(self, feature_maps, image_size):
        """
        Extract features for each anatomical region from feature maps
        
        Args:
            feature_maps (torch.Tensor): Feature maps from the backbone [B, C, H, W]
            image_size (tuple): Original image size (height, width)
            
        Returns:
            torch.Tensor: Region features [B, K, D]
        """
        batch_size = feature_maps.shape[0]
        feature_height, feature_width = feature_maps.shape[2:]
        device = feature_maps.device
        
        # Get region masks
        region_masks = self.get_region_mask((feature_height, feature_width), device)  # [K, H, W]
        
        # Expand masks for batch dimension
        region_masks = region_masks.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, K, H, W]
        
        # Expand feature maps for region dimension
        expanded_features = feature_maps.unsqueeze(1).expand(-1, self.num_regions, -1, -1, -1)  # [B, K, C, H, W]
        
        # Apply masks to features
        masked_features = expanded_features * region_masks.unsqueeze(2)  # [B, K, C, H, W]
        
        # Average pooling over spatial dimensions
        region_features = masked_features.sum(dim=(3, 4)) / (region_masks.sum(dim=(2, 3)).unsqueeze(2) + 1e-10)  # [B, K, C]
        
        return region_features
    
    def anatomical_consistency_score(self, boxes, image_size):
        """
        Calculate Anatomical Consistency Score (ACS) for predicted boxes
        
        Args:
            boxes (torch.Tensor): Bounding boxes [B, N, 4] in format [x_min, y_min, x_max, y_max]
            image_size (tuple): Original image size (height, width)
            
        Returns:
            torch.Tensor: ACS scores [B, N]
        """
        batch_size, num_boxes = boxes.shape[:2]
        device = boxes.device
        
        # Get region masks at original image resolution
        region_masks = self.get_region_mask(image_size, device)  # [K, H, W]
        
        # Convert region masks to boxes
        region_boxes = []
        for k in range(self.num_regions):
            mask = region_masks[k]
            if mask.sum() > 0:
                # Find bounding box of the mask
                y_indices, x_indices = torch.where(mask > 0)
                x_min, y_min = x_indices.min(), y_indices.min()
                x_max, y_max = x_indices.max(), y_indices.max()
                region_boxes.append([x_min, y_min, x_max, y_max])
            else:
                # Fallback for empty masks
                region_boxes.append([0, 0, 1, 1])
                
        region_boxes = torch.tensor(region_boxes, device=device)  # [K, 4]
        
        # Calculate IoU between predicted boxes and all region boxes
        ious = torch.zeros((batch_size, num_boxes, self.num_regions), device=device)
        
        for b in range(batch_size):
            for n in range(num_boxes):
                box = boxes[b, n]
                
                # Calculate intersection areas
                left = torch.max(box[0], region_boxes[:, 0])
                right = torch.min(box[2], region_boxes[:, 2])
                top = torch.max(box[1], region_boxes[:, 1])
                bottom = torch.min(box[3], region_boxes[:, 3])
                
                # Calculate width and height of intersection
                width = torch.clamp(right - left, min=0)
                height = torch.clamp(bottom - top, min=0)
                
                # Calculate intersection area
                intersection = width * height
                
                # Calculate union area
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                region_areas = (region_boxes[:, 2] - region_boxes[:, 0]) * (region_boxes[:, 3] - region_boxes[:, 1])
                union = box_area + region_areas - intersection
                
                # Calculate IoU
                ious[b, n] = intersection / (union + 1e-10)
        
        # Take maximum IoU with any region as the score
        acs_scores = ious.max(dim=2)[0]
        
        return acs_scores 