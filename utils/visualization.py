import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from matplotlib.figure import Figure

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor to display
    
    Args:
        tensor (torch.Tensor): Image tensor [C, H, W]
        mean (list): Mean values for each channel
        std (list): Std values for each channel
        
    Returns:
        numpy.ndarray: Denormalized image [H, W, C] in range [0, 1]
    """
    # Clone tensor
    img = tensor.clone()
    
    # Denormalize
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    # Convert to numpy array and transpose
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    
    # Clip values to [0, 1]
    img = np.clip(img, 0, 1)
    
    return img

def visualize_predictions(images, preds, texts, targets=None, config=None, max_images=4):
    """
    Visualize model predictions
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        preds (torch.Tensor): Predicted boxes [B, N, 4]
        texts (list): List of text descriptions
        targets (list, optional): List of target boxes
        config (dict, optional): Configuration dictionary
        max_images (int): Maximum number of images to visualize
        
    Returns:
        matplotlib.figure.Figure: Figure with visualizations
    """
    # Limit number of images
    batch_size = min(images.shape[0], max_images)
    
    # Create figure
    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 5 * batch_size))
    
    # Handle single image case
    if batch_size == 1:
        axes = [axes]
    
    # Detection confidence threshold
    threshold = config['evaluation']['score_threshold'] if config else 0.5
    
    # Plot each image
    for i in range(batch_size):
        # Get image
        img = denormalize_image(images[i])
        
        # Get text description
        text = texts[i] if i < len(texts) else ""
        
        # Get predictions
        pred = preds[i]
        
        # Get scores and boxes
        if pred.dim() > 2 and pred.shape[1] > 5:  # [N, num_features] including ACS/PMS
            scores = pred[:, 0].cpu().numpy()
            boxes = pred[:, 1:5].cpu().numpy()
            acs_scores = pred[:, -2].cpu().numpy() if pred.shape[1] > 5 else None
            pms_scores = pred[:, -1].cpu().numpy() if pred.shape[1] > 6 else None
        else:  # Normal [N, 5] format
            scores = pred[:, 0].cpu().numpy()
            boxes = pred[:, 1:5].cpu().numpy()
            acs_scores = None
            pms_scores = None
        
        # Filter by threshold
        mask = scores > threshold
        scores = scores[mask]
        boxes = boxes[mask]
        
        if acs_scores is not None:
            acs_scores = acs_scores[mask]
        if pms_scores is not None:
            pms_scores = pms_scores[mask]
        
        # Display image
        axes[i].imshow(img)
        
        # Set title
        axes[i].set_title(f"Description: {text}")
        
        # Plot predicted boxes
        for j, (box, score) in enumerate(zip(boxes, scores)):
            # Get box coordinates
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            
            # Add rectangle to axes
            axes[i].add_patch(rect)
            
            # Add scores text
            score_text = f"Score: {score:.2f}"
            if acs_scores is not None:
                score_text += f", ACS: {acs_scores[j]:.2f}"
            if pms_scores is not None:
                score_text += f", PMS: {pms_scores[j]:.2f}"
                
            axes[i].text(
                x_min, y_min - 5, score_text,
                color='white', fontsize=10, 
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        # Plot target boxes if available
        if targets and i < len(targets):
            target = targets[i].cpu().numpy()
            
            for box in target:
                # Get box coordinates
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                # Create rectangle
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, 
                    linewidth=2, edgecolor='g', facecolor='none', linestyle='--'
                )
                
                # Add rectangle to axes
                axes[i].add_patch(rect)
        
        # Remove axis ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_ptpe(text, entities, r, theta, image_size=(512, 512)):
    """
    Visualize Polar Text-Position Encoding
    
    Args:
        text (str): Text description
        entities (dict): Extracted entities
        r (float): Radius
        theta (float): Angle in radians
        image_size (tuple): Image size (height, width)
        
    Returns:
        matplotlib.figure.Figure: Figure with visualization
    """
    # Create figure
    fig, ax = plt.subplots(1, figsize=(8, 8))
    
    # Set limits
    height, width = image_size
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # Draw background grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set center point
    center_x, center_y = width // 2, height // 2
    
    # Draw center point
    ax.plot(center_x, center_y, 'ko', markersize=5)
    
    # Convert polar to cartesian
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    
    # Draw radius line
    ax.plot([center_x, x], [center_y, y], 'r-', alpha=0.7)
    
    # Draw position point
    ax.plot(x, y, 'ro', markersize=8)
    
    # Draw angle arc
    angle_rad = 50
    theta_deg = np.degrees(theta)
    arc = patches.Arc((center_x, center_y), angle_rad*2, angle_rad*2, 
                     theta1=0, theta2=theta_deg, color='blue', alpha=0.5)
    ax.add_patch(arc)
    
    # Add text for angle
    arc_x = center_x + angle_rad * 0.8 * np.cos(theta/2)
    arc_y = center_y + angle_rad * 0.8 * np.sin(theta/2)
    ax.text(arc_x, arc_y, f"{theta_deg:.1f}°", color='blue')
    
    # Add text for distance
    ax.text(x + 10, y, f"r={r:.1f}px", color='red')
    
    # Set title
    ax.set_title(f"Polar Encoding for: {text}")
    
    # Add entity information
    info_text = "\n".join([f"{k}: {v}" for k, v in entities.items() if v is not None])
    ax.text(10, 10, info_text, fontsize=10, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_anatomy_attention(image, attention_scores, region_names):
    """
    Visualize anatomical attention scores
    
    Args:
        image (torch.Tensor or numpy.ndarray): Image
        attention_scores (torch.Tensor): Attention scores [K]
        region_names (list): List of region names
        
    Returns:
        matplotlib.figure.Figure: Figure with visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = denormalize_image(image)
    
    # Convert attention scores to numpy if needed
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Plot attention scores
    y_pos = np.arange(len(region_names))
    ax2.barh(y_pos, attention_scores)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(region_names)
    ax2.set_xlabel('Attention Score')
    ax2.set_title('Anatomical Region Attention')
    
    # Sort regions by attention score for better visualization
    sort_idx = np.argsort(attention_scores)
    sorted_scores = attention_scores[sort_idx]
    sorted_names = [region_names[i] for i in sort_idx]
    
    # Highlight top regions
    for i, score in enumerate(sorted_scores[-3:]):  # Top 3 regions
        j = len(sorted_scores) - i - 1  # Reverse index
        ax2.get_children()[j].set_color('r')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig 