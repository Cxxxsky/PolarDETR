import os
import argparse
import yaml
import torch
import torchvision.transforms as T
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import cv2

from models import PolarDETR
from utils.visualization import visualize_predictions, denormalize_image

def parse_args():
    parser = argparse.ArgumentParser(description='PolarDETR Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image (DICOM or regular image format like .png, .jpg, .jpeg, etc., I recommend .jpg)')
    parser.add_argument('--text', type=str, required=True,
                        help='Text description (e.g., "3mm cyst distal to tooth 37")')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                        help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection confidence threshold (overrides config)')
    parser.add_argument('--save_json', action='store_true',
                        help='Save detection results as JSON')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help='Device to run inference on')
    return parser.parse_args()

def load_model(model_path, device):
    """
    Load model from checkpoint
    
    Args:
        model_path (str): Path to model checkpoint
        device (torch.device): Device to load model on
        
    Returns:
        tuple: (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError(f"Config not found in checkpoint {model_path}")
    
    # Create model
    model = PolarDETR(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model, config

def load_image(image_path, transform=None):
    """
    Load and preprocess image
    
    Args:
        image_path (str): Path to image file
        transform (callable, optional): Transform to apply
        
    Returns:
        tuple: (preprocessed_tensor, original_image, image_size)
    """
    if image_path.lower().endswith('.dcm'):
        # Load DICOM file
        try:
            dicom = pydicom.dcmread(image_path)
            image = dicom.pixel_array
            
            # Convert to 3 channel image if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
            
            # Normalize to [0, 255]
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            
            pil_image = Image.fromarray(image)
        except Exception as e:
            print(f"Error loading DICOM file {image_path}: {e}")
            # Return a placeholder image
            pil_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
    else:
        # Load regular image file
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image file {image_path}: {e}")
            # Return a placeholder image
            pil_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
    
    # Save original size
    original_size = pil_image.size  # (width, height)
    
    # Clone image for visualization
    original_image = pil_image.copy()
    
    # Apply transform if provided
    if transform is not None:
        tensor_image = transform(pil_image)
    else:
        # Default transform
        tensor_image = T.functional.to_tensor(pil_image)
        tensor_image = T.functional.normalize(
            tensor_image, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    return tensor_image, original_image, original_size

def post_process_predictions(outputs, threshold=None, max_predictions=5):
    """
    Post-process model predictions
    
    Args:
        outputs (dict): Model outputs
        threshold (float, optional): Detection confidence threshold
        max_predictions (int): Maximum number of predictions to return
        
    Returns:
        tuple: (scores, boxes, acs, pms)
    """
    pred_logits = outputs['pred_logits']  # [B, num_queries, 1]
    pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
    
    # Extract scores
    scores = pred_logits.sigmoid().squeeze(-1)  # [B, num_queries]
    
    # Get anatomical consistency scores (should always be available)
    acs = outputs.get('acs', None)  # [B, num_queries]
    
    # Get position matching scores if available
    pms = outputs.get('pms', None)  # [B, num_queries]
    
    # Apply threshold
    if threshold is not None:
        mask = scores > threshold
        scores = scores[mask]
        boxes = pred_boxes[mask.unsqueeze(-1).expand_as(pred_boxes)].view(-1, 4)
        
        if acs is not None:
            acs = acs[mask]
        if pms is not None:
            pms = pms[mask]
    else:
        # Take top predictions
        B, num_queries = scores.shape
        topk = min(max_predictions, num_queries)
        
        # Get indices of top scores
        topk_indices = torch.topk(scores, topk, dim=1).indices  # [B, topk]
        
        # Gather boxes and scores
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, topk)
        boxes = pred_boxes[batch_indices, topk_indices]  # [B, topk, 4]
        scores = scores[batch_indices, topk_indices]     # [B, topk]
        
        if acs is not None:
            acs = acs[batch_indices, topk_indices]       # [B, topk]
        if pms is not None:
            pms = pms[batch_indices, topk_indices]       # [B, topk]
    
    return scores, boxes, acs, pms

def visualize_and_save_results(
    image, text, scores, boxes, acs=None, pms=None, 
    output_path=None, config=None
):
    """
    Visualize and save detection results
    
    Args:
        image (PIL.Image): Original image
        text (str): Text description
        scores (torch.Tensor): Detection scores [B, N]
        boxes (torch.Tensor): Predicted boxes [B, N, 4]
        acs (torch.Tensor, optional): Anatomical consistency scores [B, N]
        pms (torch.Tensor, optional): Position matching scores [B, N]
        output_path (str, optional): Path to save visualization
        config (dict, optional): Model configuration
        
    Returns:
        matplotlib.figure.Figure: Visualization figure
    """
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display image
    ax.imshow(image)
    
    # Remove axis
    ax.axis('off')
    
    # Set title
    ax.set_title(f'Description: {text}')
    
    # Get number of predictions
    num_preds = boxes.shape[1]
    
    # Loop through predictions
    for i in range(num_preds):
        # Get box coordinates
        x_min, y_min, x_max, y_max = boxes[0, i].tolist()
        width = x_max - x_min
        height = y_max - y_min
        
        # Get score
        score = scores[0, i].item()
        
        # Create additional info text
        info_text = f'Score: {score:.2f}'
        
        # Add ACS if available
        if acs is not None:
            acs_score = acs[0, i].item()
            info_text += f', ACS: {acs_score:.2f}'
        
        # Add PMS if available
        if pms is not None:
            pms_score = pms[0, i].item()
            info_text += f', PMS: {pms_score:.2f}'
        
        # Create rectangle
        rect = patches.Rectangle(
            (x_min, y_min), width, height, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        
        # Add rectangle to axes
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            x_min, y_min - 5, info_text,
            color='white', fontsize=10, 
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {output_path}")
    
    return fig

def save_json_results(
    text, scores, boxes, acs=None, pms=None, 
    output_path=None, image_size=None
):
    """
    Save detection results as JSON
    
    Args:
        text (str): Text description
        scores (torch.Tensor): Detection scores [B, N]
        boxes (torch.Tensor): Predicted boxes [B, N, 4]
        acs (torch.Tensor, optional): Anatomical consistency scores [B, N]
        pms (torch.Tensor, optional): Position matching scores [B, N]
        output_path (str): Path to save JSON file
        image_size (tuple, optional): Original image size (width, height)
    """
    # Create results dictionary
    results = {
        'text_description': text,
        'detections': []
    }
    
    if image_size:
        results['image_size'] = {
            'width': image_size[0],
            'height': image_size[1]
        }
    
    # Get number of predictions
    num_preds = boxes.shape[1]
    
    # Loop through predictions
    for i in range(num_preds):
        # Get box coordinates
        x_min, y_min, x_max, y_max = boxes[0, i].tolist()
        
        # Get score
        score = scores[0, i].item()
        
        # Create detection dict
        detection = {
            'box': [x_min, y_min, x_max, y_max],
            'score': score
        }
        
        # Add ACS if available
        if acs is not None:
            detection['acs'] = acs[0, i].item()
        
        # Add PMS if available
        if pms is not None:
            detection['pms'] = pms[0, i].item()
        
        # Add to results
        results['detections'].append(detection)
    
    # Save as JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    model, config = load_model(args.model_path, device)
    print("Model loaded successfully")
    
    # Create image transform
    transform = T.Compose([
        T.Resize(tuple(config['data']['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    tensor_image, original_image, original_size = load_image(args.image, transform)
    print(f"Image loaded with size: {original_size}")
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(tensor_image, [args.text])
    
    # Post-process predictions
    threshold = args.threshold if args.threshold is not None else config['evaluation']['score_threshold']
    scores, boxes, acs, pms = post_process_predictions(outputs, threshold)
    
    print(f"Found {boxes.shape[1]} detections with threshold {threshold}")
    
    # Visualize and save results
    if args.output_dir:
        # Create output file paths
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        vis_path = os.path.join(args.output_dir, f"{base_name}_result.png")
        
        # Visualize
        fig = visualize_and_save_results(
            original_image, args.text, scores, boxes, acs, pms, 
            output_path=vis_path, config=config
        )
        
        # Save results as JSON if requested
        if args.save_json:
            json_path = os.path.join(args.output_dir, f"{base_name}_result.json")
            save_json_results(
                args.text, scores, boxes, acs, pms, 
                output_path=json_path, image_size=original_size
            )
    else:
        # Just visualize without saving
        fig = visualize_and_save_results(
            original_image, args.text, scores, boxes, acs, pms, 
            config=config
        )
        plt.show()


if __name__ == '__main__':
    main() 