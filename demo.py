import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import math

from models.encoders.ptpe import PolarTextPositionEncoder
from utils.visualization import visualize_ptpe
from utils.fdi_parser import FDIParser

def parse_args():
    parser = argparse.ArgumentParser(description='PolarDETR PTPE Demo')
    parser.add_argument('--text', type=str, default="3mm cyst distal to tooth 37",
                        help='Text description in English (e.g., "3mm cyst distal to tooth 37")')
    parser.add_argument('--output_dir', type=str, default='outputs/demo',
                        help='Output directory')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--fdi_xml', type=str, default='data/FDI_MATCH.xml',
                        help='Path to FDI_MATCH.xml file')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize the polar encoding')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Check if FDI_MATCH.xml exists
    if not os.path.exists(args.fdi_xml):
        print(f"Warning: FDI_MATCH.xml not found at {args.fdi_xml}, using default mappings")
        fdi_xml_path = None
    else:
        fdi_xml_path = args.fdi_xml
    
    # Create FDI parser
    fdi_parser = FDIParser(fdi_xml_path)
    
    # Mock the PTPE module without loading the full transformer
    class MockPTPE:
        def __init__(self, config, fdi_parser):
            self.config = config
            self.fdi_parser = fdi_parser
            self.pixel_size = config["data"]["pixel_size"]
    
    # Create PTPE module
    ptpe = MockPTPE(config, fdi_parser)
    
    # Process text
    text = args.text
    print(f"Processing text: {text}")
    
    # Extract entities
    entities = fdi_parser.extract_entities(text)
    print("Extracted entities:")
    for key, value in entities.items():
        print(f"  {key}: {value}")
    
    # Map to polar coordinates
    if entities["tooth_number"] is not None:
        angle = fdi_parser.get_angle(entities["tooth_number"])
        direction_angle = 0
        if entities["direction"] is not None:
            direction_angle = fdi_parser.get_direction_angle(entities["direction"])
        
        theta = math.radians(angle + direction_angle)
        r = entities["distance"] / ptpe.pixel_size
    else:
        # Fallback if no tooth number
        quadrant = entities["quadrant"]
        angle = fdi_parser.get_quadrant_angle(quadrant) if quadrant else 0
        direction_angle = 0
        if entities["direction"] is not None:
            direction_angle = fdi_parser.get_direction_angle(entities["direction"])
            
        theta = math.radians(angle + direction_angle)
        r = entities["distance"] / ptpe.pixel_size
    
    print(f"Polar coordinates: r={r:.2f}, theta={theta:.2f} rad ({np.degrees(theta):.2f}°)")
    
    # Generate position encoding
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    log_r = math.log(r + 1)
    encoding = [sin_theta, cos_theta, log_r]
    
    print(f"Position encoding: {encoding}")
    
    # Visualize
    if args.visualize:
        fig = visualize_ptpe(text, entities, r, theta)
        
        # Save visualization
        output_path = os.path.join(args.output_dir, "ptpe_visualization.png")
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {output_path}")
        
        # Show if running interactively
        plt.show()

if __name__ == '__main__':
    main() 