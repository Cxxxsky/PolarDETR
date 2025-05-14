import os
import torch
import torch.utils.data as data
import numpy as np
import pydicom
from PIL import Image
import json
import cv2

class DentalDataset(data.Dataset):
    """
    Dataset for loading dental images with textual descriptions
    """
    
    def __init__(self, data_dir, split='train', transform=None, max_boxes=10):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Data directory path
            split (str): Data split ('train', 'val', or 'test')
            transform (callable, optional): Transform to apply to images
            max_boxes (int): Maximum number of target boxes per image
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_boxes = max_boxes
        
        # Load dataset annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """
        Load dataset annotations from JSON file
        
        Returns:
            list: List of annotation dictionaries
        """
        # For demonstration, we define a small dummy dataset
        # In a real implementation, annotations would be loaded from a file
        
        # Path to annotations file
        annotations_file = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(annotations_file):
            # Load from file if it exists
            with open(annotations_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return empty list if file doesn't exist
            print(f"Warning: Annotations file {annotations_file} not found.")
            return []
    
    def __len__(self):
        """
        Get dataset size
        
        Returns:
            int: Number of samples in dataset
        """
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx (int): Item index
            
        Returns:
            tuple: (image, text, target)
                image: torch.Tensor of shape [3, H, W]
                text: Text description
                target: Dict containing target boxes and labels
        """
        # Get annotation
        annotation = self.annotations[idx]
        
        # Get image path
        image_path = os.path.join(self.data_dir, annotation['image_path'])
        
        # Load image
        image = self._load_image(image_path)
        
        # Get text description
        text = annotation['text_description']
        
        # Get target boxes if available
        if 'boxes' in annotation and self.split != 'test':
            boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
            
            # Ensure we don't exceed max_boxes
            if boxes.shape[0] > self.max_boxes:
                boxes = boxes[:self.max_boxes]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        
        # Apply transforms to image
        if self.transform is not None:
            image = self.transform(image)
        
        return image, text, boxes
    
    def _load_image(self, image_path):
        """
        Load dental image (supports DICOM and regular image formats)
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            PIL.Image or numpy.ndarray: Loaded image
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
                
                return Image.fromarray(image)
            except Exception as e:
                print(f"Error loading DICOM file {image_path}: {e}")
                # Return a placeholder image
                return Image.new('RGB', (512, 512), color=(0, 0, 0))
        else:
            # Load regular image file
            try:
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image file {image_path}: {e}")
                # Return a placeholder image
                return Image.new('RGB', (512, 512), color=(0, 0, 0))

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable number of boxes
        
        Args:
            batch (list): List of (image, text, boxes) tuples
            
        Returns:
            tuple: (images, texts, targets)
                images: torch.Tensor of shape [B, 3, H, W]
                texts: List of text descriptions
                targets: List of target boxes [B, variable, 4]
        """
        images, texts, all_boxes = zip(*batch)
        
        # Stack images
        images = torch.stack(images)
        
        # Return texts as list
        texts = list(texts)
        
        # Return boxes as list (each element has different shape)
        boxes = list(all_boxes)
        
        return images, texts, boxes


class DentalDataModule:
    """
    Data module for managing dental datasets and dataloaders
    """
    
    def __init__(self, data_dir, batch_size=16, num_workers=4, transforms=None):
        """
        Initialize data module
        
        Args:
            data_dir (str): Data directory path
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            transforms (dict): Dictionary of transforms for each split
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms or {}
        
    def train_dataloader(self):
        """
        Get training dataloader
        
        Returns:
            torch.utils.data.DataLoader: Training dataloader
        """
        dataset = DentalDataset(
            self.data_dir,
            split='train',
            transform=self.transforms.get('train')
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=DentalDataset.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """
        Get validation dataloader
        
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        dataset = DentalDataset(
            self.data_dir,
            split='val',
            transform=self.transforms.get('val')
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DentalDataset.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Get test dataloader
        
        Returns:
            torch.utils.data.DataLoader: Test dataloader
        """
        dataset = DentalDataset(
            self.data_dir,
            split='test',
            transform=self.transforms.get('test')
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=DentalDataset.collate_fn,
            pin_memory=True
        ) 