"""
Dataset classes for Clash Royale Emote Facial Expression Detection
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, Callable, Tuple


class EmotionDataset(Dataset):
    """
    Dataset class for facial expression images mapped to Clash Royale emotes.
    
    Supports multiple data formats:
    - Directory structure: data/processed/{split}/{emotion_class}/image.jpg
    - CSV format: image_path, emotion_label
    """
    
    # Mapping from emotion labels to Clash Royale emotes
    EMOTION_TO_EMOTE = {
        'angry': 'angry_emote',
        'disgust': 'disgust_emote',
        'fear': 'scared_emote',
        'happy': 'laughing_emote',
        'sad': 'crying_king',
        'surprise': 'surprised_emote',
        'neutral': 'thinking_emote'
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        csv_file: Optional[str] = None,
        img_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            data_dir: Root directory containing the data
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            csv_file: Optional CSV file with image paths and labels
            img_size: Target image size (height, width)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Load data
        if csv_file:
            self.data = self._load_from_csv(csv_file)
        else:
            self.data = self._load_from_directory()
        
        # Create label mappings
        self.classes = sorted(list(set([item['label'] for item in self.data])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.data)} images for {split} split")
        print(f"Classes: {self.classes}")
    
    def _load_from_directory(self):
        """Load images from directory structure: data_dir/split/class/image.jpg"""
        data = []
        split_dir = os.path.join(self.data_dir, 'processed', self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    data.append({
                        'path': img_path,
                        'label': class_name,
                        'emote': self.EMOTION_TO_EMOTE.get(class_name, 'default')
                    })
        
        return data
    
    def _load_from_csv(self, csv_file):
        """Load images from CSV file with columns: image_path, emotion"""
        df = pd.read_csv(csv_file)
        data = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.data_dir, row['image_path'])
            emotion = row['emotion']
            
            data.append({
                'path': img_path,
                'label': emotion,
                'emote': self.EMOTION_TO_EMOTE.get(emotion, 'default')
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer class label
            emote: String emote name (for visualization)
        """
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['path']).convert('RGB')
        
        # Resize if no transform provided
        if self.transform is None:
            image = image.resize(self.img_size)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        else:
            image = self.transform(image)
        
        # Get label index
        label = self.class_to_idx[item['label']]
        
        return image, label, item['emote']
    
    def get_class_distribution(self):
        """Returns dictionary with count of samples per class"""
        distribution = {}
        for item in self.data:
            label = item['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def get_sample_weights(self):
        """
        Calculate sample weights for balanced sampling.
        Useful for handling class imbalance.
        """
        class_counts = self.get_class_distribution()
        total_samples = len(self.data)
        
        # Calculate weight for each class (inverse frequency)
        class_weights = {
            cls: total_samples / count 
            for cls, count in class_counts.items()
        }
        
        # Assign weight to each sample
        sample_weights = [
            class_weights[item['label']] 
            for item in self.data
        ]
        
        return torch.DoubleTensor(sample_weights)


class FER2013Dataset(Dataset):
    """
    Dataset class specifically for FER2013 format.
    FER2013 CSV format: emotion, pixels, Usage
    """
    
    EMOTION_LABELS = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    def __init__(
        self,
        csv_file: str,
        split: str = 'Training',
        transform: Optional[Callable] = None,
        img_size: int = 48
    ):
        """
        Args:
            csv_file: Path to FER2013 CSV file
            split: One of 'Training', 'PublicTest', or 'PrivateTest'
            transform: Optional transform
            img_size: Image dimension (FER2013 uses 48x48)
        """
        self.transform = transform
        self.img_size = img_size
        
        # Load and filter data
        df = pd.read_csv(csv_file)
        self.data = df[df['Usage'] == split].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} images for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Parse pixel string to image
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
        image = pixels.reshape(self.img_size, self.img_size)
        
        # Convert to RGB (duplicate grayscale channel)
        image = np.stack([image] * 3, axis=-1)
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        label = int(row['emotion'])
        emotion_name = self.EMOTION_LABELS[label]
        emote = EmotionDataset.EMOTION_TO_EMOTE.get(emotion_name, 'default')
        
        return image, label, emote


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
):
    """
    Convenience function to create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        train_transform: Transform for training data
        val_transform: Transform for val/test data
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # Create datasets
    train_dataset = EmotionDataset(data_dir, 'train', transform=train_transform)
    val_dataset = EmotionDataset(data_dir, 'val', transform=val_transform)
    test_dataset = EmotionDataset(data_dir, 'test', transform=val_transform)
    
    # Optional: Use weighted sampling for balanced batches
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Comment this out if you don't want weighted sampling
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Example usage
    from torchvision import transforms
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = EmotionDataset(
        data_dir='data',
        split='train',
        transform=train_transform
    )
    
    # Print info
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test loading a sample
    image, label, emote = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}, Emote: {emote}")