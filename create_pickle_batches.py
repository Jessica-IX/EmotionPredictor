#!/usr/bin/env python3
"""
Create pickle batch files from preprocessed_data directory structure.

This script:
1. Scans all batch directories in preprocessed_data
2. Loads images (canvas.png) and emotion probabilities (emotion_probabilities.json)
3. Extracts CLIP embeddings for each image
4. Creates pickle batch files organized by train/val/test splits
"""

import os
import os.path as osp
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip

from emotionpredictor.data_tools import create_dataset
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# ArtEmis emotion order (must match exactly)
ARTEMIS_EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]


class ImageEmotionDataset(Dataset):
    """Dataset for images and emotion distributions."""
    
    def __init__(self, image_paths: List[str], emotion_distributions: List[np.ndarray], transform=None):
        self.image_paths = image_paths
        self.emotion_distributions = emotion_distributions
        self.transform = transform
        
        assert len(image_paths) == len(emotion_distributions), \
            "Number of images must match number of emotion distributions"
        
        # Validate and normalize distributions
        for i, dist in enumerate(emotion_distributions):
            assert len(dist) == 9, \
                f"Emotion distribution {i} must have 9 values, got {len(dist)}"
            # Ensure normalized
            dist_sum = sum(dist)
            if dist_sum > 0:
                if abs(dist_sum - 1.0) > 0.01:
                    dist = np.array(dist) / dist_sum
                    self.emotion_distributions[i] = dist.astype('float32')
            else:
                # If all zeros, set uniform distribution
                dist = np.ones(9, dtype=np.float32) / 9.0
                self.emotion_distributions[i] = dist
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get emotion distribution
        emotion_dist = torch.tensor(self.emotion_distributions[idx], dtype=torch.float32)
        
        return {
            'image': image,
            'label': emotion_dist
        }


def load_emotion_data(batch_dir: Path) -> Tuple[str, np.ndarray]:
    """
    Load emotion probabilities from JSON file.
    
    Returns:
        (image_path, emotion_distribution)
    """
    json_path = batch_dir / "emotion_probabilities.json"
    image_path = batch_dir / "canvas.png"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing {image_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract probabilities in ArtEmis order
    labels = data.get('labels', [])
    probabilities = data.get('probabilities', [])
    
    # Create distribution in ArtEmis order
    artemis_probs = np.zeros(9, dtype=np.float32)
    label_to_idx = {label: i for i, label in enumerate(ARTEMIS_EMOTIONS)}
    
    for label, prob in zip(labels, probabilities):
        if label in label_to_idx:
            idx = label_to_idx[label]
            artemis_probs[idx] = prob
    
    # Normalize to ensure sum = 1.0
    if artemis_probs.sum() > 0:
        artemis_probs = artemis_probs / artemis_probs.sum()
    
    return str(image_path), artemis_probs


def find_all_batches(data_root: Path) -> List[Path]:
    """Find all batch directories."""
    batches = []
    for session_dir in sorted(data_root.glob("*-session-*")):
        if not session_dir.is_dir():
            continue
        for batch_dir in sorted(session_dir.glob("batch_*")):
            if batch_dir.is_dir():
                batches.append(batch_dir)
    return batches


def create_pickle_batches(
    data_root: str,
    output_dir: str,
    clip_model_name: str = 'RN50x16',
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    recreate: bool = False
):
    """
    Create pickle batch files from preprocessed_data.
    
    Args:
        data_root: Root directory containing session folders with batch_* subdirectories
        output_dir: Directory to save pickle batch files (will create train/, val/, test/)
        clip_model_name: CLIP model to use
        batch_size: Batch size for processing
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of DataLoader workers
        recreate: If True, overwrite existing batch files
    """
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 0.01, \
        "Splits must sum to 1.0"
    
    print(f"Scanning batches in {data_root}...")
    all_batches = find_all_batches(data_root)
    print(f"Found {len(all_batches)} batch directories")
    
    if len(all_batches) == 0:
        raise ValueError(f"No batch directories found in {data_root}")
    
    # Load all image/emotion pairs
    print("Loading image and emotion data...")
    image_paths = []
    emotion_dists = []
    failed = 0
    
    for batch_dir in tqdm(all_batches, desc="Loading batches"):
        try:
            img_path, emotion_dist = load_emotion_data(batch_dir)
            image_paths.append(img_path)
            emotion_dists.append(emotion_dist)
        except Exception as e:
            print(f"Warning: Failed to load {batch_dir}: {e}")
            failed += 1
    
    print(f"Successfully loaded {len(image_paths)} batches ({failed} failed)")
    
    if len(image_paths) == 0:
        raise ValueError("No valid batches found")
    
    # Split data
    n_total = len(image_paths)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    splits = {
        'train': (train_indices, n_train),
        'val': (val_indices, n_val),
        'test': (test_indices, n_test)
    }
    
    print(f"\nData splits:")
    print(f"  Train: {n_train} batches")
    print(f"  Val:   {n_val} batches")
    print(f"  Test:  {n_test} batches")
    
    # Load CLIP model
    print(f"\nLoading CLIP model: {clip_model_name}...")
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    clip_visual = clip_model.visual.to(device)
    clip_visual.eval()
    
    # Process each split
    for split_name, (indices, n_samples) in splits.items():
        if n_samples == 0:
            print(f"\nSkipping {split_name} split (no samples)")
            continue
        
        print(f"\nProcessing {split_name} split ({n_samples} samples)...")
        
        # Get data for this split
        split_image_paths = [image_paths[i] for i in indices]
        split_emotion_dists = [emotion_dists[i] for i in indices]
        
        # Create dataset
        dataset = ImageEmotionDataset(
            image_paths=split_image_paths,
            emotion_distributions=split_emotion_dists,
            transform=preprocess
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Create output directory
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        if recreate:
            # Clear existing files
            for f in split_output_dir.glob("batch*.bin"):
                f.unlink()
        
        # Extract embeddings and save as pickle batches
        print(f"Extracting CLIP embeddings and saving to {split_output_dir}...")
        create_dataset(
            model=clip_visual,
            data_loader=dataloader,
            save_path=str(split_output_dir),
            recreate=recreate
        )
        
        n_batches = len(list(split_output_dir.glob("batch*.bin")))
        print(f"✓ Created {n_batches} batch files for {split_name} split")
    
    print(f"\n{'='*60}")
    print(f"✓ All pickle batch files created in {output_dir}")
    print(f"  Total batches: {len(image_paths)}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create pickle batch files for fine-tuning')
    parser.add_argument(
        '--data_root',
        type=str,
        default='preprocessed_data',
        help='Root directory containing session folders with batch_* subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='finetuning_data',
        help='Output directory for pickle batch files'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='RN50x16',
        help='CLIP model name'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.15,
        help='Fraction of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_split',
        type=float,
        default=0.15,
        help='Fraction of data for testing (default: 0.15)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Overwrite existing batch files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    create_pickle_batches(
        data_root=args.data_root,
        output_dir=args.output_dir,
        clip_model_name=args.clip_model,
        batch_size=args.batch_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        recreate=args.recreate
    )

