#!/usr/bin/env python3
"""
Fine-tuning script for EmotionPredictor models.

This script loads a pre-trained model checkpoint and fine-tunes it on custom data
stored in pickle batch format.
"""

import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from emotionpredictor.training import Trainer, SLP
from emotionpredictor.data_tools import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP model embedding sizes
EMBEDDING_SIZES = {
    "C-RN50": 1024,
    "C-RN101": 512,
    "C-RN50x4": 640,
    "C-RN50x16": 768,
    "C-ViT-B32": 512,
    "C-ViT-B16": 512,
    "I-alexnet": 4096,
    "I-resnet50": 2048
}


def get_embedding_size(checkpoint_name: str, data_loaders: dict) -> int:
    """Extract embedding size from checkpoint name or infer from data."""
    # First, try to infer from data (most reliable)
    if "train" in data_loaders:
        try:
            first_batch = data_loaders["train"].load_batch(0)
            detected_size = first_batch["image"].shape[1]
            print(f"Inferred embedding size from data: {detected_size}")
            return detected_size
        except Exception as e:
            print(f"Warning: Could not infer from data: {e}")
    
    # Fallback: Try to get from known sizes (check longest matches first)
    # Sort by key length descending to match most specific first
    sorted_keys = sorted(EMBEDDING_SIZES.items(), key=lambda x: len(x[0]), reverse=True)
    for key, size in sorted_keys:
        if checkpoint_name.startswith(key) or key in checkpoint_name:
            print(f"Matched checkpoint name '{checkpoint_name}' to '{key}' -> size {size}")
            return size
    
    raise ValueError(f"Could not determine embedding size for {checkpoint_name}")


def finetune_model(
    checkpoint_path: str,
    data_path: str,
    output_path: str,
    learning_rate: float = 1e-3,
    epochs: int = 20,
    use_validation: bool = True,
    test_network: bool = True
):
    """
    Fine-tune a pre-trained EmotionPredictor model.
    
    Args:
        checkpoint_path: Path to pre-trained checkpoint (e.g., "neural_checkpoints/C-RN50x16")
        data_path: Path to fine-tuning data directory (must have train/ and val/ subdirectories)
        output_path: Where to save the fine-tuned model
        learning_rate: Learning rate for fine-tuning (default: 1e-3)
        epochs: Number of epochs to train
        use_validation: If True, use train_eval with validation monitoring
        test_network: If True, test network capacity before training
    """
    
    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Load data loaders
    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        raise ValueError(f"Data directory {data_path} does not exist")
    
    loaders = get_loaders(str(data_path))
    
    # Check required splits
    if "train" not in loaders:
        raise ValueError("Data directory must contain a 'train' subdirectory")
    if use_validation and "val" not in loaders:
        print("Warning: No validation set found. Using train_eval requires 'val' directory.")
        use_validation = False
    
    # Determine embedding size
    checkpoint_name = checkpoint_path.name
    embedding_size = get_embedding_size(checkpoint_name, loaders)
    print(f"Detected embedding size: {embedding_size}")
    
    # Create model
    print(f"Creating model with input_size={embedding_size}, output_size=9")
    model = SLP(input_size=embedding_size, output_size=9).to(device)
    
    # Load pre-trained weights
    if checkpoint_path.exists():
        print(f"Loading pre-trained weights from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✓ Pre-trained weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Training from scratch...")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Training from scratch.")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer_fn=torch.optim.Adam,
        lr=learning_rate,
        data_loaders=loaders,
        device=device
    )
    
    # Test network (optional - verifies model can learn)
    if test_network:
        print("\nTesting network capacity...")
        try:
            trainer.test_network(test_epochs=20, target_loss=0.5)
            print("✓ Network test passed")
        except AssertionError as e:
            print(f"Warning: Network test failed: {e}")
            print("Continuing anyway...")
        except Exception as e:
            print(f"Warning: Network test error: {e}")
            print("Continuing anyway...")
    
    # Fine-tune
    print(f"\n{'='*60}")
    print(f"Starting fine-tuning for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    if use_validation:
        # Use train_eval for validation monitoring
        print("Using validation monitoring...")
        epochs_per_lr = max(1, epochs // 3)  # Split epochs across learning rates
        lrs = [learning_rate, learning_rate * 0.1, learning_rate * 0.01]
        trainer.train_eval(
            epochs_per_lr=epochs_per_lr,
            lrs=lrs
        )
    else:
        # Simple training
        trainer.train_n_epochs(epochs)
    
    # Save fine-tuned model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    print(f"\n{'='*60}")
    print(f"✓ Fine-tuned model saved to {output_path}")
    print(f"{'='*60}")
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune EmotionPredictor model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='neural_checkpoints/C-RN50x16',
        help='Path to pre-trained checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='finetuning_data',
        help='Path to fine-tuning data directory (with train/, val/, test/ subdirectories)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='neural_checkpoints/finetuned_model',
        help='Path to save fine-tuned model'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs to train (default: 20)'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation monitoring (use simple training)'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip network capacity test'
    )
    
    args = parser.parse_args()
    
    finetune_model(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        learning_rate=args.lr,
        epochs=args.epochs,
        use_validation=not args.no_validation,
        test_network=not args.no_test
    )

