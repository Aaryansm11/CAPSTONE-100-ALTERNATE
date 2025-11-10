#!/usr/bin/env python3
"""
Corrected Training Script for ECG/PPG Discovery
Uses proper SimCLR-style contrastive learning
"""

import torch
from torch.utils.data import DataLoader, random_split
import h5py
import pickle
import json
import numpy as np
from pathlib import Path

# Import corrected components
from corrected_contrastive_training import (
    CorrectedContrastiveLoss,
    StrongAugmentation,
    ContrastiveDataset,
    CorrectedTrainer
)
from contrastive_model import WaveformEncoder

class Config:
    """Configuration class"""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)

def load_dataset(h5_path, metadata_path):
    """Load preprocessed dataset from HDF5"""
    print("Loading dataset...")

    # Load segments from HDF5
    with h5py.File(h5_path, 'r') as h5f:
        segments = h5f['segments'][:]
        print(f"  ✓ Loaded {len(segments):,} segments")
        print(f"  ✓ Shape: {segments.shape}")
        print(f"  ✓ Size: {segments.nbytes / 1e9:.2f} GB")

    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        print(f"  ✓ Loaded metadata for {len(metadata):,} segments")

    return segments, metadata

def main():
    print("=" * 70)
    print("CORRECTED CONTRASTIVE TRAINING - ECG/PPG DISCOVERY")
    print("=" * 70)

    # Configuration
    config_path = "production_medium/config.json"
    config = Config(config_path)

    # Paths
    h5_path = "production_medium/full_dataset.h5"
    metadata_path = "production_medium/full_dataset_metadata.pkl"
    output_dir = "production_medium_corrected"

    Path(output_dir).mkdir(exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")

    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print(f"\n{'='*70}")
    segments, metadata = load_dataset(h5_path, metadata_path)

    # Create contrastive dataset with strong augmentation
    print(f"\n{'='*70}")
    print("Creating contrastive dataset...")
    augmentation = StrongAugmentation(
        noise_std=0.05,
        scale_range=(0.8, 1.2),
        shift_range=0.1,
        dropout_prob=0.2
    )

    full_dataset = ContrastiveDataset(
        segments=segments,
        metadata=metadata,
        augmentation=augmentation
    )

    print(f"  ✓ Dataset size: {len(full_dataset):,} samples")

    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  ✓ Train: {len(train_dataset):,} samples")
    print(f"  ✓ Val: {len(val_dataset):,} samples")

    # Create data loaders
    print(f"\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True  # Ensures consistent batch size
    )

    print(f"  ✓ Batch size: {config.batch_size}")
    print(f"  ✓ Batches per epoch: {len(train_loader)}")
    print(f"  ✓ Workers: {config.num_workers}")

    # Create model
    print(f"\n{'='*70}")
    print("Creating model...")

    model = WaveformEncoder(
        input_channels=8,
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: WaveformEncoder")
    print(f"  ✓ Parameters: {total_params:,}")
    print(f"  ✓ Embedding dim: {config.embedding_dim}")
    print(f"  ✓ Hidden dims: {config.hidden_dims}")

    # Create trainer
    print(f"\n{'='*70}")
    print("Creating trainer...")

    trainer = CorrectedTrainer(
        model=model,
        config=config,
        device=device
    )

    # Train!
    print(f"\n{'='*70}")
    train_losses = trainer.train(
        dataloader=train_loader,
        num_epochs=config.num_epochs,
        save_dir=output_dir
    )

    # Save training history
    history_path = f"{output_dir}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'config': config.__dict__
        }, f, indent=2)

    print(f"\n✓ Training history saved: {history_path}")

    # Test embedding diversity
    print(f"\n{'='*70}")
    print("Testing embedding diversity...")

    model.eval()
    with torch.no_grad():
        # Sample 100 segments
        sample_idx = np.random.choice(len(segments), 100, replace=False)
        sample_segments = torch.FloatTensor(segments[sample_idx])

        # Transpose to (batch, channels, seq_len)
        sample_segments = sample_segments.transpose(1, 2).to(device)

        # Generate embeddings
        embeddings = model(sample_segments)

        # Normalize
        embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)

        # Compute pairwise similarities
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        # Get off-diagonal (inter-sample) similarities
        mask = ~torch.eye(len(embeddings_norm), dtype=torch.bool, device=device)
        inter_sim = sim_matrix[mask]

        mean_sim = inter_sim.mean().item()
        std_sim = inter_sim.std().item()

        print(f"  • Mean inter-sample similarity: {mean_sim:.4f}")
        print(f"  • Std inter-sample similarity: {std_sim:.4f}")

        if mean_sim < 0.7:
            print(f"  ✓ GOOD: Embeddings are diverse!")
        elif mean_sim < 0.9:
            print(f"  ⚠ WARNING: Embeddings show moderate similarity")
        else:
            print(f"  ✗ ERROR: Representation collapse detected!")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {output_dir}/")
    print(f"Use checkpoint_epoch_final.pth for inference")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
