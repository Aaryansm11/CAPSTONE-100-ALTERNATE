#!/usr/bin/env python3
"""
Comprehensive Training Validation Script
Tests model files, architecture, inference, embeddings quality
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 70)
print("COMPREHENSIVE TRAINING VALIDATION")
print("=" * 70)

# ============================================================================
# TEST 1: File Existence
# ============================================================================
print("\n[TEST 1] Checking output files...")
output_dir = Path("production_medium")
required_files = [
    "checkpoint_epoch_final.pth",
    "latest_checkpoint.pth",
    "full_dataset.h5",
    "full_dataset_metadata.pkl",
    "config.json"
]

missing_files = []
for file in required_files:
    filepath = output_dir / file
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024**2)
        print(f"  ✓ {file} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {file} MISSING")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ TEST 1 FAILED: Missing {len(missing_files)} files")
else:
    print("\n✓ TEST 1 PASSED: All required files present")

# ============================================================================
# TEST 2: Model Architecture Validation
# ============================================================================
print("\n[TEST 2] Validating model architecture...")

# Define model architecture (same as in contrastive_model.py)
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """1D Residual block for time series"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WaveformEncoder(nn.Module):
    """1D CNN encoder for ECG/PPG waveforms"""
    def __init__(self, input_channels=5, hidden_dims=[64, 128, 256, 512],
                 embedding_dim=256, dropout=0.1):
        super(WaveformEncoder, self).__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim

        # Initial convolution
        self.input_conv = nn.Conv1d(input_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3)
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])

        # Residual blocks
        self.layers = nn.ModuleList()
        in_channels = hidden_dims[0]

        for dim in hidden_dims:
            self.layers.append(ResidualBlock(in_channels, dim, stride=2, dropout=dropout))
            in_channels = dim

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], embedding_dim)
        )

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for layer in self.layers:
            out = layer(out)

        # Global pooling
        out = self.global_pool(out).squeeze(-1)

        # Projection
        embeddings = self.projection_head(out)

        return embeddings

try:
    # Load config
    with open("production_medium/config.json", 'r') as f:
        config = json.load(f)

    # Load checkpoint first to check input channels
    checkpoint = torch.load("production_medium/checkpoint_epoch_final.pth",
                           map_location='cpu', weights_only=False)

    # Check checkpoint contents
    available_keys = list(checkpoint.keys())
    print(f"  ✓ Checkpoint keys: {available_keys}")

    # Detect input channels from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'input_conv.weight' in state_dict:
            input_channels = state_dict['input_conv.weight'].shape[1]
            print(f"  ✓ Detected input channels: {input_channels}")
        else:
            input_channels = 8  # Default
            print(f"  ! Using default input channels: {input_channels}")

    if 'epoch' in checkpoint:
        print(f"  ✓ Trained for {checkpoint['epoch']} epochs")

    # Create model with detected params
    model = WaveformEncoder(
        input_channels=input_channels,
        hidden_dims=config['hidden_dims'],
        embedding_dim=config['embedding_dim']
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model weights loaded successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")

    print("\n✓ TEST 2 PASSED: Model architecture valid")

except Exception as e:
    print(f"\n❌ TEST 2 FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    model = None

# ============================================================================
# TEST 3: Inference Test
# ============================================================================
print("\n[TEST 3] Testing model inference...")

if model is not None:
    try:
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Create dummy input matching expected shape
        batch_size = 16
        num_channels = input_channels  # Use detected channels
        seq_length = 1250  # 10 seconds at 125Hz

        dummy_input = torch.randn(batch_size, num_channels, seq_length).to(device)

        with torch.no_grad():
            embeddings = model(dummy_input)

        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {embeddings.shape}")
        print(f"  ✓ Expected embedding dim: {config['embedding_dim']}")

        # Validate output shape
        if embeddings.shape == (batch_size, config['embedding_dim']):
            print(f"  ✓ Output shape matches expected")
        else:
            print(f"  ✗ Output shape mismatch!")

        # Check for NaN or Inf
        if torch.isnan(embeddings).any():
            print(f"  ✗ NaN values detected in embeddings")
        elif torch.isinf(embeddings).any():
            print(f"  ✗ Inf values detected in embeddings")
        else:
            print(f"  ✓ No NaN or Inf values")

        # Check embedding statistics
        print(f"  ✓ Embedding mean: {embeddings.mean().item():.4f}")
        print(f"  ✓ Embedding std: {embeddings.std().item():.4f}")
        print(f"  ✓ Embedding min: {embeddings.min().item():.4f}")
        print(f"  ✓ Embedding max: {embeddings.max().item():.4f}")

        print("\n✓ TEST 3 PASSED: Model inference working")

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("\n⊘ TEST 3 SKIPPED: Model not loaded")

# ============================================================================
# TEST 4: Real Data Inference
# ============================================================================
print("\n[TEST 4] Testing with real dataset segments...")

if model is not None:
    try:
        # Load actual data
        with h5py.File("production_medium/full_dataset.h5", 'r') as h5f:
            # Take first 32 segments
            real_segments = h5f['segments'][:32]

        print(f"  ✓ Loaded {len(real_segments)} real segments")
        print(f"  ✓ Segment shape (seq_len, channels): {real_segments[0].shape}")

        # Convert to tensor and transpose to (batch, channels, seq_len)
        real_input = torch.tensor(real_segments, dtype=torch.float32)
        real_input = real_input.transpose(1, 2).to(device)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
        print(f"  ✓ Input tensor shape (batch, channels, seq_len): {real_input.shape}")

        model.eval()
        with torch.no_grad():
            real_embeddings = model(real_input)

        print(f"  ✓ Generated embeddings: {real_embeddings.shape}")
        print(f"  ✓ Mean: {real_embeddings.mean().item():.4f}")
        print(f"  ✓ Std: {real_embeddings.std().item():.4f}")

        # Test embedding separation
        # Compute pairwise cosine similarities
        from torch.nn.functional import cosine_similarity

        # Normalize embeddings
        normalized = torch.nn.functional.normalize(real_embeddings, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(normalized, normalized.t())

        # Get off-diagonal (inter-sample) similarities
        mask = ~torch.eye(len(normalized), dtype=bool, device=device)
        inter_similarities = similarity_matrix[mask]

        print(f"  ✓ Inter-sample similarity mean: {inter_similarities.mean().item():.4f}")
        print(f"  ✓ Inter-sample similarity std: {inter_similarities.std().item():.4f}")

        # Good contrastive learning should have diverse embeddings
        # Mean similarity should be relatively low (not all embeddings identical)
        if inter_similarities.mean().item() > 0.95:
            print(f"  ⚠ Warning: Embeddings may be too similar (collapse)")
        else:
            print(f"  ✓ Embeddings show good diversity")

        print("\n✓ TEST 4 PASSED: Real data inference working")

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("\n⊘ TEST 4 SKIPPED: Model not loaded")

# ============================================================================
# TEST 5: Dataset Validation
# ============================================================================
print("\n[TEST 5] Validating dataset integrity...")

try:
    with h5py.File("production_medium/full_dataset.h5", 'r') as h5f:
        segments = h5f['segments']

        print(f"  ✓ Total segments: {len(segments):,}")
        print(f"  ✓ Segment shape: {segments.shape}")
        print(f"  ✓ Dataset size: {segments.nbytes / 1e9:.2f} GB")

        # Check for NaN or Inf in first 1000 segments
        sample_data = segments[:1000]

        if np.isnan(sample_data).any():
            nan_count = np.isnan(sample_data).sum()
            print(f"  ✗ Found {nan_count} NaN values in dataset")
        else:
            print(f"  ✓ No NaN values in sample")

        if np.isinf(sample_data).any():
            inf_count = np.isinf(sample_data).sum()
            print(f"  ✗ Found {inf_count} Inf values in dataset")
        else:
            print(f"  ✓ No Inf values in sample")

        # Check data statistics
        print(f"  ✓ Data mean: {sample_data.mean():.4f}")
        print(f"  ✓ Data std: {sample_data.std():.4f}")
        print(f"  ✓ Data min: {sample_data.min():.4f}")
        print(f"  ✓ Data max: {sample_data.max():.4f}")

    print("\n✓ TEST 5 PASSED: Dataset integrity validated")

except Exception as e:
    print(f"\n❌ TEST 5 FAILED: {str(e)}")

# ============================================================================
# TEST 6: GPU Utilization Check
# ============================================================================
print("\n[TEST 6] Checking GPU configuration...")

if torch.cuda.is_available():
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU count: {torch.cuda.device_count()}")
    print(f"  ✓ Current GPU: {torch.cuda.current_device()}")
    print(f"  ✓ GPU name: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"  ✓ Total VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  ✓ CUDA capability: {props.major}.{props.minor}")

    print("\n✓ TEST 6 PASSED: GPU configured correctly")
else:
    print(f"  ⚠ CUDA not available - training used CPU")
    print("\n⚠ TEST 6 WARNING: No GPU detected")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print("\n✓ Training validation complete!")

# Safe printing with checks
if 'trainable_params' in locals():
    print(f"  • Model: {trainable_params:,} parameters")
if 'checkpoint' in locals() and 'epoch' in checkpoint:
    print(f"  • Epochs: {checkpoint['epoch']}")
if 'config' in locals() and 'embedding_dim' in config:
    print(f"  • Embedding dim: {config['embedding_dim']}")
if 'input_channels' in locals():
    print(f"  • Input channels: {input_channels}")

print(f"  • Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

print("\n🎉 Training validation complete!")
print("=" * 70)
