#!/usr/bin/env python3
"""
Test Embedding Diversity - Check for representation collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import numpy as np
from pathlib import Path

# Import model architecture
from contrastive_model import WaveformEncoder

print("=" * 70)
print("EMBEDDING DIVERSITY TEST")
print("=" * 70)

# Load config
with open("production_medium/config.json", 'r') as f:
    config = json.load(f)

# Load checkpoint
checkpoint = torch.load("production_medium/checkpoint_epoch_final.pth",
                       map_location='cpu', weights_only=False)

# Detect input channels
state_dict = checkpoint['model_state_dict']
input_channels = state_dict['input_conv.weight'].shape[1]

# Create and load model
model = WaveformEncoder(
    input_channels=input_channels,
    hidden_dims=config['hidden_dims'],
    embedding_dim=config['embedding_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print(f"\n Model loaded on {device}")
print(f" Input channels: {input_channels}")
print(f" Embedding dim: {config['embedding_dim']}")

# Load dataset and sample from different parts
print(f"\n Loading dataset samples...")
with h5py.File("production_medium/full_dataset.h5", 'r') as h5f:
    total_segments = len(h5f['segments'])

    # Sample segments from different positions
    sample_indices = np.linspace(0, total_segments-1, 500, dtype=int)
    segments = h5f['segments'][sample_indices]

print(f" Loaded {len(segments)} segments from across dataset")

# Convert and transpose
segments_tensor = torch.tensor(segments, dtype=torch.float32)
segments_tensor = segments_tensor.transpose(1, 2).to(device)  # (batch, channels, seq_len)

print(f" Input shape: {segments_tensor.shape}")

# Generate embeddings in batches
print(f"\n Generating embeddings...")
batch_size = 64
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(segments_tensor), batch_size):
        batch = segments_tensor[i:i+batch_size]
        emb = model(batch)
        all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0)
print(f" Generated {len(embeddings)} embeddings")

# Analyze diversity
print(f"\n Analyzing embedding diversity...")

# Normalize embeddings
normalized = F.normalize(embeddings, dim=1)

# Compute pairwise similarities (sample to avoid memory issues)
sample_size = min(200, len(normalized))
sample_idx = np.random.choice(len(normalized), sample_size, replace=False)
sample_emb = normalized[sample_idx]

similarity_matrix = torch.mm(sample_emb, sample_emb.t())

# Get off-diagonal (inter-sample) similarities
mask = ~torch.eye(len(sample_emb), dtype=bool)
inter_similarities = similarity_matrix[mask]

print(f"\n Similarity Statistics (n={sample_size} samples):")
print(f"   Mean similarity: {inter_similarities.mean().item():.4f}")
print(f"   Std similarity: {inter_similarities.std().item():.4f}")
print(f"   Min similarity: {inter_similarities.min().item():.4f}")
print(f"   Max similarity: {inter_similarities.max().item():.4f}")
print(f"   Median similarity: {inter_similarities.median().item():.4f}")

# Embedding statistics
print(f"\n Embedding Statistics:")
print(f"   Mean: {embeddings.mean().item():.4f}")
print(f"   Std: {embeddings.std().item():.4f}")
print(f"   L2 norm (avg): {embeddings.norm(dim=1).mean().item():.4f}")
print(f"   L2 norm (std): {embeddings.norm(dim=1).std().item():.4f}")

# Check for collapse
mean_sim = inter_similarities.mean().item()
if mean_sim > 0.99:
    print(f"\n WARNING: Representation collapse detected!")
    print(f"   All embeddings are nearly identical (sim={mean_sim:.4f})")
elif mean_sim > 0.90:
    print(f"\n  WARNING: High similarity ({mean_sim:.4f})")
    print(f"   Embeddings may lack diversity")
elif mean_sim > 0.70:
    print(f"\n GOOD: Moderate similarity ({mean_sim:.4f})")
    print(f"   Embeddings show reasonable diversity")
else:
    print(f"\n EXCELLENT: Low similarity ({mean_sim:.4f})")
    print(f"   Embeddings are highly diverse")

# Check variance across dimensions
dim_variance = embeddings.var(dim=0)
low_variance_dims = (dim_variance < 0.1).sum().item()
print(f"\n Dimension-wise variance:")
print(f"   Low variance dims (<0.1): {low_variance_dims}/{config['embedding_dim']}")
print(f"   Mean variance: {dim_variance.mean().item():.4f}")
print(f"   Std variance: {dim_variance.std().item():.4f}")

if low_variance_dims > config['embedding_dim'] * 0.5:
    print(f"    More than 50% dimensions have low variance")
else:
    print(f"   Good variance distribution across dimensions")

print("\n" + "=" * 70)
print("EMBEDDING DIVERSITY TEST COMPLETE")
print("=" * 70)
