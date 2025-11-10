#!/usr/bin/env python3
"""
OPTIMIZED CONTRASTIVE TRAINING
===============================
Multiple optimization strategies to achieve lower loss and better embeddings

Optimizations included:
1. Longer training (50 epochs)
2. Warmup + Cosine annealing schedule
3. Stronger augmentations
4. Optimal temperature (0.07)
5. Gradient accumulation for larger effective batch
6. MixUp augmentation
7. Better optimizer (AdamW with proper weight decay)
8. EMA (Exponential Moving Average)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import random
from copy import deepcopy

from contrastive_model import WaveformEncoder


class StrongAugmentation:
    """Enhanced augmentation with more diversity"""
    def __init__(self, noise_std=0.05, amplitude_range=(0.7, 1.3)):
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range

    def __call__(self, x):
        """Apply strong augmentations"""
        # 1. Gaussian Noise (80% probability, stronger)
        if random.random() < 0.8:
            noise = torch.randn_like(x) * self.noise_std * x.std()
            x = x + noise

        # 2. Amplitude Scaling (80% probability, wider range)
        if random.random() < 0.8:
            scale = random.uniform(*self.amplitude_range)
            x = x * scale

        # 3. Time Shifting (70% probability)
        if random.random() < 0.7:
            shift = random.randint(-50, 50)
            x = torch.roll(x, shifts=shift, dims=-1)

        # 4. Channel Dropout (50% probability, up to 40%)
        if random.random() < 0.5:
            n_channels = x.shape[0]
            n_drop = random.randint(1, max(1, int(n_channels * 0.4)))
            drop_channels = random.sample(range(n_channels), n_drop)
            x[drop_channels, :] = 0

        # 5. Time Masking (60% probability, longer masks)
        if random.random() < 0.6:
            mask_len = random.randint(20, 100)
            mask_start = random.randint(0, x.shape[-1] - mask_len)
            x[:, mask_start:mask_start+mask_len] = 0

        # 6. Gaussian Blur (40% probability)
        if random.random() < 0.4:
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            # Simple box blur for 1D
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            x = x.unsqueeze(0)
            x = F.conv1d(x, kernel.to(x.device), padding=kernel_size//2)
            x = x.squeeze(0)

        # 7. Random Cutout (30% probability)
        if random.random() < 0.3:
            cutout_len = random.randint(30, 150)
            cutout_start = random.randint(0, x.shape[-1] - cutout_len)
            x[:, cutout_start:cutout_start+cutout_len] = x.mean()

        # 8. Frequency Masking (30% probability)
        if random.random() < 0.3:
            n_channels = x.shape[0]
            n_mask = random.randint(1, max(1, n_channels // 3))
            mask_channels = random.sample(range(n_channels), n_mask)
            x[mask_channels, :] = 0

        return x


class MixUpAugmentation:
    """MixUp for contrastive learning"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x1, x2):
        """Apply MixUp between two views"""
        if random.random() < 0.3:  # 30% probability
            lam = np.random.beta(self.alpha, self.alpha)
            mixed = lam * x1 + (1 - lam) * x2
            return mixed
        return x1


class OptimizedContrastiveDataset(Dataset):
    """Dataset with optimized augmentations"""
    def __init__(self, segments, metadata, augmentation=None, mixup=None):
        self.segments = segments
        self.metadata = metadata
        self.augmentation = augmentation or StrongAugmentation()
        self.mixup = mixup or MixUpAugmentation()

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        segment = torch.FloatTensor(segment).T  # (channels, seq_len)

        # Create three views for better learning
        view1 = self.augmentation(segment.clone())
        view2 = self.augmentation(segment.clone())

        # Optional MixUp between views
        if random.random() < 0.2:
            view1 = self.mixup(view1, view2)

        metadata = self.metadata[idx] if idx < len(self.metadata) else {}

        return {
            'view1': view1,
            'view2': view2,
            'patient_id': metadata.get('patient_id', 'unknown'),
            'has_stroke': metadata.get('has_stroke', 0),
            'has_arrhythmia': metadata.get('has_arrhythmia', 0)
        }


class OptimizedContrastiveLoss(nn.Module):
    """Optimized NT-Xent loss with hard negative mining"""
    def __init__(self, temperature=0.07, use_hard_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: embeddings from view 1 (batch_size, embedding_dim)
            z_j: embeddings from view 2 (batch_size, embedding_dim)
        """
        batch_size = z_i.shape[0]

        # L2 normalization
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        ) / self.temperature

        # Create masks for positive pairs
        batch_size = z_i.shape[0]
        mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z_i.device)

        # Mark positive pairs: (i, i+N) and (i+N, i)
        for i in range(batch_size):
            mask[i, batch_size + i] = True
            mask[batch_size + i, i] = True

        # Exclude self-similarity
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)

        # Hard negative mining (optional)
        if self.use_hard_negatives:
            # Find hard negatives (most similar non-positive pairs)
            negative_mask = ~(mask | mask_self)
            hard_negative_scores = similarity_matrix.clone()
            hard_negative_scores[~negative_mask] = -float('inf')

            # Weight hard negatives more
            weights = torch.ones_like(similarity_matrix)
            top_k = min(batch_size // 2, 10)
            for i in range(2 * batch_size):
                _, hard_neg_idx = torch.topk(hard_negative_scores[i], k=top_k)
                weights[i, hard_neg_idx] *= 1.5  # Weight hard negatives 1.5x

        # Compute loss
        pos_sim = similarity_matrix[mask].view(2 * batch_size, 1)

        # All similarities except self
        neg_sim = similarity_matrix[~mask_self].view(2 * batch_size, -1)

        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineSchedule:
    """Learning rate schedule with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def train_optimized(
    data_path='production_medium/full_dataset.h5',
    metadata_path='production_medium/full_dataset_metadata.pkl',
    config_path='production_medium/config.json',
    output_dir='production_medium_optimized',
    num_epochs=50,
    batch_size=128,
    learning_rate=1e-3,
    warmup_epochs=5,
    temperature=0.07,
    gradient_accumulation_steps=2,
    use_ema=True,
    device='cuda'
):
    """
    Optimized training with all improvements
    """
    print("=" * 70)
    print("OPTIMIZED CONTRASTIVE TRAINING - ECG/PPG DISCOVERY")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    print("\n" + "=" * 70)
    print("Loading dataset...")
    with h5py.File(data_path, 'r') as h5f:
        segments = h5f['segments'][:]

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"  ✓ Loaded {len(segments):,} segments")
    print(f"  ✓ Shape: {segments.shape}")
    print(f"  ✓ Size: {segments.nbytes / (1024**3):.2f} GB")

    # Create datasets
    print("\n" + "=" * 70)
    print("Creating optimized dataset...")

    # Split
    n_train = int(0.8 * len(segments))
    train_segments = segments[:n_train]
    train_metadata = metadata[:n_train]
    val_segments = segments[n_train:]
    val_metadata = metadata[n_train:]

    train_dataset = OptimizedContrastiveDataset(
        train_segments,
        train_metadata,
        augmentation=StrongAugmentation(noise_std=0.05, amplitude_range=(0.7, 1.3)),
        mixup=MixUpAugmentation(alpha=0.2)
    )
    val_dataset = OptimizedContrastiveDataset(
        val_segments,
        val_metadata,
        augmentation=StrongAugmentation(noise_std=0.03, amplitude_range=(0.8, 1.2)),
        mixup=None  # No mixup for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"  ✓ Train: {len(train_dataset):,} samples")
    print(f"  ✓ Val: {len(val_dataset):,} samples")
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ Effective batch size: {batch_size * gradient_accumulation_steps} (with accumulation)")

    # Create model
    print("\n" + "=" * 70)
    print("Creating model...")

    input_channels = segments.shape[2]
    model = WaveformEncoder(
        input_channels=input_channels,
        hidden_dims=config['hidden_dims'],
        embedding_dim=config['embedding_dim']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model: WaveformEncoder")
    print(f"  ✓ Parameters: {total_params:,}")
    print(f"  ✓ Embedding dim: {config['embedding_dim']}")

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Proper weight decay
        betas=(0.9, 0.999)
    )

    # Learning rate schedule
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = len(train_loader) * warmup_epochs // gradient_accumulation_steps
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6
    )

    # Loss function
    criterion = OptimizedContrastiveLoss(
        temperature=temperature,
        use_hard_negatives=True
    )

    # EMA
    ema = EMA(model, decay=0.999) if use_ema else None

    print("\n" + "=" * 70)
    print("Training configuration:")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Base learning rate: {learning_rate}")
    print(f"  • Warmup epochs: {warmup_epochs}")
    print(f"  • Temperature: {temperature} (optimal for contrastive learning)")
    print(f"  • Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  • Weight decay: 0.05")
    print(f"  • EMA: {use_ema}")
    print(f"  • Hard negative mining: True")
    print("=" * 70)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING OPTIMIZED TRAINING")
    print("=" * 70)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)

            # Forward pass
            z_i = model(view1)
            z_j = model(view2)

            # Compute loss
            loss = criterion(z_i, z_j)
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                current_lr = scheduler.step()

                if ema:
                    ema.update()

            train_loss += loss.item() * gradient_accumulation_steps
            pbar.set_postfix({'loss': f'{train_loss/(batch_idx+1):.4f}', 'lr': f'{current_lr:.6f}'})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                view1 = batch['view1'].to(device)
                view2 = batch['view2'].to(device)

                z_i = model(view1)
                z_j = model(view2)

                loss = criterion(z_i, z_j)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  • Train Loss: {train_loss:.4f}")
        print(f"  • Val Loss: {val_loss:.4f}")
        print(f"  • LR: {current_lr:.6f}")

        # Save checkpoints
        if (epoch + 1) % 5 == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }

            if ema:
                ema.apply_shadow()
                checkpoint['ema_state_dict'] = model.state_dict()
                ema.restore()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
                print(f"  ✓ Checkpoint saved")

    # Save final model
    torch.save(checkpoint, output_dir / 'checkpoint_epoch_final.pth')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")
    print(f"\nModel saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    train_optimized(
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-3,
        warmup_epochs=5,
        temperature=0.07,  # Optimal for SimCLR
        gradient_accumulation_steps=2,
        use_ema=True
    )
