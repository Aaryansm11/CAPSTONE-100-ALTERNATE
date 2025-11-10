#!/usr/bin/env python3
"""
CORRECTED Contrastive Training Module for ECG/PPG Discovery
Implements proper SimCLR-style NT-Xent loss with augmented pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import random

class CorrectedContrastiveLoss(nn.Module):
    """
    CORRECTED NT-Xent loss for SimCLR-style contrastive learning

    Proper implementation:
    - Takes TWO views of the same batch (view1, view2)
    - Positive pairs: (view1[i], view2[i]) - same sample, different augmentations
    - Negative pairs: all other samples in the batch
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: embeddings from view 1, shape (batch_size, embedding_dim)
            z_j: embeddings from view 2, shape (batch_size, embedding_dim)

        Returns:
            Contrastive loss (scalar)
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views: [z_i; z_j] -> shape (2*batch_size, embedding_dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix for all pairs
        # shape: (2*batch_size, 2*batch_size)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        ) / self.temperature

        # Create mask to identify positive pairs
        # Positive pairs are: (i, i+batch_size) and (i+batch_size, i)
        positives_mask = torch.zeros((2 * batch_size, 2 * batch_size),
                                     dtype=torch.bool, device=z_i.device)

        for i in range(batch_size):
            positives_mask[i, batch_size + i] = True
            positives_mask[batch_size + i, i] = True

        # Create mask to exclude self-similarities (diagonal)
        negatives_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)

        # For each sample, extract positive and negative similarities
        loss = 0.0
        for i in range(2 * batch_size):
            # Positive similarity (only one positive pair per sample)
            pos_sim = similarity_matrix[i][positives_mask[i]]

            # Negative similarities (all other samples)
            neg_sim = similarity_matrix[i][negatives_mask[i] & ~positives_mask[i]]

            # Compute NT-Xent loss for this sample
            # log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # = pos - log(exp(pos) + sum(exp(neg)))
            # = pos - log_sum_exp([pos, neg])
            logits = torch.cat([pos_sim, neg_sim])
            loss -= pos_sim - torch.logsumexp(logits, dim=0)

        # Average over all samples
        loss = loss / (2 * batch_size)

        return loss


class StrongAugmentation:
    """
    Strong data augmentation for time-series waveforms
    Multiple augmentations applied with high probability
    """

    def __init__(self, noise_std=0.05, scale_range=(0.8, 1.2),
                 shift_range=0.1, dropout_prob=0.2):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        """
        Apply random augmentations to input tensor

        Args:
            x: input tensor, shape (channels, seq_len)

        Returns:
            Augmented tensor, same shape
        """
        # 1. Gaussian Noise (always applied)
        if random.random() < 0.7:
            noise = torch.randn_like(x) * self.noise_std * x.std()
            x = x + noise

        # 2. Amplitude Scaling (always applied)
        if random.random() < 0.7:
            scale = random.uniform(*self.scale_range)
            x = x * scale

        # 3. Time Shifting
        if random.random() < 0.5:
            seq_len = x.shape[1]
            shift = int(random.uniform(-self.shift_range, self.shift_range) * seq_len)
            if shift != 0:
                x = torch.roll(x, shift, dims=1)

        # 4. Channel Dropout
        if random.random() < 0.3:
            num_channels = x.shape[0]
            if num_channels > 1:
                # Dropout 1-2 channels
                num_drop = random.randint(1, min(2, num_channels - 1))
                channels_to_drop = random.sample(range(num_channels), num_drop)
                x[channels_to_drop] = 0

        # 5. Time Masking
        if random.random() < 0.3:
            seq_len = x.shape[1]
            mask_len = int(random.uniform(0.05, 0.15) * seq_len)
            mask_start = random.randint(0, seq_len - mask_len)
            x[:, mask_start:mask_start + mask_len] = 0

        # 6. Gaussian Blur (smoothing)
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            padding = kernel_size // 2
            x = F.avg_pool1d(x.unsqueeze(0), kernel_size, stride=1, padding=padding).squeeze(0)

        return x


class ContrastiveDataset(Dataset):
    """
    Dataset wrapper that applies augmentations for contrastive learning
    Returns two differently augmented views of each sample
    """

    def __init__(self, segments, metadata, augmentation=None):
        """
        Args:
            segments: numpy array or torch tensor, shape (N, seq_len, channels)
            metadata: list of metadata dicts
            augmentation: augmentation function to apply
        """
        self.segments = torch.FloatTensor(segments) if isinstance(segments, np.ndarray) else segments
        self.metadata = metadata
        self.augmentation = augmentation or StrongAugmentation()

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # Get base segment
        segment = self.segments[idx].clone()
        metadata = self.metadata[idx]

        # Ensure shape is (channels, seq_len)
        if segment.dim() == 2 and segment.shape[0] > segment.shape[1]:
            segment = segment.transpose(0, 1)

        # Handle NaN/Inf
        segment = torch.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per channel
        for i in range(segment.shape[0]):
            if segment[i].std() > 1e-6:
                segment[i] = (segment[i] - segment[i].mean()) / segment[i].std()

        # Create two augmented views of the SAME segment
        view1 = self.augmentation(segment.clone())
        view2 = self.augmentation(segment.clone())

        return {
            'view1': view1,
            'view2': view2,
            'patient_id': metadata.get('patient_id', 'unknown'),
            'has_stroke': metadata.get('has_stroke', 0),
            'has_arrhythmia': metadata.get('has_arrhythmia', 0)
        }


class CorrectedTrainer:
    """
    Corrected trainer for SimCLR-style contrastive learning
    """

    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # CORRECTED loss with higher temperature
        self.criterion = CorrectedContrastiveLoss(temperature=0.5)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
        )

        self.train_losses = []
        self.current_epoch = 0

        print(f"✓ Corrected trainer initialized")
        print(f"  • Temperature: 0.5 (higher to prevent collapse)")
        print(f"  • Learning rate: {config.learning_rate}")
        print(f"  • Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, dataloader):
        """Train one epoch with corrected contrastive learning"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            try:
                # Get two views of the same samples
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)

                # Check for NaN
                if torch.isnan(view1).any() or torch.isnan(view2).any():
                    continue

                # Forward pass through both views
                z1 = self.model(view1)
                z2 = self.model(view2)

                # Check for NaN embeddings
                if torch.isnan(z1).any() or torch.isnan(z2).any():
                    print(f"  Warning: NaN embeddings at batch {batch_idx}")
                    continue

                # Compute contrastive loss (CORRECTED)
                loss = self.criterion(z1, z2)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"  Warning: NaN loss at batch {batch_idx}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Track loss
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })

            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, dataloader, num_epochs, save_dir):
        """Full training loop"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"STARTING CORRECTED CONTRASTIVE TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            avg_loss = self.train_epoch(dataloader)
            self.train_losses.append(avg_loss)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  • Loss: {avg_loss:.4f}")
            print(f"  • LR: {current_lr:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(save_dir, epoch + 1)

        # Save final model
        self.save_checkpoint(save_dir, 'final')

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"Best loss: {min(self.train_losses):.4f} (epoch {self.train_losses.index(min(self.train_losses))+1})")

        return self.train_losses

    def save_checkpoint(self, save_dir, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'config': self.config.__dict__
        }

        checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Also save as latest
        latest_path = f"{save_dir}/latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
