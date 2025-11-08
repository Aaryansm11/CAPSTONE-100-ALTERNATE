#!/usr/bin/env python3
"""
Fixed Contrastive Training Module for ECG/PPG Discovery
Implements NT-Xent loss and robust training for self-supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

class FixedWaveformDataset(Dataset):
    """Fixed dataset with proper data handling"""

    def __init__(self, segments, segment_metadata, clinical_features=None, augment=True):
        self.segments = torch.FloatTensor(segments)
        self.segment_metadata = segment_metadata
        self.clinical_features = clinical_features
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].clone()
        metadata = self.segment_metadata[idx]

        # Ensure segment is (channels, seq_len)
        if segment.dim() == 2 and segment.shape[0] > segment.shape[1]:
            segment = segment.transpose(0, 1)

        # Handle NaN values
        segment = torch.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply augmentations if requested
        if self.augment:
            segment = self.apply_augmentations(segment)

        return {
            'segment': segment,
            'patient_id': metadata.get('patient_id', 'unknown'),
            'segment_idx': metadata.get('segment_idx', idx),
            'clinical_category': metadata.get('clinical_category', 'unknown')
        }

    def apply_augmentations(self, segment):
        """Apply data augmentations"""
        # Gaussian noise
        if random.random() < 0.3:
            noise = torch.normal(0, 0.01, size=segment.shape)
            segment = segment + noise

        # Amplitude scaling
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            segment = segment * scale

        # Time shifting
        if random.random() < 0.3:
            shift = random.randint(-10, 10)
            if shift != 0:
                segment = torch.roll(segment, shift, dims=1)

        return segment

class FixedContrastiveLoss(nn.Module):
    """NT-Xent loss for contrastive learning"""

    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: optional labels for monitoring
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Create positive pairs by splitting batch in half
        if batch_size % 2 != 0:
            embeddings = embeddings[:-1]
            batch_size = embeddings.shape[0]

        z_i = embeddings[:batch_size//2]
        z_j = embeddings[batch_size//2:]

        # Compute similarity matrices
        sim_i_j = torch.mm(z_i, z_j.t()) / self.temperature
        sim_j_i = torch.mm(z_j, z_i.t()) / self.temperature

        # Positive pairs
        positive_samples = torch.diag(sim_i_j)

        # Negative samples for z_i
        sim_i_i = torch.mm(z_i, z_i.t()) / self.temperature
        mask_i = torch.eye(z_i.shape[0], device=device).bool()
        sim_i_i.masked_fill_(mask_i, -float('inf'))
        negative_i = torch.cat([sim_i_j, sim_i_i], dim=1)

        # Negative samples for z_j
        sim_j_j = torch.mm(z_j, z_j.t()) / self.temperature
        mask_j = torch.eye(z_j.shape[0], device=device).bool()
        sim_j_j.masked_fill_(mask_j, -float('inf'))
        negative_j = torch.cat([sim_j_i, sim_j_j], dim=1)

        # Compute losses
        loss_i = -positive_samples + torch.logsumexp(negative_i, dim=1)
        loss_j = -positive_samples + torch.logsumexp(negative_j, dim=1)

        loss = (loss_i + loss_j).mean()
        return loss

class FixedTrainer:
    """Fixed trainer with robust error handling"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = FixedContrastiveLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            try:
                segments = batch['segment'].to(self.device)
                batch_size = segments.shape[0]

                # Ensure even batch size for contrastive learning
                if batch_size % 2 != 0:
                    segments = segments[:-1]

                if segments.shape[0] < 2:
                    continue

                # Forward pass
                embeddings = self.model(segments)

                # Handle NaN embeddings
                if torch.isnan(embeddings).any():
                    print(f"NaN embeddings detected in batch {batch_idx}")
                    continue

                # Compute loss
                loss = self.criterion(embeddings)

                # Handle NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss in batch {batch_idx}")
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        self.scheduler.step()

        return avg_loss

    def train(self, dataloader, num_epochs=25, save_path=None):
        """Full training loop"""
        train_losses = []

        print(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)
            train_losses.append(avg_loss)

            print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {self.scheduler.get_last_lr()[0]:.2e}')

            # Save checkpoint every 5 epochs
            if save_path and epoch % 5 == 0:
                checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch}.pth"
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        # Save final model
        if save_path:
            final_path = f"{save_path}/final_model.pth"
            self.save_model(final_path)
            print(f"Final model saved: {final_path}")

        return train_losses

    def save_model(self, path):
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def create_contrastive_pairs(segments):
    """Create contrastive pairs from segments"""
    # Simple approach: treat each segment as both anchor and positive
    # This works for self-supervised learning where we want the model
    # to learn general representations
    pairs = []
    for i, segment in enumerate(segments):
        pairs.append((segment, segment))  # Self-pair for now
    return pairs

def plot_training_results(losses, save_path=None):
    """Plot training curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}/training_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Test function
def test_contrastive_training():
    """Test the contrastive training setup"""
    print("Testing contrastive training...")

    # Create dummy data
    batch_size = 64
    seq_len = 1250
    num_channels = 4

    dummy_segments = np.random.randn(100, seq_len, num_channels)
    dummy_metadata = [{'patient_id': f'p{i}', 'segment_idx': i, 'clinical_category': 'test'} for i in range(100)]

    # Create dataset
    dataset = FixedWaveformDataset(dummy_segments, dummy_metadata)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model (assuming we have WaveformEncoder)
    from contrastive_model import WaveformEncoder
    model = WaveformEncoder()

    # Create trainer
    trainer = FixedTrainer(model)

    # Train for a few epochs
    losses = trainer.train(dataloader, num_epochs=3)

    print(f"Training completed. Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    test_contrastive_training()