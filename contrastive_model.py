#!/usr/bin/env python3
"""
Contrastive Learning Model for ECG/PPG Arrhythmia Discovery
Implements self-supervised representation learning for waveform segments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class WaveformDataset(Dataset):
    """
    Dataset for loading ECG/PPG segments with augmentations
    """

    def __init__(self, segments, segment_metadata, augment=True):
        """
        Args:
            segments: numpy array of shape (num_segments, seq_len, num_channels)
            segment_metadata: list of metadata dicts for each segment
            augment: whether to apply data augmentations
        """
        self.segments = segments
        self.segment_metadata = segment_metadata
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        metadata = self.segment_metadata[idx]

        # Convert to torch tensor and transpose to (channels, seq_len)
        segment = torch.FloatTensor(segment).transpose(0, 1)

        if self.augment:
            # Apply augmentations
            segment = self.apply_augmentations(segment)

        return {
            'segment': segment,
            'patient_id': metadata['patient_id'],
            'has_ecg': metadata['has_ecg'],
            'has_ppg': metadata['has_ppg'],
        }

    def apply_augmentations(self, segment):
        """Apply time series augmentations"""
        # Noise injection
        if random.random() < 0.3:
            noise_scale = 0.02 * torch.std(segment)
            segment = segment + torch.randn_like(segment) * noise_scale

        # Time scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            seq_len = segment.shape[1]
            new_len = int(seq_len * scale_factor)
            if new_len != seq_len:
                segment = F.interpolate(segment.unsqueeze(0), size=new_len, mode='linear', align_corners=False).squeeze(0)
                # Crop or pad to original length
                if new_len > seq_len:
                    start = random.randint(0, new_len - seq_len)
                    segment = segment[:, start:start + seq_len]
                else:
                    pad_needed = seq_len - new_len
                    pad_left = random.randint(0, pad_needed)
                    pad_right = pad_needed - pad_left
                    segment = F.pad(segment, (pad_left, pad_right), mode='reflect')

        # Amplitude scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            segment = segment * scale_factor

        # Channel dropout (mask some channels)
        if random.random() < 0.2:
            num_channels = segment.shape[0]
            if num_channels > 1:
                mask_channels = random.randint(1, max(1, num_channels // 2))
                channels_to_mask = random.sample(range(num_channels), mask_channels)
                for ch in channels_to_mask:
                    segment[ch] = 0

        return segment

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
    """
    1D CNN encoder for ECG/PPG waveforms
    Outputs fixed-size embeddings for contrastive learning
    """

    def __init__(self,
                 input_channels=5,
                 hidden_dims=[64, 128, 256, 512],
                 embedding_dim=256,
                 dropout=0.1):
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
        """
        Args:
            x: (batch_size, channels, seq_len)
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # Initial convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for layer in self.layers:
            out = layer(out)

        # Global pooling
        out = self.global_pool(out).squeeze(-1)  # (batch_size, hidden_dims[-1])

        # Projection
        embeddings = self.projection_head(out)

        return embeddings

class ContrastiveLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2):
        """
        Args:
            embeddings1, embeddings2: (batch_size, embedding_dim)
                Two augmented views of the same batch
        """
        batch_size = embeddings1.shape[0]

        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # Concatenate embeddings
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (2*batch_size, embedding_dim)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create labels - positive pairs are at positions (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(2 * batch_size, device=embeddings.device)
        labels[:batch_size] += batch_size
        labels[batch_size:] -= batch_size

        # Mask to remove self-similarities
        mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss

class ContrastiveTrainer:
    """
    Trainer for contrastive learning on waveform data
    """

    def __init__(self,
                 model,
                 train_loader,
                 val_loader=None,
                 learning_rate=1e-3,
                 temperature=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = ContrastiveLoss(temperature=temperature)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch in pbar:
                segments = batch['segment'].to(self.device)  # (batch_size, channels, seq_len)

                # Create two augmented views
                segments1 = segments.clone()
                segments2 = segments.clone()

                # Apply different augmentations to each view
                for i in range(segments.shape[0]):
                    if random.random() < 0.5:
                        segments1[i] = self.apply_augmentation(segments1[i])
                    if random.random() < 0.5:
                        segments2[i] = self.apply_augmentation(segments2[i])

                # Forward pass
                embeddings1 = self.model(segments1)
                embeddings2 = self.model(segments2)

                # Compute loss
                loss = self.criterion(embeddings1, embeddings2)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        """Validate model"""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                segments = batch['segment'].to(self.device)

                # Create two views (same augmentation as training)
                segments1 = segments.clone()
                segments2 = segments.clone()

                embeddings1 = self.model(segments1)
                embeddings2 = self.model(segments2)

                loss = self.criterion(embeddings1, embeddings2)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def apply_augmentation(self, segment):
        """Apply single augmentation (used during training)"""
        # Simple noise augmentation
        noise_scale = 0.02 * torch.std(segment)
        return segment + torch.randn_like(segment) * noise_scale

    def train(self, num_epochs):
        """Train the model"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Print progress
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "/media/jaadoo/sexy/ecg ppg/best_model.pth")
                    print("Saved best model")

    def plot_losses(self):
        """Plot training curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig("/media/jaadoo/sexy/ecg ppg/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    # Load preprocessed dataset
    print("Loading dataset...")
    data = np.load("/media/jaadoo/sexy/ecg ppg/test_dataset.npz", allow_pickle=True)

    segments = data['segments']
    segment_metadata = data['segment_metadata']

    print(f"Dataset shape: {segments.shape}")
    print(f"Number of segments: {len(segments)}")

    # Split into train/val
    train_size = int(0.8 * len(segments))
    indices = np.random.permutation(len(segments))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_segments = segments[train_indices]
    train_metadata = [segment_metadata[i] for i in train_indices]

    val_segments = segments[val_indices]
    val_metadata = [segment_metadata[i] for i in val_indices]

    # Create datasets
    train_dataset = WaveformDataset(train_segments, train_metadata, augment=True)
    val_dataset = WaveformDataset(val_segments, val_metadata, augment=False)

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Disable multiprocessing for now
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    input_channels = segments.shape[2]  # Number of channels
    model = WaveformEncoder(
        input_channels=input_channels,
        hidden_dims=[64, 128, 256, 512],
        embedding_dim=256,
        dropout=0.1
    )

    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        temperature=0.1
    )

    # Train model
    trainer.train(num_epochs=20)

    # Plot results
    trainer.plot_losses()

    print("Training complete! Model saved to best_model.pth")

if __name__ == "__main__":
    main()