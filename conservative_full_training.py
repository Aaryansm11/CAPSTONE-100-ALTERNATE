#!/usr/bin/env python3
"""
Conservative Full Dataset Training
Prevents overfitting with proper regularization and validation
Uses corrected contrastive learning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import h5py
import pickle
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Import corrected modules
from contrastive_model import WaveformEncoder
from corrected_contrastive_training import CorrectedContrastiveLoss, StrongAugmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conservative_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ConservativeDataset(Dataset):
    """Dataset with strong augmentation for contrastive learning - Memory efficient"""

    def __init__(self, h5_path, metadata_path, augment=True):
        self.augment = augment
        self.augmentation = StrongAugmentation() if augment else None
        self.h5_path = h5_path

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Store dataset info but use lazy loading to save RAM
        with h5py.File(h5_path, 'r') as h5f:
            self.n_segments = h5f['segments'].shape[0]
            self.segment_shape = h5f['segments'].shape[1:]

        logger.info(f"Dataset initialized: {self.n_segments:,} segments (lazy loading)")

        # Open h5 file for this worker
        self.h5_file = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Lazy load segment from h5 file
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        segment = self.h5_file['segments'][idx].astype(np.float32)
        segment = torch.FloatTensor(segment).transpose(0, 1)

        # Normalize each channel
        for i in range(segment.shape[0]):
            if torch.std(segment[i]) > 1e-6:
                segment[i] = (segment[i] - torch.mean(segment[i])) / torch.std(segment[i])

        if self.augment:
            # Create two augmented views
            view1 = self.augmentation(segment.clone())
            view2 = self.augmentation(segment.clone())
            return {'view1': view1, 'view2': view2, 'metadata': self.metadata[idx]}
        else:
            return {'segment': segment, 'metadata': self.metadata[idx]}


class ConservativeTrainer:
    """Conservative trainer with overfitting prevention"""

    def __init__(self, config, output_dir='production_full'):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model with dropout
        self.model = WaveformEncoder(
            input_channels=8,
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            dropout=config.get('dropout', 0.2)  # Prevent overfitting
        ).to(self.device)

        # Loss with optimal temperature
        self.criterion = CorrectedContrastiveLoss(temperature=config['temperature'])

        # AdamW with weight decay (better regularization)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.05),  # Strong regularization
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        self.warmup_epochs = 5
        self.total_epochs = config['num_epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_epochs - self.warmup_epochs
        )

        # Early stopping
        self.use_early_stopping = config.get('use_early_stopping', True)
        self.patience = config.get('patience', 5)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.embedding_diversities = []

        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Device: {self.device}")
        logger.info(f"Conservative settings: dropout={config.get('dropout', 0.2)}, "
                   f"weight_decay={config.get('weight_decay', 0.05)}, "
                   f"early_stopping={self.use_early_stopping}")

    def train(self, train_loader, val_loader):
        """Train with validation and early stopping"""
        logger.info(f"Starting conservative training for {self.total_epochs} epochs")

        for epoch in range(self.total_epochs):
            # Warmup learning rate
            if epoch < self.warmup_epochs:
                warmup_lr = self.config['learning_rate'] * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            # Train epoch
            train_loss, train_diversity = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.embedding_diversities.append(train_diversity)

            # Validate
            val_loss, val_diversity = self._validate_epoch(val_loader)
            self.val_losses.append(val_loss)

            # Update learning rate after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Log progress
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{self.total_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Diversity: {train_diversity:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if self.use_early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    self._save_checkpoint(epoch + 1, is_best=True)
                    logger.info(f"✓ New best validation loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"Patience: {self.patience_counter}/{self.patience}")

                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch + 1)

        # Final save
        self._save_checkpoint('final')
        logger.info("Training complete!")

        # Save training history
        self._save_history()

    def _train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # For diversity calculation
        all_embeddings = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
            for batch in pbar:
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)

                # Skip if invalid
                if torch.isnan(view1).any() or torch.isnan(view2).any():
                    continue

                try:
                    # Forward pass
                    embeddings1 = self.model(view1)
                    embeddings2 = self.model(view2)

                    # Contrastive loss
                    loss = self.criterion(embeddings1, embeddings2)

                    if torch.isnan(loss):
                        continue

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Store embeddings for diversity calculation
                    if num_batches % 10 == 0:  # Sample every 10 batches
                        all_embeddings.append(embeddings1.detach().cpu())

                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except Exception as e:
                    logger.warning(f"Training batch error: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)

        # Calculate embedding diversity
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            normalized = F.normalize(all_embeddings, dim=1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            # Mean similarity (excluding diagonal)
            mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
            diversity = similarity_matrix[mask].abs().mean().item()
        else:
            diversity = 0.0

        return avg_loss, diversity

    def _validate_epoch(self, val_loader):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_embeddings = []

        with torch.no_grad():
            for batch in val_loader:
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)

                if torch.isnan(view1).any() or torch.isnan(view2).any():
                    continue

                try:
                    embeddings1 = self.model(view1)
                    embeddings2 = self.model(view2)

                    loss = self.criterion(embeddings1, embeddings2)

                    if torch.isnan(loss):
                        continue

                    total_loss += loss.item()
                    num_batches += 1

                    # Store for diversity
                    all_embeddings.append(embeddings1.cpu())

                except:
                    continue

        avg_loss = total_loss / max(num_batches, 1)

        # Calculate diversity
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            normalized = F.normalize(all_embeddings, dim=1)
            similarity_matrix = torch.mm(normalized, normalized.t())
            mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
            diversity = similarity_matrix[mask].abs().mean().item()
        else:
            diversity = 0.0

        return avg_loss, diversity

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'embedding_diversities': self.embedding_diversities,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }

        checkpoint_path = f"{self.output_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = f"{self.output_dir}/best_model.pth"
            torch.save(checkpoint, best_path)

    def _save_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'embedding_diversities': self.embedding_diversities,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        with open(f"{self.output_dir}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Conservative Full Dataset Training')
    parser.add_argument('--config', type=str, default='production_fullconfig.json',
                       help='Config file path')
    parser.add_argument('--data-dir', type=str, default='production_full',
                       help='Directory with dataset')
    parser.add_argument('--output-dir', type=str, default='production_full',
                       help='Output directory')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(f"{args.output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load dataset
    dataset_path = f"{args.data_dir}/full_dataset.h5"
    metadata_path = f"{args.data_dir}/full_dataset_metadata.pkl"

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Build the dataset first using production_pipeline.py")
        return

    logger.info("Initializing datasets...")
    full_dataset = ConservativeDataset(dataset_path, metadata_path, augment=True)

    # Split
    val_split = config.get('validation_split', 0.2)
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    logger.info(f"Train: {train_size:,} segments | Val: {val_size:,} segments")

    # Multi-worker loading for better performance
    num_workers = 6
    logger.info(f"Using {num_workers} workers for data loading")

    # Create loaders with prefetching for better GPU utilization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],  # Use batch size from config
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Stable batch sizes
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Train
    trainer = ConservativeTrainer(config, args.output_dir)
    trainer.train(train_loader, val_loader)

    logger.info(f"✓ Training complete! Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()

