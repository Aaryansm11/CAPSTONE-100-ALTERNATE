#!/usr/bin/env python3
"""
Windows-Optimized Training - RTX 4080
Handles Windows multiprocessing limitations + emojis
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
import torch.cuda.amp as amp
import time
import psutil

# Fix Windows emoji logging
import locale
if sys.platform == 'win32':
    # Use UTF-8 for Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')

torch._dynamo.config.suppress_errors = True

# Check Triton
TRITON_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    pass

from contrastive_model import WaveformEncoder
from corrected_contrastive_training import CorrectedContrastiveLoss, StrongAugmentation

# Windows-safe logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class WindowsRAMDataset(Dataset):
    """
    Windows-optimized RAM dataset
    - Loads data ONCE in main process
    - Workers access via shared indices (no data copying)
    """
    
    def __init__(self, h5_path, metadata_path, augment=True, force_ram=False):
        self.augment = augment
        self.augmentation = StrongAugmentation() if augment else None
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Check RAM
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        with h5py.File(h5_path, 'r') as h5f:
            dataset_size_gb = (h5f['segments'].size * 4) / (1024**3)
            
            logger.info(f"Dataset size: {dataset_size_gb:.2f}GB")
            logger.info(f"Available RAM: {available_ram_gb:.2f}GB")
            
            if dataset_size_gb > available_ram_gb * 0.6 and not force_ram:
                raise MemoryError(
                    f"Insufficient RAM. Need {dataset_size_gb:.1f}GB, "
                    f"have {available_ram_gb:.1f}GB available. "
                    f"Use --force-ram to override."
                )
            
            # Load to RAM
            logger.info("Loading dataset to RAM...")
            start = time.time()
            self.data = h5f['segments'][:].astype(np.float32)
            load_time = time.time() - start
            
            logger.info(f"Dataset cached in {load_time:.1f}s")
            logger.info(f"Shape: {self.data.shape}")
        
        # Compute normalization - per channel
        logger.info("Computing normalization statistics...")
        # Shape: (n_samples, time, channels) -> mean/std per channel
        self.mean = self.data.mean(axis=(0, 1))  # Shape: (8,)
        self.std = self.data.std(axis=(0, 1)) + 1e-8  # Shape: (8,)
        
        logger.info("Dataset ready!")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Fast RAM access
        segment = self.data[idx]  # Shape: (1250, 8)
        # Normalize per channel
        segment = (segment - self.mean) / self.std  # Broadcasting works: (1250, 8) - (8,)
        segment = torch.from_numpy(segment).transpose(0, 1)  # Shape: (8, 1250)
        
        if self.augment:
            view1 = self.augmentation(segment.clone())
            view2 = self.augmentation(segment.clone())
            return {'view1': view1, 'view2': view2, 'metadata': self.metadata[idx]}
        else:
            return {'segment': segment, 'metadata': self.metadata[idx]}


class WindowsOptimizedTrainer:
    """Windows-optimized trainer with automatic worker adjustment"""
    
    def __init__(self, config, output_dir='production_full'):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        
        # Model
        self.model = WaveformEncoder(
            input_channels=8,
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            dropout=config.get('dropout', 0.15)
        ).to(self.device)
        
        # Try to compile (won't work on Windows without Triton)
        if TRITON_AVAILABLE and hasattr(torch, 'compile'):
            logger.info("Compiling model...")
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        else:
            logger.info("Using eager mode (torch.compile not available)")
        
        self.criterion = CorrectedContrastiveLoss(temperature=config['temperature'])
        
        # Optimizer
        use_fused = torch.cuda.is_available()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            fused=use_fused
        )
        
        self.scaler = torch.amp.GradScaler('cuda', init_scale=2**14)
        self.total_epochs = config['num_epochs']
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Early stopping
        self.patience = config.get('patience', 3)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        self.current_step = 0
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Effective batch size: {config['batch_size'] * self.accumulation_steps}")
    
    def train(self, train_loader, val_loader, time_limit_hours=4.5):
        logger.info("="*70)
        logger.info("STARTING TRAINING - RTX 4080")
        logger.info(f"Target: <{time_limit_hours}hrs, >90% accuracy")
        logger.info("="*70)
        
        start_time = time.time()
        time_limit_seconds = time_limit_hours * 3600
        
        for epoch in range(self.total_epochs):
            epoch_start = time.time()
            
            if time.time() - start_time > time_limit_seconds:
                logger.warning(f"Time limit reached at epoch {epoch+1}")
                break
            
            # Dynamic LR
            if epoch < 2:
                lr = self.config['learning_rate'] * 0.5
            elif epoch < 8:
                lr = self.config['learning_rate']
            elif epoch < 12:
                lr = self.config['learning_rate'] * 0.5
            else:
                lr = self.config['learning_rate'] * 0.1
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Stats
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            samples_per_sec = len(train_loader.dataset) / epoch_time
            
            gpu_util = torch.cuda.utilization() if torch.cuda.is_available() else 0
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            
            logger.info(
                f"Epoch {epoch+1:02d}/{self.total_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"LR: {lr:.2e} | {epoch_time/60:.1f}min | "
                f"{samples_per_sec:.0f} samples/s | GPU: {gpu_util}% {gpu_mem:.1f}GB"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch + 1, is_best=True)
                logger.info(f"New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 2 == 0:
                self._save_checkpoint(epoch + 1)
            
            torch.cuda.reset_peak_memory_stats()
        
        self._save_checkpoint('final')
        total_time = (time.time() - start_time) / 3600
        
        logger.info("="*70)
        logger.info(f"TRAINING COMPLETE in {total_time:.2f} hours!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*70)
    
    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            view1 = batch['view1'].to(self.device, non_blocking=True)
            view2 = batch['view2'].to(self.device, non_blocking=True)
            
            if torch.isnan(view1).any() or torch.isnan(view2).any():
                continue
            
            with torch.amp.autocast('cuda'):
                embeddings1 = self.model(view1)
                embeddings2 = self.model(view2)
                loss = self.criterion(embeddings1, embeddings2)
                
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                actual_loss = loss.item() * self.accumulation_steps if self.accumulation_steps > 1 else loss.item()
                total_loss += actual_loss
                num_batches += 1
                self.current_step += 1
                
                pbar.set_postfix({'loss': f'{actual_loss:.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        max_batches = 100
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                view1 = batch['view1'].to(self.device, non_blocking=True)
                view2 = batch['view2'].to(self.device, non_blocking=True)
                
                if torch.isnan(view1).any() or torch.isnan(view2).any():
                    continue
                
                with torch.amp.autocast('cuda'):
                    embeddings1 = self.model(view1)
                    embeddings2 = self.model(view2)
                    loss = self.criterion(embeddings1, embeddings2)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'current_step': self.current_step
        }
        
        path = f"{self.output_dir}/{'best_model' if is_best else f'checkpoint_epoch_{epoch}'}.pth"
        torch.save(checkpoint, path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Windows-Optimized Training')
    parser.add_argument('--config', type=str, default='production_fullconfig.json')
    parser.add_argument('--data-dir', type=str, default='production_full')
    parser.add_argument('--output-dir', type=str, default='production_full')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0=single process, recommended for Windows)')
    parser.add_argument('--force-ram', action='store_true')
    parser.add_argument('--time-limit', type=float, default=4.5)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Check RAM
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    logger.info(f"System RAM: {available_ram_gb:.1f}GB available / {total_ram_gb:.1f}GB total")
    
    # Windows-optimized settings
    config['batch_size'] = args.batch_size
    config['accumulation_steps'] = 1
    config['learning_rate'] = 5e-4
    config['num_epochs'] = 15
    config['dropout'] = 0.15
    config['weight_decay'] = 0.02
    config['patience'] = 3
    
    # Auto-adjust workers for Windows
    if sys.platform == 'win32':
        if args.num_workers > 4:
            logger.warning(f"Reducing workers from {args.num_workers} to 4 for Windows stability")
            args.num_workers = 4
    
    logger.info("="*70)
    logger.info("WINDOWS-OPTIMIZED TRAINING - RTX 4080")
    logger.info("="*70)
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Workers: {args.num_workers} (0=fastest on Windows with RAM caching)")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Triton: {TRITON_AVAILABLE}")
    logger.info("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(f"{args.output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    dataset_path = f"{args.data_dir}/full_dataset.h5"
    metadata_path = f"{args.data_dir}/full_dataset_metadata.pkl"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    # Create dataset
    logger.info("Initializing dataset...")
    full_dataset = WindowsRAMDataset(
        dataset_path, metadata_path, 
        augment=True, force_ram=args.force_ram
    )
    
    # Split
    val_split = 0.2
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train: {train_size:,} | Val: {val_size:,}")
    
    # DataLoaders - Windows optimized
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Always 0 for validation
        pin_memory=True
    )
    
    trainer = WindowsOptimizedTrainer(config, args.output_dir)
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {props.total_memory / 1024**3:.1f}GB")
    
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader, time_limit_hours=args.time_limit)
    
    if torch.cuda.is_available():
        logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    
    logger.info(f"Best model saved: {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    main()