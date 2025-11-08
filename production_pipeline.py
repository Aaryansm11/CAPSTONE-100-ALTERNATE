#!/usr/bin/env python3
"""
Production-Ready Full Dataset Pipeline for 60GB MIMIC Data
Scalable implementation for discovery training on complete dataset
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import pickle
from multiprocessing import Pool, cpu_count
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_preprocessing import WaveformPreprocessor
from robust_data_preprocessing import RobustWaveformPreprocessor
from contrastive_model import WaveformEncoder
from fixed_contrastive_training import FixedContrastiveLoss, FixedTrainer

class ProductionConfig:
    """Configuration for production training"""

    def __init__(self, config_path=None):
        # Data paths
        self.mimic_waveform_path = "/media/jaadoo/sexy/mimic_data"
        self.mimic_clinical_path = "/media/jaadoo/sexy/physionet.org/files/mimiciii/1.4"
        self.output_dir = "/media/jaadoo/sexy/ecg ppg/production"

        # Preprocessing parameters
        self.segment_length_sec = 10
        self.sampling_rate = 125
        self.filter_low = 0.5
        self.filter_high = 40.0
        self.overlap_ratio = 0.0

        # Training parameters
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.num_epochs = 50
        self.embedding_dim = 256
        self.hidden_dims = [64, 128, 256, 512]
        self.temperature = 0.1

        # Data selection
        self.max_patients = None  # None = all patients
        self.max_segments_per_patient = 10000  # Increased limit for maximum data extraction
        self.min_segments_per_patient = 10    # Quality threshold

        # Hardware
        self.num_workers = min(8, cpu_count())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Chunked processing
        self.chunk_size = 500  # Process 500 patients at a time
        self.save_frequency = 10  # Save every 10 epochs

        # Load from config if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_config(self, config_path):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class EfficientDatasetBuilder:
    """Memory-efficient dataset builder for large-scale processing"""

    def __init__(self, config):
        self.config = config
        self.preprocessor = RobustWaveformPreprocessor(
            segment_length_sec=config.segment_length_sec,
            sampling_rate=config.sampling_rate,
            filter_low=config.filter_low,
            filter_high=config.filter_high,
            overlap_ratio=config.overlap_ratio
        )

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{config.output_dir}/production.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def find_all_patients(self):
        """Find all patients with substantial waveform data"""
        self.logger.info("Scanning for patients with waveform data...")

        patients_found = []
        total_size_gb = 0

        for p_dir in sorted(Path(self.config.mimic_waveform_path).glob("p*")):
            if not p_dir.is_dir():
                continue

            for patient_dir in sorted(p_dir.glob("p*")):
                if not patient_dir.is_dir():
                    continue

                try:
                    patient_id = int(patient_dir.name[1:])

                    # Count waveform files
                    dat_files = [f for f in patient_dir.glob("*.dat") if not f.stem.endswith('n')]

                    if len(dat_files) >= self.config.min_segments_per_patient:
                        # Estimate size
                        size_mb = sum(f.stat().st_size for f in dat_files) / (1024 * 1024)
                        total_size_gb += size_mb / 1024

                        patients_found.append({
                            'patient_id': patient_id,
                            'folder_path': str(patient_dir),
                            'num_files': len(dat_files),
                            'size_mb': size_mb
                        })

                        # Early termination if max patients reached
                        if (self.config.max_patients and
                            len(patients_found) >= self.config.max_patients):
                            break

                except (ValueError, OSError) as e:
                    continue

            if (self.config.max_patients and
                len(patients_found) >= self.config.max_patients):
                break

        self.logger.info(f"Found {len(patients_found)} patients with {total_size_gb:.1f}GB total data")

        # Sort by data size (descending) for quality
        patients_found.sort(key=lambda x: x['size_mb'], reverse=True)

        return patients_found

    def load_clinical_data(self):
        """Load clinical data efficiently"""
        self.logger.info("Loading clinical data...")

        # Load only essential columns for memory efficiency
        self.patients_df = pd.read_csv(
            f"{self.config.mimic_clinical_path}/PATIENTS.csv.gz",
            usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'EXPIRE_FLAG']
        )

        self.diagnoses_df = pd.read_csv(
            f"{self.config.mimic_clinical_path}/DIAGNOSES_ICD.csv.gz",
            usecols=['SUBJECT_ID', 'ICD9_CODE']
        )

        self.logger.info(f"Loaded clinical data: {len(self.patients_df)} patients, {len(self.diagnoses_df)} diagnoses")

    def get_clinical_features(self, patient_id):
        """Extract clinical features efficiently"""
        try:
            # Demographics
            patient_info = self.patients_df[self.patients_df['SUBJECT_ID'] == patient_id].iloc[0]
            birth_year = int(patient_info['DOB'][:4])
            age = 2025 - birth_year if birth_year <= 2020 else 2025 - birth_year + 100

            # Diagnoses
            patient_diagnoses = self.diagnoses_df[self.diagnoses_df['SUBJECT_ID'] == patient_id]
            diagnosis_codes = patient_diagnoses['ICD9_CODE'].astype(str).tolist()

            # Outcome labels
            stroke_codes = ['430', '431', '432', '433', '434', '435', '436', '437', '438']
            arrhythmia_codes = ['427']

            has_stroke = any(code[:3] in stroke_codes for code in diagnosis_codes)
            has_arrhythmia = any(code[:3] in arrhythmia_codes for code in diagnosis_codes)

            return {
                'age': min(max(age, 0), 120),  # Reasonable bounds
                'gender': 1 if patient_info['GENDER'] == 'M' else 0,
                'mortality': int(patient_info['EXPIRE_FLAG']),
                'has_stroke': int(has_stroke),
                'has_arrhythmia': int(has_arrhythmia),
                'num_diagnoses': len(patient_diagnoses)
            }
        except:
            # Default values if patient not found
            return {
                'age': 50, 'gender': 0, 'mortality': 0,
                'has_stroke': 0, 'has_arrhythmia': 0, 'num_diagnoses': 0
            }

    def process_patient_chunk(self, patient_chunk):
        """Process a chunk of patients efficiently"""
        chunk_segments = []
        chunk_metadata = []

        for patient_info in tqdm(patient_chunk, desc="Processing chunk"):
            patient_id = patient_info['patient_id']
            patient_folder = Path(patient_info['folder_path'])

            # Get clinical features
            clinical_features = self.get_clinical_features(patient_id)

            # Find waveform files
            dat_files = [f for f in patient_folder.glob("*.dat") if not f.stem.endswith('n')]

            # Process all available files per patient for maximum data extraction

            patient_segments = []
            patient_segment_count = 0

            for dat_file in dat_files:
                if patient_segment_count >= self.config.max_segments_per_patient:
                    break

                record_name = dat_file.stem
                record_path = patient_folder / record_name

                try:
                    result = self.preprocessor.process_record(str(record_path))

                    if result and result['num_good_segments'] > 0:
                        # Use all available segments from this file
                        segments = result['segments']

                        patient_segments.append(segments)

                        # Add metadata
                        for i in range(len(segments)):
                            chunk_metadata.append({
                                'patient_id': patient_id,
                                'record_name': record_name,
                                'segment_idx': i,
                                **clinical_features
                            })
                            patient_segment_count += 1

                            if patient_segment_count >= self.config.max_segments_per_patient:
                                break

                except Exception as e:
                    self.logger.debug(f"Error processing {record_path}: {e}")
                    continue

            # Combine patient segments
            if patient_segments:
                try:
                    combined = np.concatenate(patient_segments, axis=0)
                    chunk_segments.append(combined)
                except Exception as e:
                    self.logger.debug(f"Error combining segments for patient {patient_id}: {e}")

        return chunk_segments, chunk_metadata

    def build_hdf5_dataset(self, output_path):
        """Build dataset and save to HDF5 for efficient loading"""
        self.logger.info("Building production dataset...")

        # Load clinical data
        self.load_clinical_data()

        # Find all patients
        all_patients = self.find_all_patients()

        # Process in chunks
        total_segments = 0
        total_patients_processed = 0

        with h5py.File(output_path, 'w') as h5f:
            # Pre-allocate datasets (estimate sizes)
            max_segments = len(all_patients) * self.config.max_segments_per_patient
            segment_shape = (self.config.segment_length_sec * self.config.sampling_rate, 8)  # Max 8 channels

            segments_dset = h5f.create_dataset(
                'segments',
                (max_segments,) + segment_shape,
                dtype=np.float32,
                chunks=True,
                compression='gzip'
            )

            # Metadata will be stored separately
            metadata_list = []

            # Process in chunks
            for chunk_start in range(0, len(all_patients), self.config.chunk_size):
                chunk_end = min(chunk_start + self.config.chunk_size, len(all_patients))
                patient_chunk = all_patients[chunk_start:chunk_end]

                self.logger.info(f"Processing chunk {chunk_start//self.config.chunk_size + 1}/{(len(all_patients)-1)//self.config.chunk_size + 1}")

                chunk_segments, chunk_metadata = self.process_patient_chunk(patient_chunk)

                # Store segments in HDF5
                for segments in chunk_segments:
                    if len(segments) > 0:
                        # Pad channels if necessary
                        if segments.shape[2] < segment_shape[1]:
                            pad_width = ((0, 0), (0, 0), (0, segment_shape[1] - segments.shape[2]))
                            segments = np.pad(segments, pad_width, mode='constant')

                        # Store in HDF5
                        end_idx = total_segments + len(segments)
                        segments_dset[total_segments:end_idx] = segments
                        total_segments += len(segments)

                # Store metadata
                metadata_list.extend(chunk_metadata)
                total_patients_processed += len(patient_chunk)

                # Log progress
                self.logger.info(f"Processed {total_patients_processed}/{len(all_patients)} patients, {total_segments:,} segments")

            # Resize dataset to actual size
            segments_dset.resize((total_segments,) + segment_shape)

            # Store metadata
            h5f.attrs['total_segments'] = total_segments
            h5f.attrs['total_patients'] = total_patients_processed
            h5f.attrs['config'] = json.dumps(self.config.__dict__, default=str)

        # Save metadata separately
        with open(output_path.replace('.h5', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_list[:total_segments], f)

        self.logger.info(f"Dataset saved: {total_segments:,} segments from {total_patients_processed} patients")
        return total_segments, total_patients_processed

class ProductionDataset(Dataset):
    """Production dataset that loads from HDF5"""

    def __init__(self, h5_path, metadata_path, train=True):
        self.h5_path = h5_path
        self.train = train

        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Keep HDF5 file handle
        self.h5f = None
        self._open_h5()

    def _open_h5(self):
        if self.h5f is None or not self.h5f:
            self.h5f = h5py.File(self.h5_path, 'r')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        self._open_h5()

        # Load segment
        segment = self.h5f['segments'][idx].astype(np.float32)
        metadata = self.metadata[idx]

        # Convert to tensor and transpose
        segment_tensor = torch.FloatTensor(segment).transpose(0, 1)

        # Normalize
        for i in range(segment_tensor.shape[0]):
            if torch.std(segment_tensor[i]) > 1e-6:
                segment_tensor[i] = (segment_tensor[i] - torch.mean(segment_tensor[i])) / torch.std(segment_tensor[i])

        return {
            'segment': segment_tensor,
            'patient_id': metadata['patient_id'],
            'has_stroke': metadata['has_stroke'],
            'has_arrhythmia': metadata['has_arrhythmia']
        }

    def __del__(self):
        if hasattr(self, 'h5f') and self.h5f:
            self.h5f.close()

class ProductionTrainer:
    """Production trainer with checkpointing and monitoring"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Create model
        self.model = WaveformEncoder(
            input_channels=8,  # Max channels
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            dropout=0.1
        ).to(self.device)

        # Loss and optimizer
        self.criterion = FixedContrastiveLoss(temperature=config.temperature)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # Tracking
        self.train_losses = []
        self.epoch = 0

        self.logger.info(f"Model created: {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def train(self, train_loader, val_loader=None):
        """Train the model"""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step()

            # Log progress
            log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f}"
            if val_loss:
                log_msg += f", Val Loss: {val_loss:.4f}"
            log_msg += f", LR: {self.optimizer.param_groups[0]['lr']:.6f}"

            self.logger.info(log_msg)

            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(epoch + 1)

        # Final save
        self._save_checkpoint('final')

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch {self.epoch+1}") as pbar:
            for batch in pbar:
                segments = batch['segment'].to(self.device)

                # Skip if invalid data
                if torch.isnan(segments).any():
                    continue

                try:
                    # Forward pass
                    embeddings1 = self.model(segments)

                    # Augmented version
                    segments_aug = segments + 0.01 * torch.randn_like(segments)
                    embeddings2 = self.model(segments_aug)

                    # Loss
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

                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except Exception as e:
                    continue

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                segments = batch['segment'].to(self.device)

                if torch.isnan(segments).any():
                    continue

                try:
                    embeddings = self.model(segments)
                    loss = torch.mean(torch.norm(embeddings, dim=1))  # Simple validation loss

                    total_loss += loss.item()
                    num_batches += 1

                except:
                    continue

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'config': self.config.__dict__
        }

        checkpoint_path = f"{self.config.output_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Also save as latest
        latest_path = f"{self.config.output_dir}/latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Production ECG/PPG Discovery Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--build-dataset', action='store_true', help='Build dataset first')
    parser.add_argument('--train-only', action='store_true', help='Train only (dataset exists)')
    parser.add_argument('--max-patients', type=int, help='Maximum patients to process')
    parser.add_argument('--output-dir', type=str, default='/media/jaadoo/sexy/ecg ppg/production')

    args = parser.parse_args()

    # Create config
    config = ProductionConfig(args.config)
    if args.max_patients:
        config.max_patients = args.max_patients
    if args.output_dir:
        config.output_dir = args.output_dir

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config
    config.save_config(f"{config.output_dir}/config.json")

    # Build dataset if requested
    dataset_path = f"{config.output_dir}/full_dataset.h5"
    metadata_path = f"{config.output_dir}/full_dataset_metadata.pkl"

    if args.build_dataset or not os.path.exists(dataset_path):
        print("Building dataset...")
        builder = EfficientDatasetBuilder(config)
        total_segments, total_patients = builder.build_hdf5_dataset(dataset_path)

    # Train if requested
    if not args.build_dataset:
        print("Starting training...")

        # Load dataset
        dataset = ProductionDataset(dataset_path, metadata_path)

        # Create train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Train
        trainer = ProductionTrainer(config)
        trainer.train(train_loader, val_loader)

        print("Training complete!")

if __name__ == "__main__":
    main()