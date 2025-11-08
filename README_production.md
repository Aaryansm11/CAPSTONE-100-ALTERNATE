# Production ECG/PPG Discovery Pipeline

## Overview
Complete production-ready pipeline for training on the full 60GB MIMIC-III waveform dataset for arrhythmia pattern discovery and stroke risk prediction.

## Features

### 🚀 **Scalable Processing**
- **Memory-efficient HDF5 storage** for large datasets
- **Chunked processing** (500 patients at a time)
- **Multiprocessing** for parallel waveform processing
- **Automatic data quality control** and validation

### 📊 **Production Dataset**
- **All ~2,415 patients** with waveform data available
- **Clinical integration** with demographics, diagnoses, medications
- **Outcome labels**: Stroke, arrhythmia, mortality
- **Quality filtering**: Minimum segments per patient
- **Memory limits**: Configurable max segments per patient

### 🧠 **Robust Training**
- **Fixed contrastive learning** with numerical stability
- **Gradient clipping** and learning rate scheduling
- **Checkpointing** every N epochs
- **Distributed training** support
- **Comprehensive logging** and monitoring

### 📈 **Monitoring & Validation**
- **Real-time loss tracking**
- **Patient-level train/val splits** (no data leakage)
- **Clinical validation** metrics
- **Embedding visualization** tools

## Quick Start

### 1. Build Full Dataset
```bash
# Process all available patients (~60GB)
cd "/media/jaadoo/sexy/ecg ppg"
conda activate ecgppg

python production_pipeline.py --build-dataset --output-dir production_full

# Or limit to specific number for testing
python production_pipeline.py --build-dataset --max-patients 100 --output-dir production_test
```

### 2. Train Discovery Model
```bash
# Train on full dataset
python production_pipeline.py --train-only --output-dir production_full

# Resume from checkpoint
python production_pipeline.py --train-only --config production_full/config.json
```

### 3. Custom Configuration
```json
{
  "max_patients": null,
  "max_segments_per_patient": 1000,
  "batch_size": 64,
  "learning_rate": 3e-4,
  "num_epochs": 50,
  "embedding_dim": 256,
  "hidden_dims": [64, 128, 256, 512],
  "chunk_size": 500,
  "num_workers": 8
}
```

## Pipeline Architecture

### Data Processing Flow
```
Raw MIMIC Data (60GB)
    ↓
Patient Discovery (~2,415 patients)
    ↓
Clinical Data Integration
    ↓
Waveform Preprocessing (10s segments)
    ↓
Quality Control & Filtering
    ↓
HDF5 Dataset Storage
    ↓
Contrastive Learning Training
    ↓
Learned Embeddings for Discovery
```

### Expected Output Sizes

| Configuration | Patients | Estimated Segments | Dataset Size | Training Time |
|--------------|----------|-------------------|--------------|---------------|
| **Full Dataset** | ~2,415 | ~1-2 million | 60GB | ~2-3 days |
| **Large Test** | 500 | ~200k | 12GB | ~12 hours |
| **Medium Test** | 100 | ~40k | 2.5GB | ~2 hours |
| **Small Test** | 20 | ~8k | 500MB | ~20 minutes |

## Key Features for 60GB Processing

### 1. **Memory Management**
```python
# Chunked processing to avoid OOM
chunk_size = 500  # Process 500 patients at a time
max_segments_per_patient = 1000  # Limit memory per patient

# HDF5 compression
compression='gzip'  # Reduces storage by ~50%
```

### 2. **Patient Selection Strategy**
```python
# Quality-based selection
min_segments_per_patient = 10    # Skip patients with little data
max_patients = None              # Process all available

# Prioritize by data volume
patients.sort(key=lambda x: x['size_mb'], reverse=True)
```

### 3. **Clinical Integration**
```python
# Outcome labels extracted
stroke_codes = ['430', '431', '432', '433', '434', '435', '436', '437', '438']
arrhythmia_codes = ['427']  # Cardiac dysrhythmias

# Demographics
age, gender, mortality = extract_demographics(patient_id)
```

### 4. **Training Optimization**
```python
# Efficient data loading
pin_memory=True                  # Faster GPU transfer
num_workers=8                   # Parallel data loading

# Gradient stability
clip_grad_norm_(max_norm=1.0)   # Prevent exploding gradients
temperature=0.1                 # Stable contrastive learning
```

## Hardware Requirements

### Minimum
- **GPU**: RTX 3080/4080 (12GB VRAM)
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space (dataset + checkpoints)
- **CPU**: 8+ cores for data processing

### Recommended
- **GPU**: RTX 4090 (24GB VRAM) or A100
- **RAM**: 64GB system RAM
- **Storage**: 200GB SSD space
- **CPU**: 16+ cores with high clock speed

## Expected Results

### Dataset Statistics (Full 60GB)
- **Total Patients**: ~2,415
- **Total Segments**: ~1-2 million 10-second segments
- **Clinical Coverage**:
  - Stroke patients: ~40-60
  - Arrhythmia patients: ~130-200
  - Healthy controls: ~1,000+

### Discovery Outcomes
- **Novel arrhythmia patterns** via clustering
- **Clinical validation** with ICD codes
- **Risk stratification** features
- **Stroke prediction** pipeline ready

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `batch_size` or `max_segments_per_patient`
2. **Slow Processing**: Increase `num_workers` or use SSD storage
3. **NaN Loss**: Check data quality flags in preprocessing
4. **Checkpoint Corruption**: Enable more frequent saves

### Monitoring
```bash
# Check logs
tail -f production_full/production.log

# Monitor GPU usage
nvidia-smi -l 1

# Check dataset progress
ls -la production_full/*.h5
```

## Next Steps After Training

1. **Clustering & Discovery**
   ```bash
   python clustering_pipeline.py --embeddings production_full/embeddings.npz
   ```

2. **Clinical Validation**
   ```bash
   python validate_patterns.py --patterns discovered_patterns.json
   ```

3. **Stroke Risk Prediction**
   ```bash
   python stroke_prediction.py --features production_full/clinical_features.csv
   ```

## File Structure
```
production_full/
├── config.json              # Training configuration
├── full_dataset.h5          # HDF5 waveform dataset
├── full_dataset_metadata.pkl # Clinical metadata
├── production.log           # Processing logs
├── checkpoint_epoch_10.pth  # Training checkpoints
├── latest_checkpoint.pth    # Latest model
└── embeddings/              # Learned embeddings
    ├── train_embeddings.npz
    └── val_embeddings.npz
```

This production pipeline is ready to handle the complete 60GB dataset with proper memory management, clinical integration, and robust training for your capstone project's arrhythmia discovery objectives.