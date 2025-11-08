# RTX 4080 Optimization Configuration
*ECG/PPG Discovery System - GPU Optimization Guide*

## 🚀 RTX 4080 vs MX570 A Performance Comparison

### Current MX570 A Performance (2GB VRAM)
- **Batch Size**: 16 (limited by VRAM)
- **Training Speed**: ~3.5 seconds/batch
- **Total Training Time**: ~25 hours for 66,432 segments
- **Memory Usage**: 1.8GB/2GB VRAM (90% utilization)
- **Segments/Second**: ~4.6

### Expected RTX 4080 Performance (16GB VRAM)
- **Batch Size**: 128+ (8x larger batches)
- **Training Speed**: ~0.5 seconds/batch (estimated)
- **Total Training Time**: ~2-3 hours (8-10x faster)
- **Memory Usage**: 8-12GB/16GB VRAM (comfortable headroom)
- **Segments/Second**: ~256

## ⚙️ RTX 4080 Configuration Changes

### 1. Batch Size Optimization
```python
# In production_pipeline.py - update these parameters:

# BEFORE (MX570 A):
train_config = {
    'batch_size': 16,
    'num_workers': 2,
    'pin_memory': True
}

# AFTER (RTX 4080):
train_config = {
    'batch_size': 128,      # 8x larger batches
    'num_workers': 8,       # More CPU workers
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 4
}
```

### 2. Model Architecture Optimization
```python
# In contrastive_model.py - enable larger embeddings:

class WaveformEncoder(nn.Module):
    def __init__(self, embedding_dim=512):  # Increase from 128 to 512
        # ... rest of model

# Enable mixed precision training:
from torch.cuda.amp import autocast, GradScaler

# In fixed_contrastive_training.py:
class FixedTrainer:
    def __init__(self, model, device='cuda'):
        # ... existing init
        self.scaler = GradScaler()  # Enable AMP
```

### 3. Memory Management
```python
# In production_pipeline.py:

# Increase chunk sizes for RTX 4080:
chunk_config = {
    'max_patients_per_chunk': 2000,     # Increase from 500
    'max_segments_per_patient': 50000,  # Increase from 10000
    'chunk_size_gb': 8                  # Increase from 2GB
}

# Enable CUDA memory optimization:
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## 🔧 Quick Start Commands for RTX 4080

### 1. Medium Training (Optimized)
```bash
# Set RTX 4080 environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4080 architecture

# Run optimized medium training
./run_production_rtx4080.sh medium
```

### 2. Full Training (60GB Dataset)
```bash
# Full scale with RTX 4080 optimization
./run_production_rtx4080.sh full
```

## 📊 Expected Performance Improvements

| Metric | MX570 A (2GB) | RTX 4080 (16GB) | Improvement |
|--------|---------------|-----------------|-------------|
| Batch Size | 16 | 128 | 8x |
| Training Time | 25 hours | 2-3 hours | 8-10x |
| Memory Utilization | 90% | 60-75% | Safer |
| Segments/Hour | 2,650 | 22,000+ | 8x+ |
| Model Size | Limited | Full Scale | Unlimited |

## 🛠️ Files to Update for RTX 4080

### Priority 1: Core Training Files
1. **production_pipeline.py** - Batch size and memory config
2. **fixed_contrastive_training.py** - AMP and larger batches
3. **contrastive_model.py** - Larger embedding dimensions
4. **run_production.sh** - GPU-specific optimizations

### Priority 2: Configuration Files
1. **requirements.txt** - Ensure latest PyTorch with CUDA 11.8+
2. **robust_data_preprocessing.py** - Larger chunk processing

## 🎯 Immediate Actions on RTX 4080

1. **Install optimized PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Update configurations** (see sections above)

3. **Test with small batch first**:
   ```bash
   python test_rtx4080_performance.py
   ```

4. **Run medium training**:
   ```bash
   ./run_production_rtx4080.sh medium
   ```

5. **Monitor GPU utilization**:
   ```bash
   nvidia-smi -l 1
   ```

## 📈 Expected Training Timeline on RTX 4080

- **Medium (66,432 segments)**: 2-3 hours
- **Full (500,000+ segments)**: 12-15 hours
- **Validation & Testing**: 30 minutes
- **Stroke Prediction Training**: 1 hour

**Total Project Completion**: 24-48 hours vs 200+ hours on MX570 A