# Windows Setup Guide - CAPSTONE Project

## Overview
This document describes the Windows environment setup and configuration changes made to migrate the CAPSTONE ECG/PPG Discovery System from Linux to Windows.

**Date:** November 9, 2025
**Platform:** Windows 11
**GPU:** NVIDIA GeForce RTX 4080 (16GB VRAM)
**Python:** 3.11.8

---

## Environment Setup

### Virtual Environment
- **Location:** `C:\Users\ANT-PC\Documents\PROJECT-AARYAN\venv`
- **Activation:** `. venv/Scripts/activate` (from project root)
- **Python Version:** 3.11.8

### GPU Configuration
- **GPU:** NVIDIA GeForce RTX 4080
- **VRAM:** 16GB
- **CUDA Version:** 12.6
- **Driver Version:** 560.94
- **PyTorch Version:** 2.5.1+cu121

### Performance Benchmarks (RTX 4080)
```
Optimal Batch Size: 256
Throughput: 1,705.8 samples/sec
VRAM Usage: 0.78GB @ batch 256
Training Time (25 epochs): ~18 minutes (0.3 hours)
Speedup vs MX570 A: 92.4x faster
```

---

## Dependencies Installed

All required packages have been installed:

```bash
# Core ML/DL
torch==2.5.1+cu121
torchvision==0.20.1+cu121
transformers==4.56.2
tokenizers==0.22.1

# Scientific Computing
numpy==2.1.2
pandas==2.3.2
scipy==1.16.2

# Machine Learning
scikit-learn==1.7.2
xgboost==3.1.1
optuna==4.5.0

# Medical Data
wfdb==4.3.0

# Clustering & Visualization
umap-learn==0.5.9
hdbscan==0.8.40
matplotlib==3.10.7
seaborn==0.13.2
plotly==6.4.0

# Data Storage
h5py==3.15.1

# Utilities
tqdm==4.67.1
```

---

## Path Configuration Changes

### Updated Files

#### 1. `production_medium/config.json`
**Backup:** `production_medium/config.json.backup`

**Changes:**
- Updated MIMIC waveform path: `C:/MIMIC-III/waveform_data`
- Updated MIMIC clinical path: `C:/MIMIC-III/clinical_data/mimiciii/1.4`
- Output directory: `production_medium` (relative)
- **Optimized for RTX 4080:**
  - Batch size: 48 → 96 (can go up to 256)
  - Num workers: 4 → 8
  - Embedding dim: 256 (unchanged)

```json
{
    "mimic_waveform_path": "C:/MIMIC-III/waveform_data",
    "mimic_clinical_path": "C:/MIMIC-III/clinical_data/mimiciii/1.4",
    "output_dir": "production_medium",
    "max_patients": 100,
    "max_segments_per_patient": 1000,
    "batch_size": 96,
    "learning_rate": 3e-4,
    "num_epochs": 25,
    "embedding_dim": 256,
    "hidden_dims": [64, 128, 256, 512],
    "temperature": 0.1,
    "chunk_size": 500,
    "num_workers": 8,
    "save_frequency": 5
}
```

#### 2. `production_pipeline.py`
**Changes:**
- Default paths in `ProductionConfig.__init__()`:
  - `mimic_waveform_path`: Windows path
  - `mimic_clinical_path`: Windows path
  - `output_dir`: Relative path
- Updated `--output-dir` default argument

#### 3. `stroke_prediction.py`
**Changes:**
- Updated hardcoded paths to use relative paths from `production_medium/`
- Data path: `production_medium/integrated_dataset.npz`
- Model path: `production_medium/best_fixed_model.pth`
- Clustering path: `production_medium/simple_pattern_discovery/pattern_results.json`

#### 4. `simple_clustering.py`
**Changes:**
- Updated hardcoded paths to use relative paths
- Data path: `production_medium/integrated_dataset.npz`
- Model path: `production_medium/best_fixed_model.pth`

#### 5. `publication_report.py`
**Changes:**
- Default `project_dir` changed from `/media/jaadoo/sexy/ecg ppg` to `.` (current directory)
- Now uses relative paths for all file operations

---

## Directory Structure

Created the following directories:
```
CAPSTONE-100-ALTERNATE/
├── production_medium/
│   ├── checkpoints/           # Model checkpoints
│   ├── visualizations/        # Training plots and graphs
│   ├── simple_pattern_discovery/  # Pattern analysis results
│   ├── config.json           # Configuration file
│   ├── config.json.backup    # Original backup
│   ├── build.log            # Dataset building log
│   └── train.log            # Training log
├── publication_report/        # Publication-ready outputs
└── [Python scripts...]
```

---

## Dataset Deployment Requirements

### MIMIC-III Data Location

**Expected paths (update in config.json as needed):**

1. **Waveform Data:**
   ```
   C:/MIMIC-III/waveform_data/
   ├── p01/
   ├── p02/
   ├── ...
   └── p09/
   ```

2. **Clinical Data:**
   ```
   C:/MIMIC-III/clinical_data/mimiciii/1.4/
   ├── PATIENTS.csv.gz
   ├── DIAGNOSES_ICD.csv.gz
   ├── PRESCRIPTIONS.csv.gz
   └── [other clinical files...]
   ```

### Data Requirements
- **Total Size:** ~60GB (waveform) + ~7GB (clinical)
- **Format:** WFDB format for waveforms, CSV.gz for clinical
- **Access:** Requires PhysioNet credentials and CITI training
- **Data Use Agreement:** Must be signed before download

---

## Running the Pipeline

### 1. Activate Virtual Environment
```bash
cd C:\Users\ANT-PC\Documents\PROJECT-AARYAN
. venv/Scripts/activate
cd CAPSTONE-100-ALTERNATE
```

### 2. Build Dataset (First Time)
```bash
python -X utf8 production_pipeline.py --config production_medium/config.json --build-dataset
```

**Note:** Use `-X utf8` flag for proper emoji rendering in console output.

### 3. Train Model
```bash
python -X utf8 production_pipeline.py --config production_medium/config.json --train-only
```

### 4. Pattern Discovery
```bash
python -X utf8 simple_clustering.py
```

### 5. Stroke Prediction
```bash
python -X utf8 stroke_prediction.py
```

### 6. Generate Publication Report
```bash
python -X utf8 publication_report.py
```

---

## Important Notes

### Windows-Specific Considerations

1. **UTF-8 Encoding:**
   - Use `python -X utf8 script.py` to avoid emoji encoding errors
   - Or set environment variable: `set PYTHONIOENCODING=utf-8`

2. **Path Separators:**
   - Python accepts forward slashes `/` on Windows
   - All paths in config use forward slashes for cross-platform compatibility

3. **File Permissions:**
   - Ensure write permissions for output directories
   - May need to run as administrator for certain operations

4. **Shell Scripts:**
   - `.sh` scripts (run_production.sh, etc.) won't work directly
   - Use Python commands directly or create PowerShell equivalents

### GPU Optimization

**Current Settings (Conservative):**
- Batch size: 96
- Num workers: 8

**Maximum Tested:**
- Batch size: 256 (only uses 0.78GB VRAM)
- Plenty of headroom for experimentation

**To increase batch size:**
Edit `production_medium/config.json`:
```json
"batch_size": 128,  // or 256
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size in config.json

### Issue: "Module not found"
**Solution:** Ensure venv is activated and all packages installed:
```bash
pip install -r requirements.txt
```

### Issue: "UnicodeEncodeError"
**Solution:** Use `python -X utf8` flag when running scripts

### Issue: "Cannot find MIMIC data"
**Solution:** Update paths in `production_medium/config.json` to match your data location

### Issue: HDF5 errors
**Solution:** Verify h5py is installed: `pip install h5py`

---

## Next Steps

### Immediate (When Dataset Arrives)
1. Place MIMIC-III data in configured directories
2. Update `production_medium/config.json` with correct paths if different
3. Run dataset build: `python -X utf8 production_pipeline.py --config production_medium/config.json --build-dataset`
4. Monitor build.log for progress

### After Successful Build
1. Start training: `python -X utf8 production_pipeline.py --config production_medium/config.json --train-only`
2. Monitor train.log for progress
3. Expected training time: ~18 minutes (25 epochs)

### After Training
1. Run pattern discovery analysis
2. Run stroke prediction pipeline
3. Generate publication report
4. Analyze results

---

## Configuration Optimization Recommendations

For **RTX 4080** (based on performance tests):

### Aggressive Settings (Maximum Performance)
```json
{
    "batch_size": 256,
    "num_workers": 8,
    "embedding_dim": 512,
    "num_epochs": 50
}
```

### Balanced Settings (Current - Recommended)
```json
{
    "batch_size": 96,
    "num_workers": 8,
    "embedding_dim": 256,
    "num_epochs": 25
}
```

### Conservative Settings (Safest)
```json
{
    "batch_size": 64,
    "num_workers": 4,
    "embedding_dim": 128,
    "num_epochs": 25
}
```

---

## Performance Comparison

| Metric | MX570 A (2GB) | RTX 4080 (16GB) | Improvement |
|--------|---------------|-----------------|-------------|
| Batch Size | 16-48 | 256 | 5.3x |
| Throughput | ~18 samples/sec | 1,705 samples/sec | 94.7x |
| Training Time | 25 hours | 0.3 hours | 83.3x |
| VRAM Available | 2GB | 16GB | 8x |

---

## Backup and Recovery

**Backed Up Files:**
- `production_medium/config.json.backup` - Original Linux configuration

**To Restore Original Configuration:**
```bash
cd CAPSTONE-100-ALTERNATE
cp production_medium/config.json.backup production_medium/config.json
```

---

## Contact and Support

For issues or questions:
1. Check CONTEXT_REFERENCE.md for technical details
2. Review FINAL_SUMMARY.md for project overview
3. Consult RTX_4080_OPTIMIZATION.md for GPU tuning

---

## Summary of Changes

✅ Virtual environment activated
✅ All dependencies installed (17 packages)
✅ GPU tested and optimized (RTX 4080)
✅ All Linux paths converted to Windows paths
✅ Configuration optimized for RTX 4080
✅ Output directories created
✅ HDF5 operations verified
✅ Configuration backed up
✅ Documentation created

**Status:** Ready for dataset deployment and training!
