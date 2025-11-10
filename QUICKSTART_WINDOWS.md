# Quick Start Guide - Windows

## One-Time Setup ✅ (ALREADY COMPLETED)

```bash
# Navigate to project
cd C:\Users\ANT-PC\Documents\PROJECT-AARYAN

# Activate virtual environment
. venv/Scripts/activate

# Navigate to CAPSTONE
cd CAPSTONE-100-ALTERNATE
```

**All dependencies installed!** ✅
**GPU tested and ready!** ✅ RTX 4080 (16GB VRAM)
**Configuration optimized!** ✅

---

## When Dataset Arrives

### Step 1: Update Dataset Paths (if needed)

Edit [production_medium/config.json](production_medium/config.json):

```json
{
    "mimic_waveform_path": "YOUR_PATH_HERE/waveform_data",
    "mimic_clinical_path": "YOUR_PATH_HERE/mimiciii/1.4",
    ...
}
```

**Current placeholder paths:**
- Waveform: `C:/MIMIC-III/waveform_data`
- Clinical: `C:/MIMIC-III/clinical_data/mimiciii/1.4`

### Step 2: Build Dataset

```bash
python -X utf8 production_pipeline.py --config production_medium/config.json --build-dataset
```

**Expected:** ~66,432 segments from 100 patients
**Time:** Varies by disk speed (~30-60 minutes)
**Output:** `production_medium/dataset.h5`

### Step 3: Train Model

```bash
python -X utf8 production_pipeline.py --config production_medium/config.json --train-only
```

**Expected time:** ~18 minutes (25 epochs)
**Output:**
- `production_medium/best_fixed_model.pth`
- `production_medium/train.log`

### Step 4: Discover Patterns

```bash
python -X utf8 simple_clustering.py
```

**Output:** `production_medium/simple_pattern_discovery/`

### Step 5: Predict Stroke Risk

```bash
python -X utf8 stroke_prediction.py
```

**Output:** Stroke risk analysis and predictions

### Step 6: Generate Report

```bash
python -X utf8 publication_report.py
```

**Output:** `publication_report/`

---

## Current Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| GPU | RTX 4080 | 16GB VRAM |
| Batch Size | 96 | Can go up to 256 |
| Num Workers | 8 | Optimized for CPU |
| Embedding Dim | 256 | Can increase to 512 |
| Epochs | 25 | ~18 min training |
| Max Patients | 100 | Medium test |

---

## GPU Performance

**RTX 4080 Benchmark Results:**
```
Optimal Batch Size: 256
Throughput: 1,705 samples/sec
Training Time: 0.3 hours (18 minutes)
VRAM Usage: 0.78GB @ batch 256
Speedup: 92.4x vs MX570 A
```

---

## Troubleshooting

### Can't find dataset?
👉 Update paths in `production_medium/config.json`

### Out of memory?
👉 Reduce `batch_size` in config.json (try 64)

### Emoji errors?
👉 Always use `python -X utf8 script.py`

### Module not found?
👉 Activate venv: `. venv/Scripts/activate`

---

## Important Files

- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Complete setup documentation
- [production_medium/config.json](production_medium/config.json) - Configuration
- [CONTEXT_REFERENCE.md](CONTEXT_REFERENCE.md) - Technical details
- [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - Project overview

---

## Status: READY! 🚀

✅ Environment configured
✅ Dependencies installed
✅ GPU optimized (RTX 4080)
✅ Paths updated for Windows
✅ Directories created
✅ Documentation complete

**Next:** Deploy dataset and run pipeline!
