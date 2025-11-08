# ECG/PPG Discovery System - Medium Level Testing

## 📁 Project Structure (Cleaned)

### Core Pipeline Files
- `production_pipeline.py` - Main scalable pipeline for medium/full dataset training
- `run_production.sh` - Production deployment script
- `contrastive_model.py` - Self-supervised learning model architecture
- `data_preprocessing.py` - Signal processing pipeline

### Pattern Discovery & Analysis
- `simple_clustering.py` - Pattern discovery implementation
- `clinical_validation.py` - Clinical pattern validation
- `stroke_prediction.py` - ML-based stroke risk prediction
- `publication_report.py` - Publication-ready reports

### Data & Configuration
- `clinical_features.csv` - Patient clinical data
- `requirements.txt` - Python dependencies
- `production_medium/` - Current medium-level training results

### Documentation
- `FINAL_SUMMARY.md` - Complete project summary
- `README_production.md` - Production pipeline documentation

## 🚀 Current Status

**Medium-level training in progress:**
- Dataset: 100 patients, 3,294 segments
- Training: Epoch 1/25 currently running
- GPU: NVIDIA GeForce MX570 A (2GB)

## 🎯 Next Steps After Training Completes

1. **Test stroke prediction pipeline properly**
2. **Train dedicated stroke risk prediction model**
3. **Validate against clinical baselines (CHA₂DS₂-VASc)**

## 📊 Usage

```bash
# Medium-level training (100 patients)
./run_production.sh medium

# Full-scale training (2,415 patients, 60GB)
./run_production.sh full
```

## 🔬 Clinical Validation Stages

1. **Patient Mapping**: Links waveform to clinical records
2. **Outcome Labeling**: ICD-9 codes define ground truth
3. **Pattern Validation**: Clusters validated against diagnoses
4. **Feature Engineering**: Clinical demographics extracted
5. **Statistical Testing**: Chi-square tests for significance