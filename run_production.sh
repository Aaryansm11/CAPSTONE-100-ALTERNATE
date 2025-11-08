#!/bin/bash

# Production ECG/PPG Discovery Pipeline Runner
# Usage: ./run_production.sh [small|medium|large|full]

set -e  # Exit on any error

# Configuration
CONDA_ENV="ecgppg"
BASE_DIR="/media/jaadoo/sexy/ecg ppg"
SCRIPT_DIR="$BASE_DIR"

# Activate conda environment
echo "🔧 Activating conda environment: $CONDA_ENV"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd "$BASE_DIR"

# Parse command line arguments
SCALE=${1:-"small"}

case $SCALE in
    "small")
        echo "🧪 Running SMALL test (20 patients, ~500MB, ~20 minutes)"
        MAX_PATIENTS=20
        OUTPUT_DIR="production_small"
        BATCH_SIZE=32
        NUM_EPOCHS=10
        ;;
    "medium")
        echo "🔬 Running MEDIUM test (100 patients, ~2.5GB, ~2 hours)"
        MAX_PATIENTS=100
        OUTPUT_DIR="production_medium"
        BATCH_SIZE=48
        NUM_EPOCHS=25
        ;;
    "large")
        echo "🏭 Running LARGE test (500 patients, ~12GB, ~12 hours)"
        MAX_PATIENTS=500
        OUTPUT_DIR="production_large"
        BATCH_SIZE=64
        NUM_EPOCHS=40
        ;;
    "full")
        echo "🌐 Running FULL production (~2,415 patients, ~60GB, ~2-3 days)"
        MAX_PATIENTS=""  # Process all
        OUTPUT_DIR="production_full"
        BATCH_SIZE=64
        NUM_EPOCHS=50
        ;;
    *)
        echo "❌ Invalid scale: $SCALE"
        echo "Usage: ./run_production.sh [small|medium|large|full]"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create custom config
CONFIG_FILE="$OUTPUT_DIR/config.json"
cat > "$CONFIG_FILE" << EOF
{
    "mimic_waveform_path": "/media/jaadoo/sexy/mimic_data",
    "mimic_clinical_path": "/media/jaadoo/sexy/physionet.org/files/mimiciii/1.4",
    "output_dir": "$OUTPUT_DIR",
    "max_patients": ${MAX_PATIENTS:-null},
    "max_segments_per_patient": 1000,
    "batch_size": $BATCH_SIZE,
    "learning_rate": 3e-4,
    "num_epochs": $NUM_EPOCHS,
    "embedding_dim": 256,
    "hidden_dims": [64, 128, 256, 512],
    "temperature": 0.1,
    "chunk_size": 500,
    "num_workers": 4,
    "save_frequency": 5
}
EOF

echo "📋 Configuration saved to: $CONFIG_FILE"

# Check if dataset already exists
DATASET_FILE="$OUTPUT_DIR/full_dataset.h5"
METADATA_FILE="$OUTPUT_DIR/full_dataset_metadata.pkl"

if [ -f "$DATASET_FILE" ] && [ -f "$METADATA_FILE" ]; then
    echo "✅ Dataset already exists, skipping to training..."
    BUILD_FLAG=""
else
    echo "🔨 Building dataset..."
    BUILD_FLAG="--build-dataset"
fi

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  No GPU detected, will use CPU (much slower)"
fi

# Log system info
echo "📊 System Info:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"
echo "  Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Start timestamp
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "⏰ Starting at: $START_TIME"

# Run the pipeline
echo "🚀 Launching production pipeline..."
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Expected processing time: varies by scale"

# Set up error handling
trap 'echo "❌ Pipeline failed at $(date)"; exit 1' ERR

# Execute pipeline
if [ -n "$BUILD_FLAG" ]; then
    echo "📦 Phase 1: Building dataset..."
    python production_pipeline.py $BUILD_FLAG --config "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_DIR/build.log"

    echo "✅ Dataset built successfully!"
    echo "📊 Dataset info:"
    if [ -f "$DATASET_FILE" ]; then
        ls -lh "$DATASET_FILE"
        python -c "
import h5py
with h5py.File('$DATASET_FILE', 'r') as f:
    print(f'Total segments: {f.attrs[\"total_segments\"]:,}')
    print(f'Total patients: {f.attrs[\"total_patients\"]:,}')
    print(f'Segment shape: {f[\"segments\"].shape}')
"
    fi
fi

echo "🧠 Phase 2: Training model..."
python production_pipeline.py --train-only --config "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_DIR/train.log"

# End timestamp
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "✅ Pipeline completed successfully!"
echo "⏰ Started: $START_TIME"
echo "⏰ Finished: $END_TIME"

# Show results
echo ""
echo "📁 Results in: $OUTPUT_DIR/"
echo "🔍 Key files:"
echo "  - Dataset: $DATASET_FILE"
echo "  - Model checkpoints: $OUTPUT_DIR/checkpoint_*.pth"
echo "  - Training logs: $OUTPUT_DIR/*.log"
echo "  - Configuration: $CONFIG_FILE"

echo ""
echo "🎯 Next steps:"
echo "  1. Check training logs: tail $OUTPUT_DIR/train.log"
echo "  2. Analyze embeddings for pattern discovery"
echo "  3. Run clinical validation"
echo "  4. Build stroke risk prediction model"

echo ""
echo "📊 To monitor GPU usage during training:"
echo "  watch -n 1 nvidia-smi"

echo ""
echo "🔄 To resume training if interrupted:"
echo "  python production_pipeline.py --train-only --config $CONFIG_FILE"