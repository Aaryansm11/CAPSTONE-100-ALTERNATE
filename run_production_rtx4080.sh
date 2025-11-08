#!/bin/bash
# RTX 4080 Optimized Production Pipeline Runner
# ECG/PPG Discovery System - GPU Optimized Version

set -e

# RTX 4080 Optimizations
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4080 architecture
export CUDA_LAUNCH_BLOCKING=0      # Enable async operations
export TORCH_BACKENDS_CUDNN_BENCHMARK=1  # Enable cuDNN optimizations

SCALE=${1:-medium}
BASE_DIR="/media/jaadoo/sexy/ecg ppg"
MIMIC_DIR="/media/jaadoo/sexy/mimic_data"

echo "🚀 RTX 4080 Optimized ECG/PPG Discovery Pipeline"
echo "Scale: $SCALE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"

cd "$BASE_DIR"

# Verify RTX 4080 setup
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"

if [ "$SCALE" = "medium" ]; then
    echo "🎯 Medium-scale training (100 patients, RTX 4080 optimized)"
    OUTPUT_DIR="production_rtx4080_medium"
    PATIENT_LIMIT=100
    BATCH_SIZE=128        # 8x larger than MX570 A
    NUM_WORKERS=8         # More CPU workers
    EPOCHS=25

elif [ "$SCALE" = "full" ]; then
    echo "🚀 Full-scale training (~2,415 patients, RTX 4080 optimized)"
    OUTPUT_DIR="production_rtx4080_full"
    PATIENT_LIMIT=10000   # Essentially unlimited
    BATCH_SIZE=96         # Slightly smaller for massive dataset
    NUM_WORKERS=12        # Maximum CPU workers
    EPOCHS=50

else
    echo "❌ Usage: $0 [medium|full]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Starting RTX 4080 optimized training..."
echo "  Batch Size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo "  Epochs: $EPOCHS"
echo "  Output: $OUTPUT_DIR"

# Run optimized pipeline with RTX 4080 parameters
python3 production_pipeline.py \
    --mimic_dir "$MIMIC_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --patient_limit $PATIENT_LIMIT \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --pin_memory \
    --persistent_workers \
    --mixed_precision \
    --prefetch_factor 4 \
    --embedding_dim 256 \
    2>&1 | tee "$OUTPUT_DIR/training_log_rtx4080.txt"

echo "✅ RTX 4080 training completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📊 Training log: $OUTPUT_DIR/training_log_rtx4080.txt"

# Display final results
if [ -f "$OUTPUT_DIR/training_results.txt" ]; then
    echo "📈 Final Results:"
    tail -20 "$OUTPUT_DIR/training_results.txt"
fi