#!/usr/bin/env python3
"""
RTX 4080 Performance Test Script
Quick validation of GPU optimization
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def test_rtx4080_performance():
    """Test RTX 4080 performance with ECG/PPG-like data"""

    print("🚀 RTX 4080 Performance Test")
    print("=" * 50)

    # GPU Info
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory:.1f}GB")
    print(f"PyTorch version: {torch.__version__}")

    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    seq_len = 1250  # 10 seconds at 125Hz
    num_channels = 4

    # Simple test model similar to WaveformEncoder
    class TestEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(256, 128)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    model = TestEncoder().to(device)

    print("\n📊 Batch Size Performance Test")
    print("-" * 50)

    best_batch_size = 16
    best_throughput = 0

    for batch_size in batch_sizes:
        try:
            # Create test data
            test_data = torch.randn(batch_size * 10, num_channels, seq_len)
            test_dataset = TensorDataset(test_data)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                   pin_memory=True, num_workers=4)

            # Test forward pass timing
            model.train()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            start_time = time.time()
            total_samples = 0

            for batch_idx, (batch,) in enumerate(test_loader):
                if batch_idx >= 10:  # Test 10 batches
                    break

                batch = batch.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():  # Test mixed precision
                    output = model(batch)

                total_samples += batch.size(0)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            throughput = total_samples / elapsed
            memory_used = torch.cuda.max_memory_allocated() / 1024**3

            print(f"Batch {batch_size:3d}: {throughput:6.1f} samples/sec, "
                  f"Memory: {memory_used:.2f}GB")

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

            torch.cuda.reset_peak_memory_stats()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size:3d}: ❌ Out of memory")
                break
            else:
                raise e

    print(f"\n🏆 Optimal batch size: {best_batch_size}")
    print(f"🚀 Best throughput: {best_throughput:.1f} samples/sec")

    # Estimate training time
    total_segments = 66432  # From our actual dataset
    samples_per_epoch = total_segments
    estimated_time_per_epoch = samples_per_epoch / best_throughput

    print(f"\n⏱️  Training Time Estimates:")
    print(f"Segments: {total_segments:,}")
    print(f"Time per epoch: {estimated_time_per_epoch:.1f} seconds ({estimated_time_per_epoch/60:.1f} minutes)")
    print(f"Total training (25 epochs): {estimated_time_per_epoch * 25 / 3600:.1f} hours")

    # Compare to MX570 A
    mx570_time_per_epoch = 3600  # 1 hour estimate
    speedup = mx570_time_per_epoch / estimated_time_per_epoch
    print(f"\n📈 RTX 4080 vs MX570 A:")
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"MX570 A total time: {mx570_time_per_epoch * 25 / 3600:.1f} hours")
    print(f"RTX 4080 total time: {estimated_time_per_epoch * 25 / 3600:.1f} hours")

if __name__ == "__main__":
    test_rtx4080_performance()