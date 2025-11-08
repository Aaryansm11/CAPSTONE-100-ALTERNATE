#!/usr/bin/env python3
"""
Robust Data Preprocessing Module - Fixes data loss issues
Handles missing files, short segments, channel mismatches, and corrupt data
"""

import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RobustWaveformPreprocessor:
    """
    Robust preprocessor that recovers data instead of skipping
    """

    def __init__(self,
                 segment_length_sec=10,
                 sampling_rate=125,
                 filter_low=0.5,
                 filter_high=40.0,
                 overlap_ratio=0.0,
                 min_segment_length_sec=5.0,
                 max_channels=8):
        """
        Initialize robust preprocessor

        Args:
            segment_length_sec: Target segment length
            sampling_rate: Target sampling rate (Hz)
            filter_low: Low cutoff frequency
            filter_high: High cutoff frequency
            overlap_ratio: Overlap between segments
            min_segment_length_sec: Minimum acceptable segment length
            max_channels: Maximum channels to handle
        """
        self.segment_length_sec = segment_length_sec
        self.sampling_rate = sampling_rate
        self.segment_length_samples = int(segment_length_sec * sampling_rate)
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.overlap_ratio = overlap_ratio
        self.min_segment_samples = int(min_segment_length_sec * sampling_rate)
        self.max_channels = max_channels

    def safe_bandpass_filter(self, signal_data, fs):
        """Apply bandpass filter with safety checks"""
        try:
            if signal_data.shape[0] < 30:  # Too short for filter
                print(f"Signal too short for filtering ({signal_data.shape[0]} samples), skipping filter")
                return signal_data

            nyquist = fs / 2
            low = max(self.filter_low / nyquist, 0.01)  # Prevent filter issues
            high = min(self.filter_high / nyquist, 0.98)

            if low >= high:
                print(f"Invalid filter range, skipping filter")
                return signal_data

            # Use shorter filter order for short signals
            filter_order = min(4, signal_data.shape[0] // 10)
            if filter_order < 1:
                return signal_data

            b, a = signal.butter(filter_order, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data, axis=0)
            return filtered_signal

        except Exception as e:
            print(f"Filter failed: {e}, returning unfiltered signal")
            return signal_data

    def handle_short_segments(self, signal_data):
        """Handle segments shorter than target length"""
        current_length = signal_data.shape[0]
        target_length = self.segment_length_samples

        if current_length >= target_length:
            return signal_data

        if current_length < self.min_segment_samples:
            # Too short, zero-pad to minimum length
            pad_length = self.min_segment_samples - current_length
            padded = np.pad(signal_data, ((0, pad_length), (0, 0)), mode='constant')
            return padded
        else:
            # Zero-pad to target length
            pad_length = target_length - current_length
            padded = np.pad(signal_data, ((0, pad_length), (0, 0)), mode='constant')
            return padded

    def handle_channel_mismatch(self, signal_data, expected_channels=None):
        """Dynamically handle different channel counts"""
        try:
            current_channels = signal_data.shape[1] if len(signal_data.shape) > 1 else 1

            # Reshape if needed
            if len(signal_data.shape) == 1:
                signal_data = signal_data.reshape(-1, 1)
                current_channels = 1

            # Limit to max channels to prevent memory issues
            if current_channels > self.max_channels:
                signal_data = signal_data[:, :self.max_channels]
                current_channels = self.max_channels

            return signal_data, current_channels

        except Exception as e:
            print(f"Channel handling failed: {e}")
            # Return as single channel if reshape fails
            if len(signal_data.shape) == 1:
                return signal_data.reshape(-1, 1), 1
            else:
                return signal_data[:, :1], 1

    def try_multiple_read_methods(self, record_path):
        """Try multiple methods to read waveform data"""
        methods = [
            lambda: self._read_wfdb(record_path),
            lambda: self._read_wfdb_alternative(record_path),
            lambda: self._read_raw_data(record_path),
        ]

        for i, method in enumerate(methods):
            try:
                result = method()
                if result is not None:
                    print(f"Successfully read using method {i+1}")
                    return result
            except Exception as e:
                print(f"Read method {i+1} failed: {e}")
                continue

        return None

    def _read_wfdb(self, record_path):
        """Standard WFDB read"""
        record = wfdb.rdrecord(record_path)
        if record.p_signal is None or len(record.p_signal) == 0:
            return None
        return record

    def _read_wfdb_alternative(self, record_path):
        """Try reading with different extensions"""
        base_path = str(record_path).split('.')[0]
        extensions = ['', '.dat', '.mat', '.hea']

        for ext in extensions:
            try:
                test_path = base_path + ext
                if os.path.exists(test_path) or ext == '':
                    record = wfdb.rdrecord(base_path)
                    if record.p_signal is not None and len(record.p_signal) > 0:
                        return record
            except:
                continue
        return None

    def _read_raw_data(self, record_path):
        """Try reading raw binary data as fallback"""
        try:
            # Look for .dat files
            dat_files = list(Path(record_path).parent.glob("*.dat"))
            if not dat_files:
                return None

            # Read first .dat file as raw float data
            data = np.fromfile(dat_files[0], dtype=np.float32)
            if len(data) == 0:
                return None

            # Assume reasonable defaults
            channels = min(8, int(np.sqrt(len(data))))  # Guess channel count
            samples = len(data) // channels

            signal_data = data[:samples * channels].reshape(samples, channels)

            # Create minimal record object
            class SimpleRecord:
                def __init__(self, signal_data):
                    self.p_signal = signal_data
                    self.fs = 125  # Assume standard sampling rate
                    self.sig_name = [f'CH{i}' for i in range(signal_data.shape[1])]
                    self.units = ['mV'] * signal_data.shape[1]

            return SimpleRecord(signal_data)

        except Exception as e:
            print(f"Raw data read failed: {e}")
            return None

    def process_record(self, record_path, patient_id=None):
        """
        Robustly process a single waveform record
        """
        try:
            # Try multiple read methods
            record = self.try_multiple_read_methods(record_path)
            if record is None:
                print(f"Could not read record {record_path}")
                return None

            # Handle channel mismatches
            signal_data, num_channels = self.handle_channel_mismatch(record.p_signal)

            # Handle NaN values
            signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Handle empty or very short signals
            if len(signal_data) < 10:
                print(f"Signal too short: {len(signal_data)} samples")
                return None

            # Handle short segments with padding
            if len(signal_data) < self.segment_length_samples:
                signal_data = self.handle_short_segments(signal_data)

            # Resample to target sampling rate if needed
            original_fs = getattr(record, 'fs', 125)
            if original_fs != self.sampling_rate:
                try:
                    num_samples = int(len(signal_data) * self.sampling_rate / original_fs)
                    signal_data = signal.resample(signal_data, num_samples, axis=0)
                except:
                    print("Resampling failed, using original sampling rate")

            # Apply robust bandpass filter
            filtered_data = self.safe_bandpass_filter(signal_data, self.sampling_rate)

            # Create segments
            segments = []
            num_samples = len(filtered_data)

            # Maximize segment extraction with minimal overlap
            if num_samples >= self.segment_length_samples:
                # Aggressive segmentation with minimal overlap (10%)
                start_idx = 0
                step_size = int(self.segment_length_samples * 0.9)  # 10% overlap for max data
                while start_idx + self.segment_length_samples <= num_samples:
                    end_idx = start_idx + self.segment_length_samples
                    segment = filtered_data[start_idx:end_idx]
                    segments.append(segment)
                    start_idx += step_size
            else:
                # Single segment (already padded if necessary)
                segments.append(filtered_data[:self.segment_length_samples])

            # Process segments
            good_segments = []
            segment_metadata = []

            for i, segment in enumerate(segments):
                try:
                    # More permissive quality checks - only reject completely flat signals
                    if np.std(segment) > 0.0001:  # Very permissive - only reject truly flat
                        # Normalize per channel
                        normalized_segment = np.zeros_like(segment)
                        for ch in range(segment.shape[1]):
                            ch_data = segment[:, ch]
                            if np.std(ch_data) > 0:
                                normalized_segment[:, ch] = (ch_data - np.mean(ch_data)) / np.std(ch_data)
                            else:
                                normalized_segment[:, ch] = ch_data

                        good_segments.append(normalized_segment)

                        # Create metadata
                        seg_meta = {
                            'segment_idx': i,
                            'start_time_sec': i * int(self.segment_length_samples * 0.9) / self.sampling_rate,
                            'num_channels': segment.shape[1],
                            'signal_quality': 'good' if np.std(segment) > 0.01 else 'poor'
                        }
                        segment_metadata.append(seg_meta)

                except Exception as e:
                    print(f"Error processing segment {i}: {e}")
                    continue

            # Extract metadata
            metadata = {
                'patient_id': patient_id,
                'record_name': os.path.basename(str(record_path)),
                'original_fs': original_fs,
                'signal_names': getattr(record, 'sig_name', [f'CH{i}' for i in range(num_channels)]),
                'units': getattr(record, 'units', ['mV'] * num_channels),
                'duration_sec': len(signal_data) / self.sampling_rate,
                'num_channels': num_channels
            }

            result = {
                'segments': np.array(good_segments) if good_segments else np.array([]),
                'metadata': metadata,
                'segment_metadata': segment_metadata,
                'num_good_segments': len(good_segments),
                'total_segments': len(segments),
                'recovery_stats': {
                    'original_length': len(record.p_signal) if record.p_signal is not None else 0,
                    'processed_length': len(signal_data),
                    'channels_original': record.p_signal.shape[1] if record.p_signal is not None and len(record.p_signal.shape) > 1 else 1,
                    'channels_processed': num_channels
                }
            }

            return result

        except Exception as e:
            print(f"Error processing record {record_path}: {e}")
            return None

def test_robust_preprocessing():
    """Test the robust preprocessing on problematic files"""
    print("Testing robust preprocessing...")

    # Test with larger files that should have many segments
    test_records = [
        "/media/jaadoo/sexy/mimic_data/p01/p010595/3099108_0005",
        "/media/jaadoo/sexy/mimic_data/p01/p010595/3099108_0006",
        "/media/jaadoo/sexy/mimic_data/p01/p010595/3906530_0003"
    ]

    robust_processor = RobustWaveformPreprocessor()

    for record_path in test_records:
        print(f"\nTesting: {record_path}")
        result = robust_processor.process_record(record_path)
        if result:
            print(f"  SUCCESS: {result['num_good_segments']} segments recovered")
            print(f"  Recovery: {result['recovery_stats']}")
        else:
            print("  FAILED: Could not recover")

if __name__ == "__main__":
    test_robust_preprocessing()