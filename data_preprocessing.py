#!/usr/bin/env python3
"""
Data Preprocessing Module for MIMIC-III Waveform Data
Handles ECG/PPG segmentation, filtering, and preparation for self-supervised learning
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

class WaveformPreprocessor:
    """
    Preprocesses MIMIC-III waveform data for arrhythmia discovery
    """

    def __init__(self,
                 segment_length_sec=10,
                 sampling_rate=125,
                 filter_low=0.5,
                 filter_high=40.0,
                 overlap_ratio=0.0):
        """
        Initialize preprocessor

        Args:
            segment_length_sec: Length of each segment in seconds
            sampling_rate: Target sampling rate (Hz)
            filter_low: Low cutoff frequency for bandpass filter
            filter_high: High cutoff frequency for bandpass filter
            overlap_ratio: Overlap between segments (0.0 = no overlap)
        """
        self.segment_length_sec = segment_length_sec
        self.sampling_rate = sampling_rate
        self.segment_length_samples = int(segment_length_sec * sampling_rate)
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.overlap_ratio = overlap_ratio
        self.overlap_samples = int(self.segment_length_samples * overlap_ratio)
        self.step_size = self.segment_length_samples - self.overlap_samples

    def bandpass_filter(self, signal_data, fs):
        """Apply bandpass filter to remove noise"""
        nyquist = fs / 2
        low = self.filter_low / nyquist
        high = min(self.filter_high / nyquist, 0.99)

        # Design butterworth filter
        b, a = signal.butter(4, [low, high], btype='band')

        # Apply filter
        filtered_signal = signal.filtfilt(b, a, signal_data, axis=0)
        return filtered_signal

    def detect_flatline(self, signal_data, threshold=0.001):
        """Detect flatline segments (poor signal quality)"""
        std = np.std(signal_data, axis=0)
        return np.any(std < threshold)

    def detect_artifacts(self, signal_data, z_threshold=5.0):
        """Detect obvious artifacts using z-score"""
        z_scores = np.abs((signal_data - np.mean(signal_data, axis=0)) / np.std(signal_data, axis=0))
        return np.any(z_scores > z_threshold)

    def resample_signal(self, signal_data, original_fs):
        """Resample signal to target sampling rate"""
        if original_fs == self.sampling_rate:
            return signal_data

        # Calculate resampling ratio
        num_samples = int(len(signal_data) * self.sampling_rate / original_fs)
        resampled = signal.resample(signal_data, num_samples, axis=0)
        return resampled

    def normalize_signal(self, signal_data):
        """Normalize signal per channel"""
        scaler = StandardScaler()
        normalized = scaler.fit_transform(signal_data)
        return normalized, scaler

    def segment_signal(self, signal_data):
        """Split signal into fixed-length segments"""
        segments = []
        num_samples = len(signal_data)

        start_idx = 0
        while start_idx + self.segment_length_samples <= num_samples:
            end_idx = start_idx + self.segment_length_samples
            segment = signal_data[start_idx:end_idx]
            segments.append(segment)
            start_idx += self.step_size

        return np.array(segments)

    def process_record(self, record_path, patient_id=None):
        """
        Process a single waveform record

        Args:
            record_path: Path to waveform record (without extension)
            patient_id: Patient ID for metadata

        Returns:
            Dict with processed segments and metadata
        """
        try:
            # Read waveform record
            record = wfdb.rdrecord(record_path)

            # Extract metadata
            metadata = {
                'patient_id': patient_id,
                'record_name': os.path.basename(record_path),
                'original_fs': record.fs,
                'signal_names': record.sig_name,
                'units': record.units,
                'duration_sec': len(record.p_signal) / record.fs
            }

            # Check if we have the signals we want (ECG, PPG)
            ecg_channels = []
            ppg_channels = []

            for i, sig_name in enumerate(record.sig_name):
                sig_name_lower = sig_name.lower()
                if any(ecg in sig_name_lower for ecg in ['ii', 'i', 'v', 'ecg', 'avr', 'mcl']):
                    ecg_channels.append(i)
                elif any(ppg in sig_name_lower for ppg in ['ppg', 'pleth', 'pulse']):
                    ppg_channels.append(i)

            # Resample to target sampling rate
            signal_data = self.resample_signal(record.p_signal, record.fs)

            # Apply bandpass filter
            filtered_data = self.bandpass_filter(signal_data, self.sampling_rate)

            # Segment the signal
            segments = self.segment_signal(filtered_data)

            # Quality control - remove bad segments
            good_segments = []
            segment_metadata = []

            for i, segment in enumerate(segments):
                # Check for flatline and artifacts
                if not self.detect_flatline(segment) and not self.detect_artifacts(segment):
                    # Normalize segment
                    normalized_segment, scaler = self.normalize_signal(segment)
                    good_segments.append(normalized_segment)

                    # Store segment metadata
                    seg_meta = {
                        'segment_idx': i,
                        'start_time_sec': i * self.step_size / self.sampling_rate,
                        'ecg_channels': [idx for idx in ecg_channels if idx < segment.shape[1]],
                        'ppg_channels': [idx for idx in ppg_channels if idx < segment.shape[1]],
                        'has_ecg': len([idx for idx in ecg_channels if idx < segment.shape[1]]) > 0,
                        'has_ppg': len([idx for idx in ppg_channels if idx < segment.shape[1]]) > 0
                    }
                    segment_metadata.append(seg_meta)

            result = {
                'segments': np.array(good_segments) if good_segments else np.array([]),
                'metadata': metadata,
                'segment_metadata': segment_metadata,
                'num_good_segments': len(good_segments),
                'total_segments': len(segments)
            }

            return result

        except Exception as e:
            print(f"Error processing record {record_path}: {e}")
            return None

class DatasetBuilder:
    """
    Builds dataset from multiple patients for training
    """

    def __init__(self,
                 mimic_waveform_path,
                 mimic_clinical_path,
                 preprocessor=None):
        """
        Initialize dataset builder

        Args:
            mimic_waveform_path: Path to MIMIC waveform data
            mimic_clinical_path: Path to MIMIC clinical data
            preprocessor: WaveformPreprocessor instance
        """
        self.mimic_waveform_path = Path(mimic_waveform_path)
        self.mimic_clinical_path = Path(mimic_clinical_path)
        self.preprocessor = preprocessor or WaveformPreprocessor()

    def load_clinical_data(self):
        """Load relevant clinical tables"""
        print("Loading clinical data...")

        # Load PATIENTS table
        patients_path = self.mimic_clinical_path / "PATIENTS.csv.gz"
        self.patients_df = pd.read_csv(patients_path, compression='gzip')

        # Load DIAGNOSES_ICD table
        diagnoses_path = self.mimic_clinical_path / "DIAGNOSES_ICD.csv.gz"
        self.diagnoses_df = pd.read_csv(diagnoses_path, compression='gzip')

        # Load PRESCRIPTIONS table
        prescriptions_path = self.mimic_clinical_path / "PRESCRIPTIONS.csv.gz"
        self.prescriptions_df = pd.read_csv(prescriptions_path, compression='gzip')

        print(f"Loaded {len(self.patients_df)} patients, {len(self.diagnoses_df)} diagnoses, {len(self.prescriptions_df)} prescriptions")

    def find_patient_records(self, max_patients=None):
        """Find all patient waveform records"""
        patient_records = []

        # Search through patient directories
        for p_dir in sorted(self.mimic_waveform_path.glob("p*")):
            if not p_dir.is_dir():
                continue

            for patient_dir in sorted(p_dir.glob("p*")):
                if not patient_dir.is_dir():
                    continue

                patient_id = patient_dir.name

                # Find .dat files (waveform records) - exclude numerics files ending with 'n'
                dat_files = list(patient_dir.glob("*.dat"))
                if dat_files:
                    for dat_file in dat_files:
                        # Skip numerics files (ending with 'n')
                        if dat_file.stem.endswith('n'):
                            continue

                        # Remove .dat extension to get record name
                        record_name = dat_file.stem
                        record_path = patient_dir / record_name

                        # Check if corresponding header file exists
                        hea_file = patient_dir / f"{record_name}.hea"
                        if hea_file.exists():
                            patient_records.append({
                                'patient_id': patient_id,
                                'record_path': str(record_path),
                                'record_name': record_name
                            })

                if max_patients and len(set(r['patient_id'] for r in patient_records)) >= max_patients:
                    break

            if max_patients and len(set(r['patient_id'] for r in patient_records)) >= max_patients:
                break

        print(f"Found {len(patient_records)} records from {len(set(r['patient_id'] for r in patient_records))} patients")
        return patient_records

    def build_dataset(self, max_patients=50, save_path=None):
        """
        Build preprocessed dataset

        Args:
            max_patients: Maximum number of patients to process
            save_path: Path to save processed dataset

        Returns:
            Dict containing processed segments and metadata
        """
        print(f"Building dataset with max {max_patients} patients...")

        # Load clinical data
        self.load_clinical_data()

        # Find patient records
        patient_records = self.find_patient_records(max_patients)

        all_segments = []
        all_metadata = []
        patient_summary = []

        processed_count = 0

        for record_info in patient_records[:100]:  # Limit for initial experiment
            print(f"Processing {record_info['patient_id']}/{record_info['record_name']}...")

            result = self.preprocessor.process_record(
                record_info['record_path'],
                record_info['patient_id']
            )

            if result and result['num_good_segments'] > 0:
                all_segments.append(result['segments'])

                # Add record info to metadata
                for seg_meta in result['segment_metadata']:
                    seg_meta.update({
                        'patient_id': record_info['patient_id'],
                        'record_name': record_info['record_name']
                    })

                all_metadata.extend(result['segment_metadata'])

                patient_summary.append({
                    'patient_id': record_info['patient_id'],
                    'record_name': record_info['record_name'],
                    'num_segments': result['num_good_segments'],
                    'duration_sec': result['metadata']['duration_sec'],
                    'signal_names': result['metadata']['signal_names']
                })

                processed_count += 1

            if processed_count % 10 == 0:
                print(f"Processed {processed_count} records...")

        # Combine all segments - handle different channel numbers
        if all_segments:
            # Find max number of channels
            max_channels = max(seg.shape[2] for seg in all_segments)

            # Pad segments to have same number of channels
            padded_segments = []
            for seg in all_segments:
                if seg.shape[2] < max_channels:
                    # Pad with zeros for missing channels
                    pad_width = ((0, 0), (0, 0), (0, max_channels - seg.shape[2]))
                    padded_seg = np.pad(seg, pad_width, mode='constant', constant_values=0)
                    padded_segments.append(padded_seg)
                else:
                    padded_segments.append(seg)

            combined_segments = np.concatenate(padded_segments, axis=0)
        else:
            combined_segments = np.array([])

        dataset = {
            'segments': combined_segments,
            'segment_metadata': all_metadata,
            'patient_summary': patient_summary,
            'preprocessing_params': {
                'segment_length_sec': self.preprocessor.segment_length_sec,
                'sampling_rate': self.preprocessor.sampling_rate,
                'filter_low': self.preprocessor.filter_low,
                'filter_high': self.preprocessor.filter_high
            }
        }

        print(f"Dataset built: {len(combined_segments)} segments from {processed_count} records")

        if save_path:
            np.savez_compressed(save_path, **dataset)
            print(f"Dataset saved to {save_path}")

        return dataset

def main():
    """Main function to test preprocessing"""
    # Paths
    mimic_waveform_path = "/media/jaadoo/sexy/mimic_data"
    mimic_clinical_path = "/media/jaadoo/sexy/physionet.org/files/mimiciii/1.4"

    # Initialize preprocessor
    preprocessor = WaveformPreprocessor(
        segment_length_sec=10,
        sampling_rate=125,
        filter_low=0.5,
        filter_high=40.0,
        overlap_ratio=0.0
    )

    # Initialize dataset builder
    dataset_builder = DatasetBuilder(
        mimic_waveform_path=mimic_waveform_path,
        mimic_clinical_path=mimic_clinical_path,
        preprocessor=preprocessor
    )

    # Build small dataset for testing
    dataset = dataset_builder.build_dataset(
        max_patients=10,
        save_path="/media/jaadoo/sexy/ecg ppg/test_dataset.npz"
    )

    print(f"Final dataset shape: {dataset['segments'].shape}")
    print(f"Number of segments with ECG: {sum(1 for m in dataset['segment_metadata'] if m['has_ecg'])}")
    print(f"Number of segments with PPG: {sum(1 for m in dataset['segment_metadata'] if m['has_ppg'])}")

if __name__ == "__main__":
    main()