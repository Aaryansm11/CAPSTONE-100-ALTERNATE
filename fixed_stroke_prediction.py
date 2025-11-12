#!/usr/bin/env python3
"""
FIXED Stroke Risk Prediction Pipeline
- Auto-detects model architecture from checkpoint
- Uses HDF5 dataset format
- Works with corrected model from production_medium_corrected
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
import pickle
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from contrastive_model import WaveformEncoder

class AutoDetectStrokeRiskExtractor:
    """Extract features with auto-detected model architecture"""

    def __init__(self, checkpoint_path, config_path, clustering_results_path=None, device='cuda'):
        self.device = device
        self.model = None
        self.config = None
        self.clustering_results = None
        self.load_model(checkpoint_path, config_path)
        if clustering_results_path:
            self.load_clustering_results(clustering_results_path)

    def load_model(self, checkpoint_path, config_path):
        """Load model with auto-detected parameters"""
        print(f"\n Auto-detecting model architecture...")

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Auto-detect input channels
        state_dict = checkpoint['model_state_dict']
        input_channels = state_dict['input_conv.weight'].shape[1]

        print(f"   Detected input channels: {input_channels}")
        print(f"   Embedding dim: {self.config['embedding_dim']}")
        print(f"   Hidden dims: {self.config['hidden_dims']}")

        # Create model
        self.model = WaveformEncoder(
            input_channels=input_channels,
            hidden_dims=self.config['hidden_dims'],
            embedding_dim=self.config['embedding_dim'],
            dropout=0.1
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"   Model loaded successfully")

    def load_clustering_results(self, results_path):
        """Load clustering results"""
        try:
            with open(results_path, 'r') as f:
                self.clustering_results = json.load(f)
            print("   Clustering results loaded")
        except:
            print("   Could not load clustering results")
            self.clustering_results = None

    def extract_embeddings(self, segments, batch_size=64):
        """Extract embeddings from segments"""
        print(f"\n Extracting embeddings...")

        # Convert to tensor
        if isinstance(segments, np.ndarray):
            segments_tensor = torch.FloatTensor(segments)
        else:
            segments_tensor = segments

        # Transpose if needed: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if segments_tensor.shape[1] > segments_tensor.shape[2]:
            segments_tensor = segments_tensor.transpose(1, 2)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(segments_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                segments_batch = batch[0].to(self.device)
                embeddings = self.model(segments_batch)
                embeddings = F.normalize(embeddings, dim=1)
                all_embeddings.append(embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        print(f"   Generated {len(embeddings)} embeddings")

        return embeddings

    def extract_waveform_features(self, segments, max_segments=5000):
        """Extract traditional waveform features (limited for speed)"""
        print(f"\n Extracting waveform features...")

        # Sample if too many segments
        if len(segments) > max_segments:
            indices = np.random.choice(len(segments), max_segments, replace=False)
            segments = segments[indices]
            print(f"   Sampled {max_segments} segments for feature extraction")

        features = []

        for i, segment in enumerate(segments):
            if i % 1000 == 0:
                print(f"   Processing segment {i}/{len(segments)}...")

            segment_features = {}

            # Per-channel statistics
            for ch in range(min(8, segment.shape[1])):  # Limit to 8 channels
                channel_data = segment[:, ch]

                # Skip empty channels
                if np.all(channel_data == 0):
                    continue

                # Basic statistics
                segment_features[f'ch{ch}_mean'] = np.mean(channel_data)
                segment_features[f'ch{ch}_std'] = np.std(channel_data)
                segment_features[f'ch{ch}_min'] = np.min(channel_data)
                segment_features[f'ch{ch}_max'] = np.max(channel_data)
                segment_features[f'ch{ch}_range'] = np.ptp(channel_data)
                segment_features[f'ch{ch}_energy'] = np.sum(channel_data ** 2)

            features.append(segment_features)

        df = pd.DataFrame(features).fillna(0)
        print(f"   Extracted {df.shape[1]} waveform features")

        return df


class StrokeRiskPredictor:
    """Stroke risk prediction using multiple models"""

    def __init__(self, output_dir='production_medium/stroke_prediction'):
        self.output_dir = output_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.results = {}

        os.makedirs(output_dir, exist_ok=True)

    def prepare_patient_features(self, embeddings, waveform_features, metadata):
        """Aggregate features at patient level"""
        print(f"\n Preparing patient-level features...")

        # Convert embeddings to DataFrame
        embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])

        # Add metadata
        patient_ids = [m.get('patient_id') for m in metadata]
        stroke_labels = [m.get('has_stroke', 0) for m in metadata]
        arrhythmia_labels = [m.get('has_arrhythmia', 0) for m in metadata]
        ages = [m.get('age', 65) for m in metadata]
        genders = [m.get('gender', 0) for m in metadata]

        embedding_df['patient_id'] = patient_ids
        embedding_df['has_stroke'] = stroke_labels
        embedding_df['has_arrhythmia'] = arrhythmia_labels
        embedding_df['age'] = ages
        embedding_df['gender'] = genders

        # Add waveform features
        waveform_features['patient_id'] = patient_ids[:len(waveform_features)]

        # Aggregate by patient
        patient_features = embedding_df.groupby('patient_id').agg({
            **{f'emb_{i}': ['mean', 'std'] for i in range(embeddings.shape[1])},
            'has_stroke': 'max',
            'has_arrhythmia': 'max',
            'age': 'first',
            'gender': 'first'
        })

        # Flatten column names
        patient_features.columns = ['_'.join(map(str, col)).strip() for col in patient_features.columns.values]
        patient_features = patient_features.reset_index()

        # Add aggregated waveform features
        patient_waveform = waveform_features.groupby('patient_id').agg(['mean', 'std', 'min', 'max'])
        patient_waveform.columns = ['_'.join(col).strip() for col in patient_waveform.columns.values]
        patient_waveform = patient_waveform.reset_index()

        # Merge
        combined = patient_features.merge(patient_waveform, on='patient_id', how='left').fillna(0)

        print(f"   Patient features shape: {combined.shape}")
        print(f"   Unique patients: {len(combined)}")
        print(f"   Stroke patients: {combined['has_stroke_max'].sum()}")

        return combined

    def train_models(self, features_df):
        """Train multiple models for stroke prediction"""
        print(f"\n Training stroke prediction models...")

        # Prepare data
        X = features_df.drop(['patient_id', 'has_stroke_max', 'has_arrhythmia_max'],
                             axis=1, errors='ignore')
        y = features_df['has_stroke_max']

        print(f"   Features shape: {X.shape}")
        print(f"   Target distribution: Stroke={y.sum()}, No Stroke={len(y)-y.sum()}")

        if y.sum() < 5:
            print("   WARNING: Too few positive samples for reliable training")
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
        except:
            print("   Cannot stratify with so few samples, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        }

        # Train and evaluate
        for name, model in models.items():
            print(f"\n   Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Metrics
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
                ap_score = average_precision_score(y_test, y_pred_proba)
            except:
                print(f"     WARNING: Could not compute AUC/AP (too few samples)")
                auc_score = 0.0
                ap_score = 0.0

            self.models[name] = model
            self.results[name] = {
                'test_auc': auc_score,
                'test_ap': ap_score,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

            print(f"     Test AUC: {auc_score:.3f}")
            print(f"     Test AP: {ap_score:.3f}")

    def visualize_results(self):
        """Create visualizations"""
        print(f"\n Creating visualizations...")

        if not self.results:
            print("   No results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Stroke Risk Prediction Results', fontsize=16, fontweight='bold')

        # ROC Curves
        ax = axes[0, 0]
        for name, results in self.results.items():
            if results['test_auc'] > 0:
                fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
                ax.plot(fpr, tpr, label=f"{name} (AUC: {results['test_auc']:.3f})")

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Model Comparison
        ax = axes[0, 1]
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['test_auc'] for name in model_names]

        ax.bar(range(len(model_names)), auc_scores, alpha=0.7)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylabel('AUC Score')
        ax.set_title('Model Performance Comparison')
        ax.grid(True, alpha=0.3)

        # Feature Importance (Random Forest)
        if 'random_forest' in self.feature_importance:
            ax = axes[1, 0]
            top_features = self.feature_importance['random_forest'].head(15)
            ax.barh(range(len(top_features)), top_features['importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 15 Features (Random Forest)')
            ax.grid(True, alpha=0.3)

        # Confusion Matrix (best model)
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_auc'])
            ax = axes[1, 1]
            cm = confusion_matrix(self.results[best_model]['y_test'],
                                 self.results[best_model]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix ({best_model})')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'stroke_prediction_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_path}")

    def save_results(self):
        """Save results"""
        print(f"\n Saving results...")

        # Save model performance
        performance_results = {}
        for name, results in self.results.items():
            performance_results[name] = {
                'test_auc': float(results['test_auc']),
                'test_ap': float(results['test_ap'])
            }

        with open(os.path.join(self.output_dir, 'model_performance.json'), 'w') as f:
            json.dump(performance_results, f, indent=2)

        # Save feature importance
        for name, importance in self.feature_importance.items():
            importance.to_csv(
                os.path.join(self.output_dir, f'{name}_feature_importance.csv'),
                index=False
            )

        print(f"   Results saved to: {self.output_dir}")


def main():
    """Main stroke prediction pipeline"""
    print("=" * 70)
    print("STROKE RISK PREDICTION PIPELINE (FIXED)")
    print("=" * 70)

    # Paths
    data_path = 'production_medium/full_dataset.h5'
    metadata_path = 'production_medium/full_dataset_metadata.pkl'
    checkpoint_path = 'production_medium_corrected/latest_checkpoint.pth'
    config_path = 'production_medium/config.json'
    clustering_path = 'production_medium/simple_pattern_discovery/clustering_results.json'

    # Check if corrected checkpoint exists
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'production_medium/checkpoint_epoch_final.pth'
        print(f" Using old checkpoint (may have representation collapse)")

    print(f"\n Loading data:")
    print(f"   Dataset: {data_path}")
    print(f"   Checkpoint: {checkpoint_path}")

    # Load dataset
    with h5py.File(data_path, 'r') as h5f:
        # Sample segments for speed (use all for final version)
        total_segments = len(h5f['segments'])
        sample_size = min(20000, total_segments)  # Sample 20k segments
        indices = np.linspace(0, total_segments-1, sample_size, dtype=int)
        segments = h5f['segments'][indices]

    # Load metadata
    with open(metadata_path, 'rb') as f:
        all_metadata = pickle.load(f)
        metadata = [all_metadata[i] for i in indices]

    print(f"   Loaded {len(segments):,} segments (sampled from {total_segments:,})")

    # Extract features
    extractor = AutoDetectStrokeRiskExtractor(checkpoint_path, config_path, clustering_path)
    embeddings = extractor.extract_embeddings(segments)
    waveform_features = extractor.extract_waveform_features(segments)

    # Initialize predictor
    predictor = StrokeRiskPredictor()

    # Prepare patient-level features
    patient_features = predictor.prepare_patient_features(embeddings, waveform_features, metadata)

    # Train models
    predictor.train_models(patient_features)

    # Visualize and save
    predictor.visualize_results()
    predictor.save_results()

    print("\n" + "=" * 70)
    print("STROKE PREDICTION PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"   Output directory: {predictor.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
