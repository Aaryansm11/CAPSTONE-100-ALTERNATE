#!/usr/bin/env python3
"""
FIXED Simple Pattern Discovery with Clustering
✅ Works with HDF5 format (full_dataset.h5)
✅ Auto-detects model architecture from checkpoint
✅ Handles any input channel count and embedding dimensions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import json
import pickle
import os
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

# Import model architecture
from contrastive_model import WaveformEncoder

class AutoDetectEmbeddingExtractor:
    """Extract embeddings with automatic architecture detection"""

    def __init__(self, checkpoint_path, config_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.load_model()

    def load_model(self):
        """Load model with auto-detected parameters"""
        print(f"\n🔍 Auto-detecting model architecture...")

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # Auto-detect input channels from checkpoint
        state_dict = checkpoint['model_state_dict']
        input_channels = state_dict['input_conv.weight'].shape[1]

        print(f"  ✓ Detected input channels: {input_channels}")
        print(f"  ✓ Embedding dim: {self.config['embedding_dim']}")
        print(f"  ✓ Hidden dims: {self.config['hidden_dims']}")

        # Create model with detected params
        self.model = WaveformEncoder(
            input_channels=input_channels,
            hidden_dims=self.config['hidden_dims'],
            embedding_dim=self.config['embedding_dim'],
            dropout=0.1
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def extract_embeddings(self, segments, batch_size=64):
        """Extract embeddings for segments"""
        print(f"\n🧠 Extracting embeddings...")
        print(f"  • Input shape: {segments.shape}")
        print(f"  • Batch size: {batch_size}")

        # Convert to tensor and ensure correct shape: (batch, channels, seq_len)
        if isinstance(segments, np.ndarray):
            segments_tensor = torch.FloatTensor(segments)
        else:
            segments_tensor = segments

        # Check and transpose if needed
        # segments come as (batch, seq_len, channels) from HDF5
        # model needs (batch, channels, seq_len)
        if segments_tensor.shape[1] > segments_tensor.shape[2]:
            # Likely (batch, seq_len, channels), need to transpose
            segments_tensor = segments_tensor.transpose(1, 2)
            print(f"  ✓ Transposed to {segments_tensor.shape}")

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(segments_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Processing batches"):
                segments_batch = batch[0].to(self.device)

                # Extract embeddings
                embeddings = self.model(segments_batch)
                embeddings = F.normalize(embeddings, dim=1)  # L2 normalize
                all_embeddings.append(embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        print(f"  ✓ Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

        return embeddings


class SimplePatternDiscovery:
    """Simple pattern discovery using clustering"""

    def __init__(self, embeddings, segment_metadata, output_dir='production_medium/simple_pattern_discovery'):
        self.embeddings = embeddings
        self.segment_metadata = segment_metadata
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.reduced_embeddings = None
        self.cluster_labels = None
        self.n_clusters = None

    def reduce_dimensions(self, n_components=50):
        """Reduce dimensions with PCA for clustering"""
        print(f"\n📉 Reducing dimensions...")
        print(f"  • Original dim: {self.embeddings.shape[1]}")
        print(f"  • Target dim: {n_components}")

        pca = PCA(n_components=n_components, random_state=42)
        self.reduced_embeddings = pca.fit_transform(self.embeddings)

        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  ✓ Explained variance: {explained_var:.1%}")

        return self.reduced_embeddings

    def cluster_embeddings(self, method='kmeans', n_clusters=9):
        """Cluster embeddings"""
        print(f"\n🎯 Clustering with {method}...")

        if self.reduced_embeddings is None:
            self.reduce_dimensions()

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
            self.n_clusters = n_clusters
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
            self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.cluster_labels = clusterer.fit_predict(self.reduced_embeddings)
            self.n_clusters = n_clusters

        # Compute metrics
        if len(set(self.cluster_labels)) > 1:
            silhouette = silhouette_score(self.reduced_embeddings, self.cluster_labels)
            davies_bouldin = davies_bouldin_score(self.reduced_embeddings, self.cluster_labels)
            calinski = calinski_harabasz_score(self.reduced_embeddings, self.cluster_labels)

            print(f"  ✓ Clusters found: {self.n_clusters}")
            print(f"  ✓ Silhouette score: {silhouette:.3f}")
            print(f"  ✓ Davies-Bouldin: {davies_bouldin:.3f}")
            print(f"  ✓ Calinski-Harabasz: {calinski:.1f}")
        else:
            print(f"  ⚠ Only one cluster found")

        # Cluster distribution
        cluster_counts = Counter(self.cluster_labels)
        print(f"\n  📊 Cluster distribution:")
        for cluster_id, count in sorted(cluster_counts.items()):
            pct = 100 * count / len(self.cluster_labels)
            print(f"    Cluster {cluster_id}: {count:,} samples ({pct:.1f}%)")

        return self.cluster_labels

    def analyze_clinical_patterns(self):
        """Analyze clinical characteristics of each cluster"""
        print(f"\n🏥 Analyzing clinical patterns...")

        if self.cluster_labels is None:
            print("  ⚠ Run clustering first!")
            return

        clinical_analysis = {}

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_metadata = [self.segment_metadata[i] for i in np.where(cluster_mask)[0]]

            # Aggregate statistics
            has_stroke = sum(1 for m in cluster_metadata if m.get('has_stroke', 0) == 1)
            has_arrhythmia = sum(1 for m in cluster_metadata if m.get('has_arrhythmia', 0) == 1)
            total = len(cluster_metadata)

            # Age statistics
            ages = [m.get('age', 65) for m in cluster_metadata]
            avg_age = np.mean(ages)

            # Gender distribution
            males = sum(1 for m in cluster_metadata if m.get('gender', 0) == 1)

            clinical_analysis[cluster_id] = {
                'total_samples': total,
                'stroke_rate': has_stroke / total if total > 0 else 0,
                'arrhythmia_rate': has_arrhythmia / total if total > 0 else 0,
                'avg_age': avg_age,
                'male_pct': males / total if total > 0 else 0
            }

            print(f"\n  Cluster {cluster_id}:")
            print(f"    • Samples: {total:,}")
            print(f"    • Stroke rate: {100*clinical_analysis[cluster_id]['stroke_rate']:.1f}%")
            print(f"    • Arrhythmia rate: {100*clinical_analysis[cluster_id]['arrhythmia_rate']:.1f}%")
            print(f"    • Avg age: {avg_age:.1f}")
            print(f"    • Male: {100*clinical_analysis[cluster_id]['male_pct']:.1f}%")

        # Save analysis
        with open(f"{self.output_dir}/clinical_analysis.json", 'w') as f:
            json.dump(clinical_analysis, f, indent=2)

        return clinical_analysis

    def discover_novel_patterns(self):
        """Identify novel/unusual patterns"""
        print(f"\n🔬 Discovering novel patterns...")

        # Find clusters with unusual characteristics
        novel_patterns = []

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_size = cluster_mask.sum()

            # Check for small, distinct clusters (potential novel patterns)
            if cluster_size < 0.05 * len(self.cluster_labels):  # Less than 5% of data
                print(f"  🆕 Small cluster {cluster_id}: {cluster_size} samples (potential novel pattern)")
                novel_patterns.append(cluster_id)

        return novel_patterns

    def visualize_patterns(self):
        """Visualize clustering results"""
        print(f"\n📊 Creating visualizations...")

        # UMAP for 2D visualization
        print("  • Computing UMAP...")
        umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_embeddings = umap_reducer.fit_transform(self.reduced_embeddings)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Clusters
        scatter = axes[0].scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c=self.cluster_labels,
            cmap='tab10',
            s=1,
            alpha=0.6
        )
        axes[0].set_title(f'Pattern Discovery: {self.n_clusters} Clusters', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')

        # Plot 2: Stroke status
        stroke_status = np.array([m.get('has_stroke', 0) for m in self.segment_metadata])
        scatter2 = axes[1].scatter(
            umap_embeddings[:, 0],
            umap_embeddings[:, 1],
            c=stroke_status,
            cmap='RdYlGn_r',
            s=1,
            alpha=0.6
        )
        axes[1].set_title('Stroke Status Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[1], label='Stroke')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pattern_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir}/pattern_visualization.png")

    def save_results(self):
        """Save clustering results"""
        print(f"\n💾 Saving results...")

        results = {
            'cluster_labels': self.cluster_labels.tolist(),
            'n_clusters': self.n_clusters,
            'embeddings_shape': self.embeddings.shape,
            'total_segments': len(self.embeddings)
        }

        with open(f"{self.output_dir}/clustering_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Save cluster assignments with metadata
        cluster_assignments = []
        for i, label in enumerate(self.cluster_labels):
            cluster_assignments.append({
                'segment_idx': i,
                'cluster': int(label),
                'patient_id': self.segment_metadata[i].get('patient_id'),
                'has_stroke': self.segment_metadata[i].get('has_stroke', 0),
                'has_arrhythmia': self.segment_metadata[i].get('has_arrhythmia', 0)
            })

        with open(f"{self.output_dir}/cluster_assignments.json", 'w') as f:
            json.dump(cluster_assignments, f, indent=2)

        print(f"  ✓ Results saved to: {self.output_dir}")


def main():
    """Main pattern discovery function"""
    print("=" * 70)
    print("🔍 SIMPLE PATTERN DISCOVERY (FIXED)")
    print("=" * 70)

    # Auto-detect paths
    data_path = 'production_medium/full_dataset.h5'
    metadata_path = 'production_medium/full_dataset_metadata.pkl'
    checkpoint_path = 'production_medium_corrected/latest_checkpoint.pth'  # Use corrected model!
    config_path = 'production_medium/config.json'

    # Check if corrected checkpoint exists, otherwise fall back
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'production_medium/latest_checkpoint.pth'
        print(f"⚠ Using old checkpoint (may have representation collapse)")

    print(f"\n📂 Loading data:")
    print(f"  • Dataset: {data_path}")
    print(f"  • Checkpoint: {checkpoint_path}")
    print(f"  • Config: {config_path}")

    # Load dataset
    with h5py.File(data_path, 'r') as h5f:
        segments = h5f['segments'][:]

    # Load metadata
    with open(metadata_path, 'rb') as f:
        segment_metadata = pickle.load(f)

    print(f"  ✓ Loaded {len(segments):,} segments")
    print(f"  ✓ Shape: {segments.shape}")

    # Extract embeddings
    extractor = AutoDetectEmbeddingExtractor(checkpoint_path, config_path)
    embeddings = extractor.extract_embeddings(segments)

    # Discover patterns
    discovery = SimplePatternDiscovery(embeddings, segment_metadata)

    # Run analysis pipeline
    discovery.reduce_dimensions(n_components=50)
    discovery.cluster_embeddings(method='kmeans', n_clusters=9)
    discovery.analyze_clinical_patterns()
    discovery.discover_novel_patterns()
    discovery.visualize_patterns()
    discovery.save_results()

    print("\n" + "=" * 70)
    print("✅ PATTERN DISCOVERY COMPLETE!")
    print("=" * 70)
    print(f"  • Output directory: {discovery.output_dir}")
    print(f"  • Clusters found: {discovery.n_clusters}")
    print(f"  • Total segments analyzed: {len(embeddings):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
