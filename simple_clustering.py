#!/usr/bin/env python3
"""
Simplified Clustering and Pattern Discovery Pipeline
Uses only pre-installed packages to discover novel arrhythmia patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import os
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from contrastive_model import WaveformEncoder

class SimpleEmbeddingExtractor:
    """Extract embeddings from trained model"""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model"""
        print(f"Loading model from: {model_path}")

        # Create model architecture (should match training)
        self.model = WaveformEncoder(
            input_channels=6,  # Max channels in dataset
            hidden_dims=[32, 64, 128, 256],
            embedding_dim=128,
            dropout=0.1
        ).to(self.device)

        # Load weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ Model loaded successfully")
        else:
            print(f"⚠️  Model not found at {model_path}, using random initialization")

    def extract_embeddings(self, segments, metadata, batch_size=32):
        """Extract embeddings for segments"""
        print("Extracting embeddings...")

        # Create simple dataset for inference
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(segments))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                segments_batch = batch[0].to(self.device)

                # Ensure correct channel ordering: (batch, channels, time)
                if segments_batch.dim() == 3 and segments_batch.shape[1] != 6:
                    segments_batch = segments_batch.transpose(1, 2)

                # Extract embeddings
                embeddings = self.model(segments_batch)
                embeddings = F.normalize(embeddings, dim=1)  # L2 normalize

                all_embeddings.append(embeddings.cpu().numpy())

        embeddings_array = np.concatenate(all_embeddings, axis=0)

        # Check for and fix NaN values
        nan_mask = np.isnan(embeddings_array)
        if nan_mask.any():
            print(f"⚠️  Found {nan_mask.sum()} NaN values, replacing with zeros")
            embeddings_array = np.nan_to_num(embeddings_array, nan=0.0)

        print(f"Extracted embeddings shape: {embeddings_array.shape}")

        return embeddings_array

class SimplePatternDiscovery:
    """Simplified pattern discovery using standard sklearn"""

    def __init__(self, embeddings, metadata, output_dir='simple_pattern_discovery'):
        self.embeddings = embeddings
        self.metadata = metadata
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def reduce_dimensions(self):
        """Reduce dimensionality for visualization"""
        print("Reducing dimensions...")

        # PCA to 50 components
        pca = PCA(n_components=min(50, self.embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(self.embeddings)

        # PCA to 2D for visualization
        pca_2d = PCA(n_components=2)
        pca_2d_embeddings = pca_2d.fit_transform(self.embeddings)

        # t-SNE for visualization (subsample for speed)
        sample_size = min(1000, len(self.embeddings))
        indices = np.random.choice(len(self.embeddings), sample_size, replace=False)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_embeddings = tsne.fit_transform(pca_embeddings[indices])

        self.reduced_embeddings = {
            'pca': pca_embeddings,
            'pca_2d': pca_2d_embeddings,
            'tsne': tsne_embeddings,
            'tsne_indices': indices
        }

        print(f"PCA explained variance (top 10): {pca.explained_variance_ratio_[:10]}")

        return self.reduced_embeddings

    def cluster_embeddings(self):
        """Apply clustering algorithms"""
        print("Clustering embeddings...")

        pca_embeddings = self.reduced_embeddings['pca']

        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(pca_embeddings)

        # K-means clustering (try different k values)
        best_k = 3
        best_score = -1

        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(pca_embeddings)

            score = silhouette_score(pca_embeddings, kmeans_labels)
            if score > best_score:
                best_score = score
                best_k = k

        # Final k-means with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(pca_embeddings)

        self.clustering_results = {
            'dbscan': dbscan_labels,
            'kmeans': kmeans_labels
        }

        # Evaluate clustering
        for method, labels in self.clustering_results.items():
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters > 1 and -1 not in labels:
                silhouette_avg = silhouette_score(pca_embeddings, labels)
                print(f"{method.upper()}: {n_clusters} clusters, silhouette: {silhouette_avg:.3f}")
            else:
                print(f"{method.upper()}: {n_clusters} clusters, {n_noise} noise points")

        return self.clustering_results

    def analyze_clinical_patterns(self):
        """Analyze clustering results against clinical categories"""
        print("Analyzing clinical patterns...")

        clinical_analysis = {}

        for method, labels in self.clustering_results.items():
            print(f"\n{method.upper()} Clinical Analysis:")

            method_analysis = {}

            for cluster_id in set(labels):
                if cluster_id == -1:  # Skip noise
                    continue

                # Get samples in this cluster
                cluster_mask = labels == cluster_id
                cluster_metadata = [self.metadata[i] for i in range(len(self.metadata)) if cluster_mask[i]]

                # Count clinical categories
                categories = [meta['clinical_category'] for meta in cluster_metadata]
                category_counts = Counter(categories)

                method_analysis[cluster_id] = {
                    'size': sum(cluster_mask),
                    'categories': dict(category_counts),
                    'dominant_category': category_counts.most_common(1)[0][0] if category_counts else 'unknown'
                }

                print(f"  Cluster {cluster_id}: {sum(cluster_mask)} samples")
                for category, count in category_counts.items():
                    percentage = (count / sum(cluster_mask)) * 100
                    print(f"    {category}: {count} ({percentage:.1f}%)")

            clinical_analysis[method] = method_analysis

        self.clinical_analysis = clinical_analysis
        return clinical_analysis

    def discover_novel_patterns(self):
        """Identify potentially novel patterns"""
        print("Discovering novel patterns...")

        novel_patterns = {}

        for method, labels in self.clustering_results.items():
            method_patterns = []

            for cluster_id in set(labels):
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_mask = labels == cluster_id
                cluster_metadata = [self.metadata[i] for i in range(len(self.metadata)) if cluster_mask[i]]

                # Count categories
                categories = [meta['clinical_category'] for meta in cluster_metadata]
                category_counts = Counter(categories)

                # Determine if pattern is novel
                is_novel = False
                pattern_type = "unknown"

                cluster_size = sum(cluster_mask)

                # Mixed category clusters
                if len(category_counts) > 1:
                    is_novel = True
                    pattern_type = "mixed_category"

                # Large healthy clusters (potential undiagnosed)
                elif (cluster_size >= 5 and
                      category_counts.most_common(1)[0][0] == 'healthy' and
                      category_counts.most_common(1)[0][1] / cluster_size > 0.8):
                    is_novel = True
                    pattern_type = "potential_undiagnosed"

                # Arrhythmia subclusters
                elif (category_counts.most_common(1)[0][0] == 'arrhythmia' and
                      cluster_size >= 3):
                    is_novel = True
                    pattern_type = "arrhythmia_subtype"

                if is_novel:
                    # Get patient IDs
                    patient_ids = [meta['patient_id'] for meta in cluster_metadata]

                    pattern = {
                        'cluster_id': cluster_id,
                        'pattern_type': pattern_type,
                        'size': cluster_size,
                        'categories': dict(category_counts),
                        'patient_ids': patient_ids,
                        'novelty_score': self._calculate_novelty_score(category_counts, cluster_size)
                    }
                    method_patterns.append(pattern)

            # Sort by novelty score
            method_patterns.sort(key=lambda x: x['novelty_score'], reverse=True)
            novel_patterns[method] = method_patterns

            print(f"{method.upper()}: Found {len(method_patterns)} novel patterns")

        self.novel_patterns = novel_patterns
        return novel_patterns

    def _calculate_novelty_score(self, category_counts, cluster_size):
        """Calculate novelty score"""
        size_factor = min(1.0, cluster_size / 10)
        diversity_factor = len(category_counts) / 3.0

        # Bonus for stroke-arrhythmia combinations
        if 'stroke' in category_counts and 'arrhythmia' in category_counts:
            diversity_factor *= 1.5

        return size_factor * diversity_factor

    def visualize_patterns(self):
        """Create visualizations"""
        print("Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ECG/PPG Pattern Discovery Results', fontsize=16)

        # Clinical categories distribution
        ax = axes[0, 0]
        categories = [meta['clinical_category'] for meta in self.metadata]
        category_counts = Counter(categories)
        ax.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
        ax.set_title('Clinical Category Distribution')

        # PCA visualization
        ax = axes[0, 1]
        pca_2d = self.reduced_embeddings['pca_2d']
        categories = [meta['clinical_category'] for meta in self.metadata]

        category_colors = {'stroke': 'red', 'arrhythmia': 'orange', 'healthy': 'green', 'unknown': 'gray'}
        colors = [category_colors.get(cat, 'gray') for cat in categories]

        ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax.set_title('PCA - Clinical Categories')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # t-SNE visualization
        ax = axes[0, 2]
        tsne_embeddings = self.reduced_embeddings['tsne']
        tsne_indices = self.reduced_embeddings['tsne_indices']
        tsne_categories = [self.metadata[i]['clinical_category'] for i in tsne_indices]
        tsne_colors = [category_colors.get(cat, 'gray') for cat in tsne_categories]

        ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=tsne_colors, alpha=0.6, s=20)
        ax.set_title('t-SNE - Clinical Categories')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        # Clustering results
        methods = ['dbscan', 'kmeans']
        for i, method in enumerate(methods):
            ax = axes[1, i]
            labels = self.clustering_results[method]
            pca_2d = self.reduced_embeddings['pca_2d']

            unique_labels = set(labels)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for k, color in zip(unique_labels, colors):
                if k == -1:
                    class_member_mask = (labels == k)
                    xy = pca_2d[class_member_mask]
                    ax.plot(xy[:, 0], xy[:, 1], 'k.', markersize=2, alpha=0.3)
                else:
                    class_member_mask = (labels == k)
                    xy = pca_2d[class_member_mask]
                    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color,
                           markeredgecolor='k', markersize=3, alpha=0.7)

            ax.set_title(f'{method.upper()} Clustering')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')

        # Novel patterns summary
        ax = axes[1, 2]
        pattern_counts = []
        pattern_labels = []

        for method, patterns in self.novel_patterns.items():
            type_counts = Counter([p['pattern_type'] for p in patterns])
            for pattern_type, count in type_counts.items():
                pattern_counts.append(count)
                pattern_labels.append(f"{method}\n{pattern_type}")

        if pattern_counts:
            ax.bar(range(len(pattern_counts)), pattern_counts)
            ax.set_xticks(range(len(pattern_labels)))
            ax.set_xticklabels(pattern_labels, rotation=45, ha='right')
            ax.set_title('Novel Patterns Discovered')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No novel patterns found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Novel Patterns Discovered')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'pattern_discovery_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {output_path}")

    def save_results(self):
        """Save results"""
        print("Saving results...")

        # Save results as JSON
        results_dict = {
            'clustering_results': {k: v.tolist() for k, v in self.clustering_results.items()},
            'clinical_analysis': self.clinical_analysis,
            'novel_patterns': self.novel_patterns
        }

        with open(os.path.join(self.output_dir, 'pattern_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save summary report
        report_path = os.path.join(self.output_dir, 'pattern_summary.txt')
        with open(report_path, 'w') as f:
            f.write("ECG/PPG Pattern Discovery Summary\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Total samples: {len(self.embeddings)}\n")
            f.write(f"Embedding dimension: {self.embeddings.shape[1]}\n\n")

            for method, patterns in self.novel_patterns.items():
                f.write(f"{method.upper()} Novel Patterns: {len(patterns)}\n")
                for i, pattern in enumerate(patterns[:5]):  # Top 5
                    f.write(f"  {i+1}. {pattern['pattern_type']} - {pattern['size']} samples\n")
                f.write("\n")

        print(f"Results saved to: {self.output_dir}")

def main():
    """Main pattern discovery function"""
    print("🔍 SIMPLE PATTERN DISCOVERY")

    # Load integrated dataset
    data = np.load('/media/jaadoo/sexy/ecg ppg/integrated_dataset.npz', allow_pickle=True)
    segments = data['segments']
    segment_metadata = data['segment_metadata']

    print(f"Loaded dataset: {segments.shape}")

    # Load trained model
    model_path = '/media/jaadoo/sexy/ecg ppg/best_fixed_model.pth'

    # Extract embeddings
    extractor = SimpleEmbeddingExtractor(model_path)
    embeddings = extractor.extract_embeddings(segments, segment_metadata)

    # Discover patterns
    discovery = SimplePatternDiscovery(embeddings, segment_metadata)

    # Run analysis
    discovery.reduce_dimensions()
    discovery.cluster_embeddings()
    discovery.analyze_clinical_patterns()
    discovery.discover_novel_patterns()
    discovery.visualize_patterns()
    discovery.save_results()

    print("\n✅ Simple pattern discovery complete!")

if __name__ == "__main__":
    main()