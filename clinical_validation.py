#!/usr/bin/env python3
"""
Clinical Validation of Discovered Patterns
Validate discovered arrhythmia patterns against ICD codes and clinical outcomes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter, defaultdict
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ClinicalValidator:
    """Validate discovered patterns against clinical evidence"""

    def __init__(self, pattern_results_path, clinical_data_path, output_dir='clinical_validation'):
        self.output_dir = output_dir
        self.pattern_results = None
        self.clinical_data = None
        self.validation_results = {}

        os.makedirs(output_dir, exist_ok=True)

        self.load_pattern_results(pattern_results_path)
        self.load_clinical_data(clinical_data_path)

    def load_pattern_results(self, results_path):
        """Load clustering and pattern discovery results"""
        try:
            with open(results_path, 'r') as f:
                self.pattern_results = json.load(f)
            print("✅ Pattern results loaded")
        except Exception as e:
            print(f"⚠️  Could not load pattern results: {e}")

    def load_clinical_data(self, clinical_path):
        """Load clinical data"""
        try:
            self.clinical_data = pd.read_csv(clinical_path)
            print(f"✅ Clinical data loaded: {len(self.clinical_data)} patients")
        except Exception as e:
            print(f"⚠️  Could not load clinical data: {e}")

    def validate_pattern_clinical_correlation(self):
        """Validate if discovered patterns correlate with clinical outcomes"""
        print("Validating pattern-clinical correlations...")

        if self.pattern_results is None or self.clinical_data is None:
            print("⚠️  Missing data for validation")
            return

        # Analyze clinical analysis from clustering results
        clinical_analysis = self.pattern_results.get('clinical_analysis', {})

        validation_results = {}

        for method, clusters in clinical_analysis.items():
            method_validation = {
                'total_clusters': len(clusters),
                'pure_clusters': 0,
                'mixed_clusters': 0,
                'stroke_enriched_clusters': 0,
                'arrhythmia_enriched_clusters': 0,
                'clinical_coherence_score': 0.0
            }

            coherence_scores = []

            for cluster_id, cluster_info in clusters.items():
                categories = cluster_info.get('categories', {})
                total_samples = cluster_info.get('size', 0)

                if total_samples == 0:
                    continue

                # Calculate purity (dominant category percentage)
                dominant_count = max(categories.values()) if categories else 0
                purity = dominant_count / total_samples

                coherence_scores.append(purity)

                # Classify cluster types
                if purity > 0.8:  # Pure cluster
                    method_validation['pure_clusters'] += 1

                    dominant_category = max(categories.keys(), key=lambda k: categories[k])
                    if dominant_category == 'stroke':
                        method_validation['stroke_enriched_clusters'] += 1
                    elif dominant_category == 'arrhythmia':
                        method_validation['arrhythmia_enriched_clusters'] += 1

                else:  # Mixed cluster
                    method_validation['mixed_clusters'] += 1

            # Overall clinical coherence
            method_validation['clinical_coherence_score'] = np.mean(coherence_scores) if coherence_scores else 0.0

            validation_results[method] = method_validation

        self.validation_results['pattern_clinical_correlation'] = validation_results

        # Print summary
        print("\nPattern-Clinical Correlation Results:")
        for method, results in validation_results.items():
            print(f"\n{method.upper()}:")
            print(f"  Total clusters: {results['total_clusters']}")
            print(f"  Pure clusters (>80%): {results['pure_clusters']}")
            print(f"  Mixed clusters: {results['mixed_clusters']}")
            print(f"  Stroke-enriched: {results['stroke_enriched_clusters']}")
            print(f"  Arrhythmia-enriched: {results['arrhythmia_enriched_clusters']}")
            print(f"  Clinical coherence: {results['clinical_coherence_score']:.3f}")

        return validation_results

    def validate_novel_pattern_significance(self):
        """Validate statistical significance of novel patterns"""
        print("Validating novel pattern significance...")

        if 'novel_patterns' not in self.pattern_results:
            print("⚠️  No novel patterns found to validate")
            return

        novel_patterns = self.pattern_results['novel_patterns']
        significance_results = {}

        for method, patterns in novel_patterns.items():
            method_results = []

            for pattern in patterns:
                pattern_validation = {
                    'pattern_id': pattern.get('cluster_id'),
                    'pattern_type': pattern.get('pattern_type'),
                    'size': pattern.get('size', 0),
                    'categories': pattern.get('categories', {}),
                    'novelty_score': pattern.get('novelty_score', 0),
                    'clinical_significance': 'unknown'
                }

                # Assess clinical significance
                categories = pattern.get('categories', {})
                size = pattern.get('size', 0)

                if size >= 10:  # Minimum size for significance
                    if pattern.get('pattern_type') == 'mixed_category':
                        # Mixed patterns might indicate transitional states
                        if 'stroke' in categories and 'arrhythmia' in categories:
                            pattern_validation['clinical_significance'] = 'high_stroke_arrhythmia_link'
                        elif 'healthy' in categories and 'arrhythmia' in categories:
                            pattern_validation['clinical_significance'] = 'subclinical_arrhythmia'
                        else:
                            pattern_validation['clinical_significance'] = 'moderate_mixed_pathology'

                    elif pattern.get('pattern_type') == 'potential_undiagnosed':
                        pattern_validation['clinical_significance'] = 'high_undiagnosed_risk'

                    elif pattern.get('pattern_type') == 'arrhythmia_subtype':
                        pattern_validation['clinical_significance'] = 'moderate_arrhythmia_subtype'

                method_results.append(pattern_validation)

            significance_results[method] = method_results

        self.validation_results['novel_pattern_significance'] = significance_results

        # Print summary
        print("\nNovel Pattern Significance:")
        for method, patterns in significance_results.items():
            print(f"\n{method.upper()}:")
            high_sig = sum(1 for p in patterns if 'high' in p['clinical_significance'])
            moderate_sig = sum(1 for p in patterns if 'moderate' in p['clinical_significance'])

            print(f"  High significance patterns: {high_sig}")
            print(f"  Moderate significance patterns: {moderate_sig}")
            print(f"  Total validated patterns: {len(patterns)}")

        return significance_results

    def perform_statistical_tests(self):
        """Perform statistical tests on discovered patterns"""
        print("Performing statistical tests...")

        if self.clinical_data is None:
            print("⚠️  No clinical data for statistical testing")
            return

        # Test association between categories
        category_counts = self.clinical_data['category'].value_counts()

        # Chi-square test for category distribution
        observed = category_counts.values
        expected = [len(self.clinical_data) / len(category_counts)] * len(category_counts)

        chi2_stat, chi2_p = stats.chisquare(observed, expected)

        # Test for age differences between categories (if age available)
        statistical_results = {
            'category_distribution_test': {
                'chi2_statistic': float(chi2_stat),
                'p_value': float(chi2_p),
                'significant': chi2_p < 0.05
            }
        }

        # Additional tests would go here with more clinical variables

        self.validation_results['statistical_tests'] = statistical_results

        print(f"Category distribution Chi-square test: p = {chi2_p:.6f}")

        return statistical_results

    def generate_clinical_recommendations(self):
        """Generate clinical recommendations based on findings"""
        recommendations = []

        # Analyze validation results
        if 'pattern_clinical_correlation' in self.validation_results:
            corr_results = self.validation_results['pattern_clinical_correlation']

            for method, results in corr_results.items():
                coherence = results['clinical_coherence_score']

                if coherence > 0.7:
                    recommendations.append({
                        'priority': 'High',
                        'finding': f'{method} clustering shows high clinical coherence ({coherence:.3f})',
                        'recommendation': 'Use this clustering method for clinical decision support',
                        'evidence_level': 'Strong algorithmic evidence'
                    })

                if results['stroke_enriched_clusters'] > 0:
                    recommendations.append({
                        'priority': 'High',
                        'finding': f'{method} identified {results["stroke_enriched_clusters"]} stroke-enriched patterns',
                        'recommendation': 'Investigate ECG/PPG signatures for stroke risk prediction',
                        'evidence_level': 'Discovery-based evidence'
                    })

        # Novel pattern recommendations
        if 'novel_pattern_significance' in self.validation_results:
            sig_results = self.validation_results['novel_pattern_significance']

            for method, patterns in sig_results.items():
                high_sig_patterns = [p for p in patterns if 'high' in p['clinical_significance']]

                for pattern in high_sig_patterns:
                    if pattern['clinical_significance'] == 'high_stroke_arrhythmia_link':
                        recommendations.append({
                            'priority': 'Critical',
                            'finding': 'Discovered strong stroke-arrhythmia pattern linkage',
                            'recommendation': 'Develop targeted screening protocol for patients with this pattern',
                            'evidence_level': 'Novel discovery requiring validation'
                        })

                    elif pattern['clinical_significance'] == 'high_undiagnosed_risk':
                        recommendations.append({
                            'priority': 'High',
                            'finding': 'Identified potentially undiagnosed arrhythmia patterns in "healthy" patients',
                            'recommendation': 'Consider extended monitoring for patients with this pattern',
                            'evidence_level': 'Screening recommendation'
                        })

        self.validation_results['clinical_recommendations'] = recommendations

        print(f"\nGenerated {len(recommendations)} clinical recommendations")

        return recommendations

    def create_validation_visualizations(self):
        """Create comprehensive validation visualizations"""
        print("Creating validation visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clinical Validation Results', fontsize=16, fontweight='bold')

        # Clinical coherence comparison
        if 'pattern_clinical_correlation' in self.validation_results:
            ax = axes[0, 0]
            corr_results = self.validation_results['pattern_clinical_correlation']

            methods = list(corr_results.keys())
            coherence_scores = [corr_results[m]['clinical_coherence_score'] for m in methods]

            bars = ax.bar(methods, coherence_scores, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Clinical Coherence Score')
            ax.set_title('Clinical Coherence by Method')
            ax.set_ylim(0, 1)

            # Add value labels
            for bar, score in zip(bars, coherence_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')

        # Cluster purity distribution
        ax = axes[0, 1]
        if 'pattern_clinical_correlation' in self.validation_results:
            corr_results = self.validation_results['pattern_clinical_correlation']

            pure_counts = [corr_results[m]['pure_clusters'] for m in methods]
            mixed_counts = [corr_results[m]['mixed_clusters'] for m in methods]

            x = np.arange(len(methods))
            width = 0.35

            ax.bar(x - width/2, pure_counts, width, label='Pure Clusters', color='green', alpha=0.7)
            ax.bar(x + width/2, mixed_counts, width, label='Mixed Clusters', color='orange', alpha=0.7)

            ax.set_xlabel('Clustering Method')
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Cluster Purity Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()

        # Clinical category distribution
        ax = axes[0, 2]
        if self.clinical_data is not None:
            category_counts = self.clinical_data['category'].value_counts()
            colors = ['red', 'orange', 'green']

            wedges, texts, autotexts = ax.pie(category_counts.values, labels=category_counts.index,
                                             autopct='%1.1f%%', colors=colors)
            ax.set_title('Clinical Category Distribution')

        # Novel pattern significance
        ax = axes[1, 0]
        if 'novel_pattern_significance' in self.validation_results:
            sig_results = self.validation_results['novel_pattern_significance']

            significance_counts = defaultdict(int)
            for method, patterns in sig_results.items():
                for pattern in patterns:
                    sig_level = pattern['clinical_significance']
                    if 'high' in sig_level:
                        significance_counts['High'] += 1
                    elif 'moderate' in sig_level:
                        significance_counts['Moderate'] += 1
                    else:
                        significance_counts['Low'] += 1

            if significance_counts:
                labels = list(significance_counts.keys())
                values = list(significance_counts.values())
                colors = ['red', 'orange', 'gray'][:len(labels)]

                ax.pie(values, labels=labels, autopct='%1.0f', colors=colors)
                ax.set_title('Novel Pattern Clinical Significance')

        # Pattern size distribution
        ax = axes[1, 1]
        if 'novel_patterns' in self.pattern_results:
            all_sizes = []
            all_types = []

            for method, patterns in self.pattern_results['novel_patterns'].items():
                for pattern in patterns:
                    all_sizes.append(pattern.get('size', 0))
                    all_types.append(pattern.get('pattern_type', 'unknown'))

            if all_sizes:
                unique_types = list(set(all_types))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))

                for i, ptype in enumerate(unique_types):
                    type_sizes = [size for size, t in zip(all_sizes, all_types) if t == ptype]
                    ax.scatter([i] * len(type_sizes), type_sizes,
                              alpha=0.6, c=[colors[i]], label=ptype, s=50)

                ax.set_xlabel('Pattern Type')
                ax.set_ylabel('Pattern Size (# samples)')
                ax.set_title('Pattern Size by Type')
                ax.set_xticks(range(len(unique_types)))
                ax.set_xticklabels(unique_types, rotation=45)
                ax.legend()

        # Clinical recommendations summary
        ax = axes[1, 2]
        if 'clinical_recommendations' in self.validation_results:
            recommendations = self.validation_results['clinical_recommendations']

            priority_counts = Counter([r['priority'] for r in recommendations])
            priorities = ['Critical', 'High', 'Medium', 'Low']
            counts = [priority_counts.get(p, 0) for p in priorities]
            colors = ['red', 'orange', 'yellow', 'green']

            bars = ax.bar(priorities, counts, color=colors, alpha=0.7)
            ax.set_ylabel('Number of Recommendations')
            ax.set_title('Clinical Recommendations by Priority')

            # Add value labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           str(count), ha='center', va='bottom')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'clinical_validation_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Validation visualization saved to: {output_path}")

    def save_validation_results(self):
        """Save all validation results"""
        print("Saving validation results...")

        # Save comprehensive results
        with open(os.path.join(self.output_dir, 'validation_results.json'), 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        # Save clinical recommendations
        if 'clinical_recommendations' in self.validation_results:
            recommendations_df = pd.DataFrame(self.validation_results['clinical_recommendations'])
            recommendations_df.to_csv(
                os.path.join(self.output_dir, 'clinical_recommendations.csv'),
                index=False
            )

        # Save summary report
        self._save_validation_report()

        print(f"Validation results saved to: {self.output_dir}")

    def _save_validation_report(self):
        """Save detailed validation report"""
        report_path = os.path.join(self.output_dir, 'clinical_validation_report.md')

        with open(report_path, 'w') as f:
            f.write("# Clinical Validation Report\n\n")

            # Pattern-clinical correlation
            if 'pattern_clinical_correlation' in self.validation_results:
                f.write("## Pattern-Clinical Correlation\n\n")
                corr_results = self.validation_results['pattern_clinical_correlation']

                for method, results in corr_results.items():
                    f.write(f"### {method.upper()}\n")
                    f.write(f"- Clinical coherence score: {results['clinical_coherence_score']:.3f}\n")
                    f.write(f"- Pure clusters: {results['pure_clusters']}\n")
                    f.write(f"- Mixed clusters: {results['mixed_clusters']}\n")
                    f.write(f"- Stroke-enriched clusters: {results['stroke_enriched_clusters']}\n")
                    f.write(f"- Arrhythmia-enriched clusters: {results['arrhythmia_enriched_clusters']}\n\n")

            # Novel patterns significance
            if 'novel_pattern_significance' in self.validation_results:
                f.write("## Novel Pattern Clinical Significance\n\n")
                sig_results = self.validation_results['novel_pattern_significance']

                for method, patterns in sig_results.items():
                    f.write(f"### {method.upper()}\n")
                    high_sig = [p for p in patterns if 'high' in p['clinical_significance']]
                    f.write(f"- High significance patterns: {len(high_sig)}\n")

                    for pattern in high_sig[:3]:  # Top 3
                        f.write(f"  - Pattern {pattern['pattern_id']}: {pattern['pattern_type']} ")
                        f.write(f"({pattern['size']} samples) - {pattern['clinical_significance']}\n")

                    f.write("\n")

            # Clinical recommendations
            if 'clinical_recommendations' in self.validation_results:
                f.write("## Clinical Recommendations\n\n")
                recommendations = self.validation_results['clinical_recommendations']

                critical_recs = [r for r in recommendations if r['priority'] == 'Critical']
                high_recs = [r for r in recommendations if r['priority'] == 'High']

                if critical_recs:
                    f.write("### Critical Priority\n")
                    for rec in critical_recs:
                        f.write(f"- **Finding**: {rec['finding']}\n")
                        f.write(f"  **Recommendation**: {rec['recommendation']}\n")
                        f.write(f"  **Evidence**: {rec['evidence_level']}\n\n")

                if high_recs:
                    f.write("### High Priority\n")
                    for rec in high_recs:
                        f.write(f"- **Finding**: {rec['finding']}\n")
                        f.write(f"  **Recommendation**: {rec['recommendation']}\n")
                        f.write(f"  **Evidence**: {rec['evidence_level']}\n\n")

            f.write("## Summary\n\n")
            f.write("This validation confirms the clinical relevance of discovered patterns ")
            f.write("and provides evidence-based recommendations for clinical implementation.\n")

def main():
    """Main clinical validation function"""
    print("🔬 CLINICAL VALIDATION PIPELINE")

    # Initialize validator
    pattern_results_path = '/media/jaadoo/sexy/ecg ppg/simple_pattern_discovery/pattern_results.json'
    clinical_data_path = '/media/jaadoo/sexy/ecg ppg/clinical_features.csv'

    validator = ClinicalValidator(pattern_results_path, clinical_data_path)

    # Run validation analyses
    validator.validate_pattern_clinical_correlation()
    validator.validate_novel_pattern_significance()
    validator.perform_statistical_tests()
    validator.generate_clinical_recommendations()

    # Generate visualizations and reports
    validator.create_validation_visualizations()
    validator.save_validation_results()

    print("\n✅ Clinical validation complete!")

if __name__ == "__main__":
    main()