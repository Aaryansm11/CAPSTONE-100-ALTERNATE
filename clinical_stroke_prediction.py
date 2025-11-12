#!/usr/bin/env python3
"""
Clinical Stroke Prediction Model
=================================
Supervised stroke risk prediction using:
- Frozen embeddings from contrastive model
- Full MIMIC clinical data (ICD-9, medications, demographics)
- Proper supervised learning approach
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import json
import pickle
import os
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score,
    f1_score, recall_score, precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from contrastive_model import WaveformEncoder

class ClinicalFeatureExtractor:
    """Extract comprehensive clinical features from MIMIC data"""

    def __init__(self, mimic_path="dataset/extracted"):
        self.mimic_path = Path(mimic_path)
        self.icd9_cache = {}
        self.med_cache = {}

        # Important ICD-9 codes for stroke risk
        self.stroke_risk_codes = {
            # Cardiovascular
            'atrial_fib': ['42731'],  # Atrial fibrillation
            'heart_failure': ['4280', '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833', '42840', '42841', '42842', '42843'],
            'hypertension': ['4019', '4010', '4011'],
            'diabetes': ['25000', '25001', '25002', '25003'],
            'cad': ['41401', '41400'],  # Coronary artery disease
            # Prior vascular events
            'prior_stroke': ['430', '431', '432', '433', '434', '435', '436', '437', '438'],
            'tia': ['435'],
            'mi': ['410'],  # Myocardial infarction
            # Risk factors
            'obesity': ['2780'],
            'hyperlipidemia': ['2720', '2721', '2722'],
            'smoking': ['V1582', '3051'],
        }

        # Medications associated with stroke risk/prevention
        self.medication_keywords = {
            'anticoagulant': ['warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban'],
            'antiplatelet': ['aspirin', 'clopidogrel', 'prasugrel'],
            'antiarrhythmic': ['amiodarone', 'sotalol', 'flecainide', 'propafenone'],
            'statin': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin'],
            'bp_med': ['lisinopril', 'metoprolol', 'amlodipine', 'losartan'],
        }

    def load_diagnoses(self):
        """Load ICD-9 diagnoses"""
        print("  Loading diagnoses...")
        diag_path = self.mimic_path / 'uploads/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz'
        try:
            df = pd.read_csv(diag_path, compression='gzip')
            print(f"    Loaded {len(df):,} diagnosis records")
            return df
        except Exception as e:
            print(f"    Warning: Could not load diagnoses: {e}")
            return pd.DataFrame()

    def load_prescriptions(self):
        """Load medication data"""
        print("  Loading prescriptions...")
        rx_path = self.mimic_path / 'uploads/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv.gz'
        try:
            df = pd.read_csv(rx_path, compression='gzip')
            print(f"    Loaded {len(df):,} prescription records")
            return df
        except Exception as e:
            print(f"    Warning: Could not load prescriptions: {e}")
            return pd.DataFrame()

    def load_patients(self):
        """Load patient demographics"""
        print("  Loading patients...")
        pt_path = self.mimic_path / 'uploads/physionet.org/files/mimiciii/1.4/PATIENTS.csv.gz'
        try:
            df = pd.read_csv(pt_path, compression='gzip')
            print(f"    Loaded {len(df):,} patient records")
            return df
        except Exception as e:
            print(f"    Warning: Could not load patients: {e}")
            return pd.DataFrame()

    def extract_icd9_features(self, patient_id, diagnoses_df):
        """Extract ICD-9 based features for a patient"""
        if patient_id in self.icd9_cache:
            return self.icd9_cache[patient_id]

        if len(diagnoses_df) == 0:
            return {}

        # Get all diagnoses for this patient
        patient_dx = diagnoses_df[diagnoses_df['SUBJECT_ID'] == patient_id]

        if len(patient_dx) == 0:
            return {}

        # Extract ICD-9 codes
        icd9_codes = set(patient_dx['ICD9_CODE'].astype(str).values)

        features = {}

        # Check for each risk factor category
        for category, codes in self.stroke_risk_codes.items():
            has_condition = any(
                any(icd9.startswith(code) for code in codes)
                for icd9 in icd9_codes
            )
            features[f'icd9_{category}'] = int(has_condition)

        # Count total diagnoses
        features['icd9_total_dx'] = len(icd9_codes)

        # Comorbidity count (Charlson-like)
        comorbidity_categories = ['heart_failure', 'diabetes', 'hypertension', 'cad', 'prior_stroke']
        features['icd9_comorbidity_count'] = sum(
            features.get(f'icd9_{cat}', 0) for cat in comorbidity_categories
        )

        self.icd9_cache[patient_id] = features
        return features

    def extract_medication_features(self, patient_id, prescriptions_df):
        """Extract medication-based features"""
        if patient_id in self.med_cache:
            return self.med_cache[patient_id]

        if len(prescriptions_df) == 0:
            return {}

        # Get all prescriptions for this patient
        patient_rx = prescriptions_df[prescriptions_df['SUBJECT_ID'] == patient_id]

        if len(patient_rx) == 0:
            return {}

        # Extract medication names (lowercase)
        meds = set(patient_rx['DRUG'].astype(str).str.lower().values)

        features = {}

        # Check for each medication category
        for category, keywords in self.medication_keywords.items():
            has_med = any(
                any(keyword in med for keyword in keywords)
                for med in meds
            )
            features[f'med_{category}'] = int(has_med)

        # Total unique medications
        features['med_total_count'] = len(meds)

        # Polypharmacy indicator (>5 meds)
        features['med_polypharmacy'] = int(len(meds) > 5)

        self.med_cache[patient_id] = features
        return features

    def extract_patient_features(self, patient_id, patients_df, metadata):
        """Extract demographic features"""
        features = {}

        # From metadata (already have these)
        features['age'] = metadata.get('age', 65)
        features['gender'] = metadata.get('gender', 0)  # 1=male, 0=female

        # Age categories (CHA2DS2-VASc style)
        age = features['age']
        features['age_65_74'] = int(65 <= age < 75)
        features['age_75_plus'] = int(age >= 75)

        # Gender risk (females have slightly higher stroke risk in some contexts)
        features['female'] = 1 - features['gender']

        return features

    def calculate_cha2ds2vasc(self, features):
        """Calculate CHA2DS2-VASc score"""
        score = 0

        # Age
        if features.get('age_75_plus', 0):
            score += 2
        elif features.get('age_65_74', 0):
            score += 1

        # Female
        if features.get('female', 0):
            score += 1

        # Heart failure
        if features.get('icd9_heart_failure', 0):
            score += 1

        # Hypertension
        if features.get('icd9_hypertension', 0):
            score += 1

        # Diabetes
        if features.get('icd9_diabetes', 0):
            score += 1

        # Stroke/TIA
        if features.get('icd9_prior_stroke', 0) or features.get('icd9_tia', 0):
            score += 2

        # Vascular disease (MI, CAD)
        if features.get('icd9_mi', 0) or features.get('icd9_cad', 0):
            score += 1

        return score


class ClinicalStrokePrediction:
    """Comprehensive stroke prediction with clinical data"""

    def __init__(self, output_dir='production_medium/clinical_stroke_prediction'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_importance = {}

    def load_embeddings_and_clinical_data(self, data_path, metadata_path, checkpoint_path, config_path, mimic_path):
        """Load embeddings and extract clinical features"""
        print("\n[1/4] Loading embeddings from contrastive model...")

        # Load model
        with open(config_path, 'r') as f:
            config = json.load(f)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        input_channels = state_dict['input_conv.weight'].shape[1]

        model = WaveformEncoder(
            input_channels=input_channels,
            hidden_dims=config['hidden_dims'],
            embedding_dim=config['embedding_dim'],
            dropout=0.1
        )
        model.load_state_dict(state_dict)
        model.eval()

        # Load data
        with h5py.File(data_path, 'r') as h5f:
            segments = h5f['segments'][:]

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        print(f"  Loaded {len(segments):,} segments from {len(set(m.get('patient_id') for m in metadata))} patients")

        # Extract embeddings
        print("  Extracting embeddings...")
        segments_tensor = torch.FloatTensor(segments).transpose(1, 2)  # (batch, channels, seq_len)

        all_embeddings = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(segments_tensor), batch_size):
                batch = segments_tensor[i:i+batch_size]
                emb = model(batch)
                emb = F.normalize(emb, dim=1)
                all_embeddings.append(emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        print(f"  Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

        # Extract clinical features
        print("\n[2/4] Extracting clinical features from MIMIC...")
        extractor = ClinicalFeatureExtractor(mimic_path)

        diagnoses_df = extractor.load_diagnoses()
        prescriptions_df = extractor.load_prescriptions()
        patients_df = extractor.load_patients()

        print("\n  Processing patient-level features...")
        patient_features_list = []

        unique_patients = list(set(m.get('patient_id') for m in metadata))

        for i, patient_id in enumerate(unique_patients):
            if i % 10 == 0:
                print(f"    Processed {i}/{len(unique_patients)} patients...")

            # Get first metadata entry for this patient
            patient_metadata = next(m for m in metadata if m.get('patient_id') == patient_id)

            # Extract features
            features = {'patient_id': patient_id}

            # Demographics
            features.update(extractor.extract_patient_features(patient_id, patients_df, patient_metadata))

            # ICD-9 codes
            features.update(extractor.extract_icd9_features(patient_id, diagnoses_df))

            # Medications
            features.update(extractor.extract_medication_features(patient_id, prescriptions_df))

            # CHA2DS2-VASc score
            features['cha2ds2vasc_score'] = extractor.calculate_cha2ds2vasc(features)

            # Target label
            features['has_stroke'] = patient_metadata.get('has_stroke', 0)
            features['has_arrhythmia'] = patient_metadata.get('has_arrhythmia', 0)

            patient_features_list.append(features)

        clinical_df = pd.DataFrame(patient_features_list)

        print(f"\n  Clinical features extracted:")
        print(f"    Total features: {len(clinical_df.columns)}")
        print(f"    Patients: {len(clinical_df)}")
        print(f"    Stroke cases: {clinical_df['has_stroke'].sum()}")

        # Aggregate embeddings at patient level
        print("\n[3/4] Aggregating embeddings at patient level...")

        embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
        embedding_df['patient_id'] = [m.get('patient_id') for m in metadata]

        # Aggregate by patient (mean and std)
        patient_embeddings = embedding_df.groupby('patient_id').agg(['mean', 'std']).reset_index()
        patient_embeddings.columns = ['_'.join(col).strip('_') for col in patient_embeddings.columns.values]

        # Merge clinical features with embeddings
        print("\n[4/4] Combining clinical features with embeddings...")
        combined_df = clinical_df.merge(patient_embeddings, on='patient_id', how='left')
        combined_df = combined_df.fillna(0)

        print(f"\n  Final dataset:")
        print(f"    Shape: {combined_df.shape}")
        print(f"    Stroke cases: {combined_df['has_stroke'].sum()} ({100*combined_df['has_stroke'].mean():.1f}%)")

        return combined_df

    def train_models(self, features_df):
        """Train supervised stroke prediction models"""
        print("\n" + "="*70)
        print("TRAINING STROKE PREDICTION MODELS")
        print("="*70)

        # Prepare features and target
        exclude_cols = ['patient_id', 'has_stroke', 'has_arrhythmia']
        X = features_df.drop(exclude_cols, axis=1, errors='ignore')
        y = features_df['has_stroke']

        print(f"\nDataset:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {len(X)}")
        print(f"  Stroke cases: {y.sum()} ({100*y.mean():.1f}%)")
        print(f"  No stroke: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")

        if y.sum() < 5:
            print("\n  WARNING: Too few positive samples for reliable training!")
            print("  Results will be preliminary. Need more stroke patients for production model.")
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        feature_names = X.columns.tolist()

        # Train-test split (stratified)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42, stratify=y
            )
        except:
            print("  Cannot stratify with so few samples, using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.25, random_state=42
            )

        print(f"\nTrain/Test split:")
        print(f"  Train: {len(X_train)} samples ({y_train.sum()} stroke)")
        print(f"  Test:  {len(X_test)} samples ({y_test.sum()} stroke)")

        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=0.1,
                penalty='l2',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        }

        # Train and evaluate
        for name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Training {name.upper()}")
            print(f"{'='*70}")

            # Train
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            y_pred_proba_train = model.predict_proba(X_train)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test)[:, 1]

            # Metrics
            try:
                train_auc = roc_auc_score(y_train, y_pred_proba_train)
                test_auc = roc_auc_score(y_test, y_pred_proba_test)
                test_ap = average_precision_score(y_test, y_pred_proba_test)
            except:
                train_auc = test_auc = test_ap = 0.0

            train_f1 = f1_score(y_train, y_pred_train)
            test_f1 = f1_score(y_test, y_pred_test)

            test_recall = recall_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, zero_division=0)

            # Store results
            self.models[name] = model
            self.results[name] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'test_ap': test_ap,
                'test_f1': test_f1,
                'test_recall': test_recall,
                'test_precision': test_precision,
                'y_test': y_test,
                'y_pred': y_pred_test,
                'y_pred_proba': y_pred_proba_test
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df

                print("\nTop 15 Important Features:")
                for idx, row in importance_df.head(15).iterrows():
                    print(f"  {row['feature']:40s} : {row['importance']:.4f}")

            # Print results
            print(f"\nPerformance:")
            print(f"  Train AUC:      {train_auc:.3f}")
            print(f"  Test AUC:       {test_auc:.3f}")
            print(f"  Test AP:        {test_ap:.3f}")
            print(f"  Test F1:        {test_f1:.3f}")
            print(f"  Test Recall:    {test_recall:.3f}")
            print(f"  Test Precision: {test_precision:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            print(f"\nConfusion Matrix:")
            print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
            print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    def visualize_results(self):
        """Create visualizations"""
        print("\nCreating visualizations...")

        if not self.results:
            print("  No results to visualize")
            return

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curves
        ax = fig.add_subplot(gs[0, 0])
        for name, results in self.results.items():
            if results['test_auc'] > 0:
                fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
                ax.plot(fpr, tpr, label=f"{name} (AUC: {results['test_auc']:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Precision-Recall Curves
        ax = fig.add_subplot(gs[0, 1])
        for name, results in self.results.items():
            if results['test_ap'] > 0:
                precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
                ax.plot(recall, precision, label=f"{name} (AP: {results['test_ap']:.3f})", linewidth=2)

        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Model Comparison
        ax = fig.add_subplot(gs[0, 2])
        model_names = list(self.results.keys())
        metrics = ['test_auc', 'test_ap', 'test_f1']
        metric_labels = ['AUC', 'AP', 'F1']

        x = np.arange(len(model_names))
        width = 0.25

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.results[name][metric] for name in model_names]
            ax.bar(x + i*width, values, width, label=label, alpha=0.8)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([n.replace('_', ' ').title() for n in model_names], rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Feature Importance (Random Forest)
        if 'random_forest' in self.feature_importance:
            ax = fig.add_subplot(gs[1, :])
            top_features = self.feature_importance['random_forest'].head(20)
            ax.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=9)
            ax.set_xlabel('Feature Importance', fontsize=11)
            ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        # Confusion Matrices
        for i, name in enumerate(list(self.results.keys())):
            ax = fig.add_subplot(gs[2, i])
            cm = confusion_matrix(self.results[name]['y_test'], self.results[name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_title(f'{name.replace("_", " ").title()}', fontsize=11, fontweight='bold')

        plt.suptitle('Clinical Stroke Risk Prediction Results', fontsize=16, fontweight='bold', y=0.995)

        output_path = os.path.join(self.output_dir, 'clinical_stroke_prediction_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

    def save_results(self):
        """Save results"""
        print("\nSaving results...")

        # Performance metrics
        performance = {}
        for name, results in self.results.items():
            performance[name] = {
                'train_auc': float(results['train_auc']),
                'test_auc': float(results['test_auc']),
                'test_ap': float(results['test_ap']),
                'test_f1': float(results['test_f1']),
                'test_recall': float(results['test_recall']),
                'test_precision': float(results['test_precision'])
            }

        with open(os.path.join(self.output_dir, 'model_performance.json'), 'w') as f:
            json.dump(performance, f, indent=2)

        # Feature importance
        for name, importance_df in self.feature_importance.items():
            importance_df.to_csv(
                os.path.join(self.output_dir, f'{name}_feature_importance.csv'),
                index=False
            )

        print(f"  Results saved to: {self.output_dir}")


def main():
    print("="*70)
    print("CLINICAL STROKE RISK PREDICTION")
    print("Using Full MIMIC Clinical Data + Learned Embeddings")
    print("="*70)

    # Paths
    data_path = 'production_medium/full_dataset.h5'
    metadata_path = 'production_medium/full_dataset_metadata.pkl'
    checkpoint_path = 'production_medium_corrected/latest_checkpoint.pth'
    config_path = 'production_medium/config.json'
    mimic_path = 'dataset/extracted'

    # Check corrected model
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'production_medium/checkpoint_epoch_final.pth'
        print(" Using old checkpoint")

    # Initialize predictor
    predictor = ClinicalStrokePrediction()

    # Load data and extract features
    combined_df = predictor.load_embeddings_and_clinical_data(
        data_path, metadata_path, checkpoint_path, config_path, mimic_path
    )

    # Save combined features
    combined_df.to_csv(os.path.join(predictor.output_dir, 'combined_features.csv'), index=False)
    print(f"\nSaved combined features to: {predictor.output_dir}/combined_features.csv")

    # Train models
    predictor.train_models(combined_df)

    # Visualize and save
    predictor.visualize_results()
    predictor.save_results()

    print("\n" + "="*70)
    print("CLINICAL STROKE PREDICTION COMPLETE!")
    print("="*70)
    print(f"Output directory: {predictor.output_dir}")
    print("\nThis model combines:")
    print("  - Learned ECG/PPG embeddings (frozen)")
    print("  - ICD-9 diagnosis codes")
    print("  - Medication history")
    print("  - Demographics")
    print("  - CHA2DS2-VASc score")
    print("="*70)


if __name__ == "__main__":
    main()
