#!/usr/bin/env python3
"""
ECG/PPG ARRHYTHMIA DISCOVERY - STREAMLIT DASHBOARD
===================================================
Interactive visualization and analysis dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
import h5py
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import model
from contrastive_model import WaveformEncoder

# Page config
st.set_page_config(
    page_title="ECG/PPG Arrhythmia Discovery",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-weight: bold;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data(base_dir='production_medium'):
    """Load all data with caching"""
    base_path = Path(base_dir)

    # Load config
    with open(base_path / 'config.json', 'r') as f:
        config = json.load(f)

    # Load metadata
    with open(base_path / 'full_dataset_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # Load dataset info (without loading all segments)
    with h5py.File(base_path / 'full_dataset.h5', 'r') as h5f:
        dataset_shape = h5f['segments'].shape
        dataset_size_gb = h5f['segments'].nbytes / (1024**3)

    # Load checkpoint info
    checkpoint_path = base_path / 'checkpoint_epoch_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    input_channels = state_dict['input_conv.weight'].shape[1]

    return {
        'config': config,
        'metadata': metadata,
        'dataset_shape': dataset_shape,
        'dataset_size_gb': dataset_size_gb,
        'input_channels': input_channels,
        'checkpoint': checkpoint,
        'base_path': base_path
    }


def main():
    # Header
    st.markdown('<p class="big-font">🫀 ECG/PPG Arrhythmia Discovery</p>', unsafe_allow_html=True)
    st.markdown("**Self-Supervised Pattern Discovery for Cardiovascular Risk Prediction**")
    st.markdown("---")

    # Load data
    try:
        data = load_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Make sure you're running from the project directory with production_medium folder")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Overview", "🔬 Dataset Explorer", "🧠 Model Analysis", "🎯 Pattern Discovery", "📈 Clinical Insights"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Info")
    st.sidebar.metric("Total Segments", f"{data['dataset_shape'][0]:,}")
    st.sidebar.metric("Unique Patients", len(set(m.get('patient_id') for m in data['metadata'])))
    st.sidebar.metric("Dataset Size", f"{data['dataset_size_gb']:.2f} GB")

    # Page routing
    if page == "📊 Overview":
        show_overview(data)
    elif page == "🔬 Dataset Explorer":
        show_dataset_explorer(data)
    elif page == "🧠 Model Analysis":
        show_model_analysis(data)
    elif page == "🎯 Pattern Discovery":
        show_pattern_discovery(data)
    elif page == "📈 Clinical Insights":
        show_clinical_insights(data)


def show_overview(data):
    """Overview page"""
    st.header("📊 Project Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Segments",
            value=f"{data['dataset_shape'][0]:,}",
            delta=f"{data['dataset_shape'][0] - 66432:,} vs old"
        )

    with col2:
        stroke_count = sum(1 for m in data['metadata'] if m.get('has_stroke', 0) == 1)
        st.metric(
            label="Stroke Cases",
            value=f"{stroke_count:,}",
            delta=f"{100*stroke_count/len(data['metadata']):.1f}%"
        )

    with col3:
        arrhy_count = sum(1 for m in data['metadata'] if m.get('has_arrhythmia', 0) == 1)
        st.metric(
            label="Arrhythmia Cases",
            value=f"{arrhy_count:,}",
            delta=f"{100*arrhy_count/len(data['metadata']):.1f}%"
        )

    with col4:
        unique_patients = len(set(m.get('patient_id') for m in data['metadata']))
        st.metric(
            label="Unique Patients",
            value=unique_patients
        )

    st.markdown("---")

    # Architecture diagram
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Model Architecture")
        st.json({
            "Input Channels": data['input_channels'],
            "Hidden Dimensions": data['config']['hidden_dims'],
            "Embedding Dimension": data['config']['embedding_dim'],
            "Total Parameters": f"{sum(p.numel() for p in WaveformEncoder(data['input_channels'], data['config']['hidden_dims'], data['config']['embedding_dim']).parameters()):,}"
        })

    with col2:
        st.subheader("⚙️ Training Configuration")
        st.json({
            "Batch Size": data['config']['batch_size'],
            "Learning Rate": data['config']['learning_rate'],
            "Epochs Trained": data['checkpoint'].get('epoch', 'N/A'),
            "Temperature": data['config']['temperature'],
            "Segment Length": f"{data['config']['segment_length_sec']}s",
            "Sampling Rate": f"{data['config']['sampling_rate']} Hz"
        })

    # Pipeline flowchart
    st.markdown("---")
    st.subheader("🔄 Pipeline Flow")

    st.markdown("""
    ```
    ┌─────────────────┐
    │ MIMIC-III Data  │
    │ (ECG + PPG)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Preprocessing   │
    │ • Filter        │
    │ • Normalize     │
    │ • Segment       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Contrastive     │
    │ Learning        │
    │ (SimCLR)        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Embedding       │
    │ Extraction      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Pattern         │
    │ Discovery       │
    │ (Clustering)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Clinical        │
    │ Analysis        │
    └─────────────────┘
    ```
    """)


def show_dataset_explorer(data):
    """Dataset explorer page"""
    st.header("🔬 Dataset Explorer")

    # Clinical statistics
    st.subheader("🏥 Clinical Statistics")

    # Age distribution
    ages = [m.get('age', 65) for m in data['metadata']]
    valid_ages = [a for a in ages if 0 < a < 120]

    fig = px.histogram(
        x=valid_ages,
        nbins=30,
        title="Age Distribution",
        labels={'x': 'Age', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Gender and diagnosis
    col1, col2 = st.columns(2)

    with col1:
        males = sum(1 for m in data['metadata'] if m.get('gender', 0) == 1)
        females = len(data['metadata']) - males

        fig = go.Figure(data=[go.Pie(
            labels=['Male', 'Female'],
            values=[males, females],
            hole=0.3
        )])
        fig.update_layout(title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stroke = sum(1 for m in data['metadata'] if m.get('has_stroke', 0) == 1)
        arrhy = sum(1 for m in data['metadata'] if m.get('has_arrhythmia', 0) == 1)

        fig = go.Figure(data=[go.Bar(
            x=['Stroke', 'Arrhythmia'],
            y=[stroke, arrhy],
            marker_color=['#ff7f0e', '#2ca02c']
        )])
        fig.update_layout(title="Diagnosis Counts", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Sample waveform viewer
    st.markdown("---")
    st.subheader("📊 Sample Waveform Viewer")

    sample_idx = st.slider("Select segment index", 0, min(1000, data['dataset_shape'][0]-1), 0)

    if st.button("Load and Display Waveform"):
        with h5py.File(data['base_path'] / 'full_dataset.h5', 'r') as h5f:
            segment = h5f['segments'][sample_idx]

        # Create subplots for each channel
        n_channels = segment.shape[1]
        fig = make_subplots(
            rows=n_channels,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Channel {i+1}" for i in range(n_channels)]
        )

        for i in range(n_channels):
            fig.add_trace(
                go.Scatter(y=segment[:, i], mode='lines', name=f'Ch {i+1}'),
                row=i+1,
                col=1
            )

        fig.update_layout(height=200*n_channels, showlegend=False, title="ECG/PPG Waveform")
        st.plotly_chart(fig, use_container_width=True)

        # Show metadata
        st.json(data['metadata'][sample_idx])


def show_model_analysis(data):
    """Model analysis page"""
    st.header("🧠 Model Analysis")

    st.subheader("📐 Architecture Details")

    # Architecture table
    arch_data = {
        "Layer": ["Input Conv", "ResBlock 1", "ResBlock 2", "ResBlock 3", "ResBlock 4", "Global Pool", "Projection"],
        "Input": [
            f"({data['input_channels']}, 1250)",
            f"({data['config']['hidden_dims'][0]}, 625)",
            f"({data['config']['hidden_dims'][1]}, 312)",
            f"({data['config']['hidden_dims'][2]}, 156)",
            f"({data['config']['hidden_dims'][3]}, 78)",
            f"({data['config']['hidden_dims'][3]}, 78)",
            f"({data['config']['hidden_dims'][3]},)"
        ],
        "Output": [
            f"({data['config']['hidden_dims'][0]}, 625)",
            f"({data['config']['hidden_dims'][1]}, 312)",
            f"({data['config']['hidden_dims'][2]}, 156)",
            f"({data['config']['hidden_dims'][3]}, 78)",
            f"({data['config']['hidden_dims'][3]}, 39)",
            f"({data['config']['hidden_dims'][3]},)",
            f"({data['config']['embedding_dim']},)"
        ]
    }

    st.dataframe(pd.DataFrame(arch_data), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Training Information")

    if 'epoch' in data['checkpoint']:
        st.info(f"Model trained for **{data['checkpoint']['epoch']} epochs**")
    else:
        st.warning("Training epoch information not available in checkpoint")

    # Parameter count
    model = WaveformEncoder(
        data['input_channels'],
        data['config']['hidden_dims'],
        data['config']['embedding_dim']
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    col1, col2 = st.columns(2)
    col1.metric("Total Parameters", f"{total_params:,}")
    col2.metric("Trainable Parameters", f"{trainable_params:,}")


def show_pattern_discovery(data):
    """Pattern discovery page"""
    st.header("🎯 Pattern Discovery")

    st.info("Pattern discovery results will be loaded here after running `fixed_simple_clustering.py`")

    # Check if clustering results exist
    clustering_results_path = data['base_path'] / 'simple_pattern_discovery' / 'clustering_results.json'

    if clustering_results_path.exists():
        with open(clustering_results_path, 'r') as f:
            results = json.load(f)

        st.success(f"✅ Found {results['n_clusters']} patterns!")

        # Show cluster distribution
        cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()

        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Count'},
            title="Cluster Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Load and show visualization
        viz_path = data['base_path'] / 'simple_pattern_discovery' / 'pattern_visualization.png'
        if viz_path.exists():
            st.image(str(viz_path), caption="Pattern Visualization (UMAP)", use_column_width=True)

    else:
        st.warning("⚠️ No clustering results found. Run `python -X utf8 fixed_simple_clustering.py` first.")


def show_clinical_insights(data):
    """Clinical insights page"""
    st.header("📈 Clinical Insights")

    # Risk analysis
    st.subheader("⚕️ Risk Factor Analysis")

    # Create dataframe from metadata
    df = pd.DataFrame(data['metadata'])

    # Stroke rate by age group
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 80, 120], labels=['<40', '40-60', '60-80', '80+'])

    stroke_by_age = df.groupby('age_group')['has_stroke'].agg(['sum', 'count', 'mean'])
    stroke_by_age.columns = ['Stroke Cases', 'Total', 'Stroke Rate']

    fig = px.bar(
        stroke_by_age,
        y='Stroke Rate',
        title="Stroke Rate by Age Group",
        labels={'value': 'Stroke Rate', 'age_group': 'Age Group'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔗 Risk Correlations")

    corr_data = df[['age', 'gender', 'has_stroke', 'has_arrhythmia', 'mortality']].corr()

    fig = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        x=corr_data.columns,
        y=corr_data.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(title="Clinical Variable Correlations")
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics table
    st.subheader("📊 Summary Statistics")

    summary = {
        "Metric": ["Total Patients", "Avg Age", "Stroke Rate", "Arrhythmia Rate", "Mortality Rate"],
        "Value": [
            len(set(df['patient_id'])),
            f"{df['age'].mean():.1f} years",
            f"{100*df['has_stroke'].mean():.1f}%",
            f"{100*df['has_arrhythmia'].mean():.1f}%",
            f"{100*df['mortality'].mean():.1f}%"
        ]
    }

    st.table(pd.DataFrame(summary))


if __name__ == "__main__":
    main()
