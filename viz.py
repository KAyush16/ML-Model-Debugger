"""
Visualization module for ML diagnostics using Plotly.
Creates interactive charts for model analysis.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional


def plot_feature_importance_comparison(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    drift_threshold: float = 0.1
) -> go.Figure:
    """
    Create side-by-side bar charts comparing train vs validation feature importance.
    Highlights features with significant drift.
    
    Args:
        importance_df: DataFrame with importance_train, importance_val, and drift columns
        top_n: Number of top features to display
        drift_threshold: Threshold for highlighting high drift
        
    Returns:
        Plotly figure
    """
    df_plot = importance_df.head(top_n).copy()
    
    df_plot['high_drift'] = df_plot['drift'] > drift_threshold
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Set Importance', 'Validation Set Importance'),
        horizontal_spacing=0.15
    )
    
    colors_train = ['red' if hd else 'steelblue' for hd in df_plot['high_drift']]
    fig.add_trace(
        go.Bar(
            y=df_plot['feature'],
            x=df_plot['importance_train'],
            orientation='h',
            marker_color=colors_train,
            name='Train',
            text=df_plot['importance_train'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    colors_val = ['red' if hd else 'lightcoral' for hd in df_plot['high_drift']]
    fig.add_trace(
        go.Bar(
            y=df_plot['feature'],
            x=df_plot['importance_val'],
            orientation='h',
            marker_color=colors_val,
            name='Validation',
            text=df_plot['importance_val'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Importance Score", row=1, col=1)
    fig.update_xaxes(title_text="Importance Score", row=1, col=2)
    fig.update_yaxes(title_text="Features", row=1, col=1)
    
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    fig.update_layout(
        height=max(400, top_n * 30),
        showlegend=False,
        title_text=f"Feature Importance Drift Analysis (Red = Drift > {drift_threshold})",
        title_x=0.5
    )
    
    return fig


def plot_prediction_confidence_heatmap(
    confidence_df: pd.DataFrame,
    class_names: Optional[list] = None
) -> go.Figure:
    """
    Create heatmap showing prediction confidence vs true labels.
    Helps identify where model is uncertain.
    
    Args:
        confidence_df: DataFrame with true_label, predicted_label, confidence, correct
        class_names: Optional list of class names for labeling
        
    Returns:
        Plotly figure
    """
    # Create bins for confidence
    confidence_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    confidence_labels = ['0-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    confidence_df = confidence_df.copy()
    confidence_df['confidence_bin'] = pd.cut(
        confidence_df['confidence'],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True
    )
    
    heatmap_data = confidence_df.groupby(['true_label', 'confidence_bin']).size().unstack(fill_value=0)
    
    heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    
    if class_names:
        heatmap_data_normalized.index = [
            class_names[int(i)] if int(i) < len(class_names) else str(i) 
            for i in heatmap_data_normalized.index
        ]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data_normalized.values,
        x=heatmap_data_normalized.columns,
        y=heatmap_data_normalized.index,
        colorscale='RdYlGn',
        text=heatmap_data_normalized.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hovertemplate='True Label: %{y}<br>Confidence: %{x}<br>Percentage: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="% of Samples")
    ))
    
    fig.update_layout(
        title='Prediction Confidence Distribution by True Label',
        xaxis_title='Confidence Range',
        yaxis_title='True Label',
        height=max(400, len(heatmap_data_normalized) * 50),
        title_x=0.5
    )
    
    return fig


def plot_error_clustering(
    X_misclassified: pd.DataFrame,
    y_true_misclassified: pd.Series,
    y_pred_misclassified: np.ndarray,
    method: str = 'pca',
    class_names: Optional[list] = None,
    random_state: int = 42
) -> go.Figure:
    """
    Visualize misclassified samples in 2D space using dimensionality reduction.
    Colors show true vs predicted labels.
    
    Args:
        X_misclassified: Features of misclassified samples
        y_true_misclassified: True labels of misclassified samples
        y_pred_misclassified: Predicted labels of misclassified samples
        method: 'pca' or 'tsne'
        class_names: Optional list of class names
        random_state: Random seed for reproducibility
        
    Returns:
        Plotly figure
    """
    if len(X_misclassified) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No misclassified samples!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=400)
        return fig
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        X_reduced = reducer.fit_transform(X_misclassified)
        explained_var = reducer.explained_variance_ratio_
        method_label = f'PCA (Explained Var: {explained_var.sum():.2%})'
    else:  
        if len(X_misclassified) > 1000:
            sample_idx = np.random.RandomState(random_state).choice(
                len(X_misclassified), 1000, replace=False
            )
            X_sample = X_misclassified.iloc[sample_idx]
            y_true_sample = y_true_misclassified.iloc[sample_idx]
            y_pred_sample = y_pred_misclassified[sample_idx]
        else:
            X_sample = X_misclassified
            y_true_sample = y_true_misclassified
            y_pred_sample = y_pred_misclassified
        
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(X_sample)-1))
        X_reduced = reducer.fit_transform(X_sample)
        y_true_misclassified = y_true_sample
        y_pred_misclassified = y_pred_sample
        method_label = 't-SNE'
    
    plot_df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'true_label': y_true_misclassified.values,
        'pred_label': y_pred_misclassified
    })
    
    if class_names:
        plot_df['true_label_name'] = plot_df['true_label'].apply(
            lambda x: class_names[int(x)] if int(x) < len(class_names) else str(x)
        )
        plot_df['pred_label_name'] = plot_df['pred_label'].apply(
            lambda x: class_names[int(x)] if int(x) < len(class_names) else str(x)
        )
    else:
        plot_df['true_label_name'] = plot_df['true_label'].astype(str)
        plot_df['pred_label_name'] = plot_df['pred_label'].astype(str)
    
    plot_df['error_type'] = (
        'True: ' + plot_df['true_label_name'] + 
        ' → Pred: ' + plot_df['pred_label_name']
    )
    
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        color='error_type',
        title=f'Error Clustering using {method_label}',
        labels={'x': 'Component 1', 'y': 'Component 2'},
        hover_data=['true_label_name', 'pred_label_name']
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        height=600,
        title_x=0.5,
        legend_title_text='Misclassification Type'
    )
    
    return fig


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: Optional[list] = None
) -> go.Figure:
    """
    Create interactive confusion matrix heatmap.
    
    Args:
        confusion_mat: Confusion matrix array
        class_names: Optional list of class names
        
    Returns:
        Plotly figure
    """
    cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    labels = class_names if class_names else [str(i) for i in range(len(confusion_mat))]
    
    annotations = []
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{confusion_mat[i, j]}<br>({cm_normalized[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black", size=10)
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Normalized<br>Count")
    ))
    
    fig.update_layout(
        title='Confusion Matrix (Normalized)',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        annotations=annotations,
        height=max(400, len(labels) * 50),
        title_x=0.5
    )
    
    return fig


def plot_metrics_comparison(
    train_metrics: dict,
    val_metrics: dict
) -> go.Figure:
    """
    Create bar chart comparing train vs validation metrics.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        
    Returns:
        Plotly figure
    """
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    train_values = [train_metrics.get(m, 0) for m in metrics_to_plot]
    val_values = [val_metrics.get(m, 0) for m in metrics_to_plot]
    
    fig = go.Figure(data=[
        go.Bar(name='Training', x=metrics_to_plot, y=train_values, marker_color='steelblue'),
        go.Bar(name='Validation', x=metrics_to_plot, y=val_values, marker_color='lightcoral')
    ])
    
    fig.update_layout(
        title='Model Performance: Train vs Validation',
        xaxis_title='Metric',
        yaxis_title='Score',
        barmode='group',
        height=400,
        yaxis_range=[0, 1],
        title_x=0.5
    )
    
    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    class_idx: int = 0,
    max_display: int = 20
) -> go.Figure:
    """
    Create SHAP summary plot showing global feature importance.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        class_idx: Class index for multiclass (default 0)
        max_display: Maximum number of features to display
        
    Returns:
        Plotly figure
    """
    if len(shap_values.shape) == 3:
        shap_vals = shap_values[:, :, class_idx]
    else:
        shap_vals = shap_values
    

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feature_names = X.columns.tolist()
    

    top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]
    
    plot_df = pd.DataFrame({
        'feature': [feature_names[i] for i in top_indices],
        'mean_abs_shap': mean_abs_shap[top_indices]
    })
    
    fig = go.Figure(data=[
        go.Bar(
            y=plot_df['feature'],
            x=plot_df['mean_abs_shap'],
            orientation='h',
            marker_color='mediumpurple'
        )
    ])
    
    fig.update_layout(
        title=f'SHAP Feature Importance (Class {class_idx})',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        height=max(400, max_display * 25),
        yaxis_autorange='reversed',
        title_x=0.5
    )
    
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    sample_idx: int,
    base_value: float,
    class_idx: int = 0
) -> go.Figure:
    """
    Create SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix
        sample_idx: Index of sample to explain
        base_value: Base value (expected value)
        class_idx: Class index for multiclass
        
    Returns:
        Plotly figure
    """

    if len(shap_values.shape) == 3:
        shap_vals = shap_values[sample_idx, :, class_idx]
    else:
        shap_vals = shap_values[sample_idx]
    
    feature_names = X.columns.tolist()
    feature_values = X.iloc[sample_idx].values
    

    indices = np.argsort(np.abs(shap_vals))[-15:][::-1]
    

    waterfall_data = []
    cumulative = base_value
    
    for idx in indices:
        waterfall_data.append({
            'feature': f"{feature_names[idx]}={feature_values[idx]:.2f}",
            'shap': shap_vals[idx],
            'cumulative': cumulative + shap_vals[idx]
        })
        cumulative += shap_vals[idx]
    
    df_waterfall = pd.DataFrame(waterfall_data)
    

    fig = go.Figure()
    
    colors = ['red' if x < 0 else 'green' for x in df_waterfall['shap']]
    
    fig.add_trace(go.Bar(
        y=df_waterfall['feature'],
        x=df_waterfall['shap'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.3f}" for x in df_waterfall['shap']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'SHAP Explanation for Sample {sample_idx} (Class {class_idx})',
        xaxis_title='SHAP Value (impact on prediction)',
        yaxis_title='Feature',
        height=max(400, len(df_waterfall) * 30),
        yaxis_autorange='reversed',
        title_x=0.5
    )
    
    return fig