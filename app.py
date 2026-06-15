import streamlit as st
import pandas as pd
import numpy as np
from data import DataProcessor
from model import MLModel, calculate_feature_importance_drift
from viz import (
    plot_feature_importance_comparison,
    plot_prediction_confidence_heatmap,
    plot_error_clustering,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_shap_summary,
    plot_shap_waterfall
)
import pickle
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(
    page_title="ML Model Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    
    if 'train_results' not in st.session_state:
        st.session_state.train_results = None
    if 'val_results' not in st.session_state:
        st.session_state.val_results = None
    
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    if 'shap_explainer' not in st.session_state:
        st.session_state.shap_explainer = None


def render_sidebar():
    """Render sidebar with data upload and model configuration."""
    with st.sidebar:
        st.markdown("## ML Model Debugger")
        st.markdown("---")
        
        st.markdown("### Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload a CSV file containing your dataset"
        )
        
        if uploaded_file is not None:
            try:
                df = st.session_state.data_processor.load_and_validate(uploaded_file)
                st.session_state.df = df
                st.success(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
                
                with st.expander("Dataset Info"):
                    st.write(f"**Shape:** {df.shape}")
                    st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
                    missing = df.isnull().sum().sum()
                    st.write(f"**Missing values:** {missing}")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        
        if st.session_state.df is not None:
            st.markdown("---")
            st.markdown("###  Configuration")
            
            col_info = st.session_state.data_processor.get_column_info(st.session_state.df)
            
            target_col = st.selectbox(
                "Target Column",
                options=col_info['all_columns'],
                index=col_info['all_columns'].index(col_info['suggested_target'])
                if col_info['suggested_target'] in col_info['all_columns'] else 0,
                help="Select the column you want to predict"
            )
            
            available_features = [col for col in col_info['all_columns'] if col != target_col]
            selected_features = st.multiselect(
                "Feature Columns",
                options=available_features,
                default=available_features,
                help="Select features to use for training"
            )
            
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                options=['drop', 'mean', 'median'],
                help="How to handle missing values"
            )
            
            test_size = st.slider(
                "Validation Split",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Proportion of data for validation"
            )
            
            st.markdown("---")
            st.markdown("### 🤖 Model Selection")
            
            model_type = st.selectbox(
                "Model Type",
                options=['logistic_regression', 'random_forest', 'xgboost'],
                format_func=lambda x: {
                    'logistic_regression': 'Logistic Regression',
                    'random_forest': 'Random Forest',
                    'xgboost': 'XGBoost'
                }[x],
                help="Select the model to train"
            )
            
            hyperparameters = {}
            
            if model_type == 'logistic_regression':
                hyperparameters['C'] = st.slider(
                    "Regularization (C)",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Inverse of regularization strength"
                )
                hyperparameters['max_iter'] = st.number_input(
                    "Max Iterations",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100
                )
            
            elif model_type == 'random_forest':
                hyperparameters['n_estimators'] = st.slider(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                hyperparameters['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=1
                )
                hyperparameters['min_samples_split'] = st.slider(
                    "Min Samples Split",
                    min_value=2,
                    max_value=20,
                    value=2,
                    step=1
                )
            
            elif model_type == 'xgboost':
                hyperparameters['n_estimators'] = st.slider(
                    "Number of Estimators",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                hyperparameters['max_depth'] = st.slider(
                    "Max Depth",
                    min_value=2,
                    max_value=20,
                    value=6,
                    step=1
                )
                hyperparameters['learning_rate'] = st.slider(
                    "Learning Rate",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01
                )
            
            st.markdown("---")
            
            with st.form("train_form"):
                train_button = st.form_submit_button(
                    "🚀 Train Model",
                    use_container_width=True,
                    type="primary"
                )
                
                if train_button:
                    train_model(
                        target_col=target_col,
                        selected_features=selected_features,
                        missing_strategy=missing_strategy,
                        test_size=test_size,
                        model_type=model_type,
                        hyperparameters=hyperparameters
                    )
            
            if st.button("🔄 Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'data_processor':
                        del st.session_state[key]
                st.rerun()


def train_model(
    target_col: str,
    selected_features: list,
    missing_strategy: str,
    test_size: float,
    model_type: str,
    hyperparameters: dict
):
    """Train the ML model with given configuration."""
    try:
        with st.spinner("🔄 Training model..."):
            df_clean = st.session_state.data_processor.handle_missing_values(
                st.session_state.df,
                strategy=missing_strategy
            )
            
            X, y = st.session_state.data_processor.prepare_features(
                df_clean,
                target_column=target_col,
                feature_columns=selected_features
            )
            
            X_train, X_val, y_train, y_val = st.session_state.data_processor.split_data(
                X, y, test_size=test_size
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            
            model = MLModel(model_type=model_type, hyperparameters=hyperparameters)
            train_info = model.train(X_train, y_train)
            
            val_results = model.evaluate(X_val, y_val)
            
            st.session_state.model = model
            st.session_state.model_trained = True
            st.session_state.train_results = train_info['train_metrics']
            st.session_state.val_results = val_results
            
            st.session_state.shap_values = None
            st.session_state.shap_explainer = None
            
            st.success(" Model trained successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during training: {str(e)}")


def render_main_content():
    """Render main content area with results and diagnostics."""

    st.markdown('<h1 class="main-header"> Real-Time ML Model Debugger</h1>', unsafe_allow_html=True)
    st.markdown("Train machine learning models and explore advanced diagnostics beyond accuracy")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.info("👈 Upload a dataset and train a model to see diagnostics")
        

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Features")
            st.markdown("""
            - Multiple model types
            - Feature importance analysis
            - Prediction confidence
            - Error clustering
            - SHAP explanations
            """)
        with col2:
            st.markdown("### Diagnostics")
            st.markdown("""
            - Feature drift detection
            - Confidence heatmaps
            - Misclassification patterns
            - Confusion matrices
            - Performance metrics
            """)
        with col3:
            st.markdown("### Export")
            st.markdown("""
            - Save trained models
            - Export diagnostics
            - Download predictions
            - Export errors
            """)
        return
    
    model = st.session_state.model
    train_metrics = st.session_state.train_results
    val_metrics = st.session_state.val_results
    
    st.markdown("## Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Validation Accuracy",
            f"{val_metrics['accuracy']:.3f}",
            delta=f"{val_metrics['accuracy'] - train_metrics['accuracy']:.3f}"
        )
    with col2:
        st.metric(
            "Validation Precision",
            f"{val_metrics['precision']:.3f}",
            delta=f"{val_metrics['precision'] - train_metrics['precision']:.3f}"
        )
    with col3:
        st.metric(
            "Validation Recall",
            f"{val_metrics['recall']:.3f}",
            delta=f"{val_metrics['recall'] - train_metrics['recall']:.3f}"
        )
    with col4:
        st.metric(
            "Validation F1",
            f"{val_metrics['f1']:.3f}",
            delta=f"{val_metrics['f1'] - train_metrics['f1']:.3f}"
        )
    

    with st.expander("Detailed Metrics Comparison"):
        fig_metrics = plot_metrics_comparison(train_metrics, val_metrics)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Metrics**")
            st.json({k: f"{v:.4f}" if isinstance(v, float) else str(v) 
                    for k, v in train_metrics.items() if k != 'confusion_matrix'})
        with col2:
            st.markdown("**Validation Metrics**")
            st.json({k: f"{v:.4f}" if isinstance(v, float) else str(v) 
                    for k, v in val_metrics.items() 
                    if k not in ['confusion_matrix', 'misclassified_indices']})
    
    st.markdown("---")
    

    st.markdown("## Advanced Diagnostics")
    
    tabs = st.tabs([
        "Feature Importance Drift",
        "Prediction Confidence",
        "Error Clustering",
        "Confusion Matrix",
        "SHAP Explainability"
    ])
    

    with tabs[0]:
        st.markdown("### Feature Importance: Training vs Validation")
        st.markdown("""
        This analysis compares how important each feature is on the training set versus the validation set.
        **Red bars** indicate features with significant importance drift, which may suggest:
        - Overfitting to specific features
        - Distribution shift between train/validation
        - Features that generalize poorly
        """)
        
        try:
            importance_comparison = calculate_feature_importance_drift(
                model,
                st.session_state.X_train,
                st.session_state.X_val,
                st.session_state.y_train,
                st.session_state.y_val
            )
            
            drift_threshold = st.slider(
                "Drift Threshold",
                min_value=0.05,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Threshold for highlighting high drift features"
            )
            
            fig_drift = plot_feature_importance_comparison(
                importance_comparison,
                top_n=15,
                drift_threshold=drift_threshold
            )
            st.plotly_chart(fig_drift, use_container_width=True)
            

            with st.expander("Detailed Drift Analysis"):
                st.dataframe(
                    importance_comparison.head(20).style.background_gradient(
                        subset=['drift'], cmap='Reds'
                    ),
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error calculating feature importance drift: {str(e)}")
    

    with tabs[1]:
        st.markdown("### Prediction Confidence Heatmap")
        st.markdown("""
        This heatmap shows the distribution of prediction confidence across different true labels.
        - **Low confidence** predictions (left side) indicate uncertainty
        - **High confidence** errors are particularly problematic
        - Look for patterns where specific classes have systematically low confidence
        """)
        
        try:
            confidence_df = model.get_prediction_confidence(
                st.session_state.X_val,
                st.session_state.y_val
            )
            
            class_names = st.session_state.data_processor.get_class_names()
            fig_confidence = plot_prediction_confidence_heatmap(confidence_df, class_names)
            st.plotly_chart(fig_confidence, use_container_width=True)
            

            col1, col2, col3 = st.columns(3)
            with col1:
                low_conf = (confidence_df['confidence'] < 0.6).sum()
                st.metric("Low Confidence Predictions", low_conf)
            with col2:
                high_conf_errors = ((confidence_df['confidence'] > 0.8) & 
                                   (~confidence_df['correct'])).sum()
                st.metric("High Confidence Errors", high_conf_errors)
            with col3:
                avg_conf_correct = confidence_df[confidence_df['correct']]['confidence'].mean()
                st.metric("Avg Confidence (Correct)", f"{avg_conf_correct:.3f}")
                
        except Exception as e:
            st.error(f"Error creating confidence heatmap: {str(e)}")
    

    with tabs[2]:
        st.markdown("### Error Clustering Analysis")
        st.markdown("""
        Visualizes misclassified samples in 2D space to identify error patterns.
        - **Clusters** of errors suggest systematic problems
        - **Scattered** errors indicate random noise
        - Colors show which classes are confused with each other
        """)
        
        try:
            X_mis, y_true_mis, y_pred_mis = model.get_misclassified_samples(
                st.session_state.X_val,
                st.session_state.y_val
            )
            
            if len(X_mis) > 0:
                col1, col2 = st.columns([3, 1])
                with col2:
                    method = st.radio(
                        "Dimensionality Reduction",
                        options=['pca', 'tsne'],
                        format_func=lambda x: x.upper()
                    )
                
                class_names = st.session_state.data_processor.get_class_names()
                fig_clustering = plot_error_clustering(
                    X_mis, y_true_mis, y_pred_mis,
                    method=method,
                    class_names=class_names
                )
                st.plotly_chart(fig_clustering, use_container_width=True)
                

                st.markdown("#### Error Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Misclassified", len(X_mis))
                    st.metric("Error Rate", f"{len(X_mis) / len(st.session_state.y_val):.2%}")
                with col2:

                    if len(class_names) > 0:
                        error_types = pd.DataFrame({
                            'true': [class_names[i] for i in y_true_mis],
                            'pred': [class_names[i] for i in y_pred_mis]
                        })
                        most_common = error_types.value_counts().head(1)
                        if len(most_common) > 0:
                            st.markdown("**Most Common Error:**")
                            st.write(f"{most_common.index[0][0]} → {most_common.index[0][1]} ({most_common.values[0]} times)")
                

                if st.button("📥 Export Misclassified Samples"):
                    mis_df = X_mis.copy()
                    mis_df['true_label'] = y_true_mis.values
                    mis_df['predicted_label'] = y_pred_mis
                    csv = mis_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"misclassified_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            else:
                st.success(" No misclassified samples! Perfect predictions!")
                
        except Exception as e:
            st.error(f"Error creating clustering visualization: {str(e)}")
    

    with tabs[3]:
        st.markdown("### Confusion Matrix")
        st.markdown("""
        Shows the detailed breakdown of predictions vs actual labels.
        - Diagonal elements = correct predictions
        - Off-diagonal = misclassifications
        - Normalized values show proportions
        """)
        
        try:
            class_names = st.session_state.data_processor.get_class_names()
            fig_cm = plot_confusion_matrix(val_metrics['confusion_matrix'], class_names)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            with st.expander("Raw Confusion Matrix"):
                cm_df = pd.DataFrame(
                    val_metrics['confusion_matrix'],
                    index=[f"True: {c}" for c in class_names] if class_names else None,
                    columns=[f"Pred: {c}" for c in class_names] if class_names else None
                )
                st.dataframe(cm_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating confusion matrix: {str(e)}")
    

    with tabs[4]:
        st.markdown("### SHAP (SHapley Additive exPlanations)")
        st.markdown("""
        SHAP values explain individual predictions by showing how much each feature contributed.
        - **Global explanations** show overall feature importance
        - **Local explanations** show why a specific prediction was made
        """)
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
        else:
            try:
                if st.session_state.shap_values is None:
                    with st.spinner("Computing SHAP values... This may take a moment."):
                        sample_size = min(100, len(st.session_state.X_val))
                        X_sample = st.session_state.X_val.sample(n=sample_size, random_state=42)
                        
                        if model.model_type in ['random_forest', 'xgboost']:
                            explainer = shap.TreeExplainer(model.model)
                        else:
                            explainer = shap.LinearExplainer(
                                model.model,
                                st.session_state.X_train.sample(n=min(100, len(st.session_state.X_train)))
                            )
                        
                        shap_values = explainer.shap_values(X_sample)
                        
                        st.session_state.shap_values = shap_values
                        st.session_state.shap_explainer = explainer
                        st.session_state.X_shap_sample = X_sample
                
                st.markdown("#### Global Feature Importance")
                try:
                    class_names = st.session_state.data_processor.get_class_names()
                    n_classes = len(class_names) if class_names else model.n_classes
                    
                    if n_classes > 2:
                        class_idx = st.selectbox(
                            "Select Class",
                            range(n_classes),
                            format_func=lambda x: class_names[x] if class_names else f"Class {x}"
                        )
                    else:
                        class_idx = 0
                    
                    fig_shap_summary = plot_shap_summary(
                        st.session_state.shap_values,
                        st.session_state.X_shap_sample,
                        class_idx=class_idx
                    )
                    st.plotly_chart(fig_shap_summary, use_container_width=True)
                except Exception as e:
                    st.error(f"Error plotting SHAP summary: {str(e)}")
                
                st.markdown("---")
                st.markdown("#### Local Explanation (Single Prediction)")
                
                sample_idx = st.number_input(
                    "Select Sample Index",
                    min_value=0,
                    max_value=len(st.session_state.X_shap_sample) - 1,
                    value=0,
                    help="Choose a sample to explain"
                )
                
                try:
                    if hasattr(st.session_state.shap_explainer, 'expected_value'):
                        base_value = st.session_state.shap_explainer.expected_value
                        if isinstance(base_value, np.ndarray):
                            base_value = base_value[class_idx] if n_classes > 2 else base_value[0]
                    else:
                        base_value = 0.0
                    
                    fig_shap_waterfall = plot_shap_waterfall(
                        st.session_state.shap_values,
                        st.session_state.X_shap_sample,
                        sample_idx=sample_idx,
                        base_value=base_value,
                        class_idx=class_idx if n_classes > 2 else 0
                    )
                    st.plotly_chart(fig_shap_waterfall, use_container_width=True)
                    
                    with st.expander("View Sample Features"):
                        sample_features = st.session_state.X_shap_sample.iloc[sample_idx]
                        st.dataframe(
                            pd.DataFrame({
                                'Feature': sample_features.index,
                                'Value': sample_features.values
                            }),
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"Error plotting local SHAP explanation: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error computing SHAP values: {str(e)}")
    

    st.markdown("---")
    st.markdown("## Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Save Model", use_container_width=True):
            try:
                model_data = {
                    'model': model.model,
                    'model_type': model.model_type,
                    'feature_names': model.feature_names,
                    'data_processor': st.session_state.data_processor
                }
                model_bytes = pickle.dumps(model_data)
                st.download_button(
                    "Download Model",
                    model_bytes,
                    f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    "application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
    
    with col2:
        if st.button("Export Predictions", use_container_width=True):
            try:
                predictions_df = pd.DataFrame({
                    'true_label': st.session_state.y_val.values,
                    'predicted_label': model.predict(st.session_state.X_val),
                    'correct': st.session_state.y_val.values == model.predict(st.session_state.X_val)
                })
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions CSV",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Error exporting predictions: {str(e)}")
    
    with col3:
        if st.button(" Export Metrics", use_container_width=True):
            try:
                metrics_dict = {
                    'train_metrics': train_metrics,
                    'val_metrics': {k: v for k, v in val_metrics.items() 
                                   if k not in ['confusion_matrix', 'misclassified_indices']},
                    'timestamp': datetime.now().isoformat()
                }
                import json
                metrics_json = json.dumps(metrics_dict, indent=2)
                st.download_button(
                    "Download Metrics JSON",
                    metrics_json,
                    f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            except Exception as e:
                st.error(f"Error exporting metrics: {str(e)}")


def main():
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()