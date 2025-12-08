import streamlit as st
import pandas as pd
import os
from src.data import load_data
from src.preprocessing import get_preprocessing_pipeline
from src.training import train_model
from src.utils import generate_run_summary, save_artifacts, push_to_github

st.set_page_config(page_title="ML Automation Prototype", layout="wide")

st.title("ML Automation Prototype (Tabular)")

if "data" not in st.session_state:
    st.session_state.data = None
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

tab1, tab2, tab3, tab4 = st.tabs(["Data Upload", "Preprocessing", "Modeling", "Results & Push"])

with tab1:
    st.header("Upload Tabular Data")
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        try:
            st.session_state.data = load_data(uploaded_file)
            st.success("File loaded successfully!")
            st.write("Preview:", st.session_state.data.head())
            st.write("Shape:", st.session_state.data.shape)
            st.write("Columns:", list(st.session_state.data.columns))
            
            target_col = st.selectbox("Select Target Column (for Supervised Learning)", 
                                      [None] + list(st.session_state.data.columns))
            st.session_state.target_col = target_col
            
            if target_col:
                st.info(f"Target distribution: {st.session_state.data[target_col].value_counts(normalize=True).to_dict() if st.session_state.data[target_col].dtype == 'object' or len(st.session_state.data[target_col].unique()) < 20 else 'Continuous'}")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

with tab2:
    st.header("Preprocessing Configuration")
    if st.session_state.data is not None:
        st.subheader("Settings")
        num_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Simple config for prototype
        drop_cols = st.multiselect("Drop Columns", st.session_state.data.columns)
        
        st.markdown("### Numerical Preprocessing")
        num_impute = st.selectbox("Missing Value Strategy (Num)", ["mean", "median", "constant"])
        scaler = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "None"])
        
        st.markdown("### Categorical Preprocessing")
        cat_impute = st.selectbox("Missing Value Strategy (Cat)", ["most_frequent", "constant"])
        encoder = st.selectbox("Encoding", ["OneHotEncoder", "OrdinalEncoder"])
        
        if st.button("Apply & Preview Preprocessing"):
            st.session_state.prepro_config = {
                'drop': drop_cols,
                'num_impute': num_impute,
                'scaler': scaler,
                'cat_impute': cat_impute,
                'encoder': encoder,
                'num_cols': [c for c in num_cols if c not in drop_cols and c != st.session_state.target_col],
                'cat_cols': [c for c in cat_cols if c not in drop_cols and c != st.session_state.target_col]
            }
            # Placeholder for actual transformation preview
            st.success("Configuration saved! (Preview logic in src/preprocessing.py)")
    else:
        st.warning("Please upload data first.")

with tab3:
    st.header("Model Selection & Training")
    if st.session_state.data is not None:
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression", "Clustering"])
        
        models = []
        if problem_type == "Classification":
            models = st.multiselect("Select Models (Max 2)", ["Logistic Regression", "Random Forest Classifier"], default=["Logistic Regression"])
        elif problem_type == "Regression":
            models = st.multiselect("Select Models (Max 2)", ["Linear Regression", "Random Forest Regressor"], default=["Linear Regression"])
        else:
            models = st.multiselect("Select Models (Max 2)", ["KMeans", "Agglomerative Clustering"], default=["KMeans"])
            
        use_cv = st.checkbox("Use Cross-Validation", value=True)
        tune_hyperparams = st.checkbox("Tune Hyperparameters (GridSearchCV)", value=False)
        
        if st.button("Train Models"):
            if not models:
                st.error("Select at least one model.")
            else:
                with st.spinner("Training..."):
                    results = train_model(st.session_state.data, 
                                          st.session_state.target_col, 
                                          problem_type, 
                                          models, 
                                          st.session_state.get('prepro_config'),
                                          use_cv,
                                          tune_hyperparams)
                    st.session_state.model_results = results
                    st.success("Training Complete!")
                    
    else:
        st.warning("Please upload data first.")

with tab4:
    st.header("Results & Artifacts")
    if st.session_state.model_results:
        st.write(st.session_state.model_results)
        # Visualizations would go here
        
        st.divider()
        st.header("Git Operations")
        repo_name = st.text_input("Repository Name (current dir default)", value="ml-prototype")
        commit_msg = st.text_input("Commit Message", value="Auto-generated ML prototype run")
        
        if st.button("Push Artifacts to GitHub"):
            with st.spinner("Pushing..."):
                link = push_to_github(st.session_state.model_results, repo_name, commit_msg)
                if link:
                    st.success(f"Pushed to GitHub! PR Link: {link}")
                else:
                    st.error("Failed to push. Check logs.")
    else:
        st.info("Train models to see results.")
