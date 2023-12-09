# Import necessary libraries
import streamlit as st
import mlflow
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


st.title("Metrics Selection for Model Evaluation")

st.write(
    "When evaluating models designed to detect schizophrenia using EEG data, choosing the right metrics is crucial to assess performance accurately. The following metrics provide valuable insights into the model's effectiveness in binary classification:"
)

st.write(
    "1. **Accuracy:** Provides an overall measure of correct classifications. The closer to 1, the better, although it can be misleading in imbalanced datasets."
)
st.write(
    "2. **Precision (Positive Predictive Value):** Indicates the accuracy of positive predictions, crucial for minimizing false positives. A higher precision is desirable."
)
st.write(
    "3. **Recall (Sensitivity, True Positive Rate):** Captures the ability to correctly identify positive instances, essential for minimizing false negatives. A higher recall is desirable."
)
st.write(
    "4. **F1 Score:** Balances precision and recall, particularly useful when there is an uneven class distribution. The closer to 1, the better, indicating a balance between precision and recall."
)
st.write(
    "5. **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** Assesses the model's ability to discriminate between positive and negative instances across different probability thresholds. An AUC-ROC value closer to 1 indicates better discrimination ability."
)
st.write(
    "6. **Area Under the Precision-Recall Curve (AUC-PR):** Evaluates the model's performance in terms of precision and recall across different probability thresholds. An AUC-PR value closer to 1 indicates better precision and recall trade-off."
)

st.write(
    "In the context of schizophrenia detection, it is crucial to minimize false negatives to ensure early detection and intervention. Precision is essential to minimize false positives, as misdiagnosing a healthy individual as having schizophrenia can have serious consequences. Achieving a balance between precision and recall is often necessary, emphasizing the importance of a comprehensive evaluation approach."
)

# Set MLflow server URI (replace 'http://localhost:5000' with the actual URI of your MLflow server)
mlflow.set_tracking_uri("http://localhost:5000")

# Get a list of runs
runs = mlflow.search_runs()

# Streamlit application
st.title("MLflow Information Viewer")

# Display runs
st.subheader("Runs:")
st.write(runs)

# Allow the user to select a run
selected_run_id = st.selectbox("Select a run:", runs["run_id"])

# Retrieve information for the selected run
if selected_run_id:
    with st.spinner("Fetching run information..."):
        # Get run information
        run_info = mlflow.get_run(selected_run_id)

        # Display run parameters
        st.subheader("Run Parameters:")
        st.write(pd.DataFrame(run_info.data.params, index=[0]))

        # Display run metrics
        st.subheader("Run Metrics:")
        st.write(pd.DataFrame(run_info.data.metrics, index=[0]))

        # Display artifacts (if any)
        st.subheader("Artifacts:")
        artifacts_dir = os.path.join(mlflow.get_artifact_uri(), selected_run_id)
        artifacts = os.listdir(artifacts_dir)
        st.write(artifacts)

        # Display the model (if logged)
        model_path = os.path.join(artifacts_dir, "random_forest_model")
        if os.path.exists(model_path):
            st.subheader("Model:")
            st.write(f"Model saved at: {model_path}")
