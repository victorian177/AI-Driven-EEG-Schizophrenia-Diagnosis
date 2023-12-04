import streamlit as st

st.title("Metrics Selection for Model Evaluation")

st.write("When evaluating models designed to detect schizophrenia using EEG data, choosing the right metrics is crucial to assess performance accurately. The following metrics provide valuable insights into the model's effectiveness in binary classification:")

st.write("1. **Accuracy:** Provides an overall measure of correct classifications. The closer to 1, the better, although it can be misleading in imbalanced datasets.")
st.write("2. **Precision (Positive Predictive Value):** Indicates the accuracy of positive predictions, crucial for minimizing false positives. A higher precision is desirable.")
st.write("3. **Recall (Sensitivity, True Positive Rate):** Captures the ability to correctly identify positive instances, essential for minimizing false negatives. A higher recall is desirable.")
st.write("4. **F1 Score:** Balances precision and recall, particularly useful when there is an uneven class distribution. The closer to 1, the better, indicating a balance between precision and recall.")
st.write("5. **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** Assesses the model's ability to discriminate between positive and negative instances across different probability thresholds. An AUC-ROC value closer to 1 indicates better discrimination ability.")
st.write("6. **Area Under the Precision-Recall Curve (AUC-PR):** Evaluates the model's performance in terms of precision and recall across different probability thresholds. An AUC-PR value closer to 1 indicates better precision and recall trade-off.")

st.write("In the context of schizophrenia detection, it is crucial to minimize false negatives to ensure early detection and intervention. Precision is essential to minimize false positives, as misdiagnosing a healthy individual as having schizophrenia can have serious consequences. Achieving a balance between precision and recall is often necessary, emphasizing the importance of a comprehensive evaluation approach.")
