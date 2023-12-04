import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


st.title("Metrics Selection for Model Evaluation")

st.write("When evaluating models designed to detect schizophrenia using EEG data, choosing the right metrics is crucial to assess performance accurately. The following metrics provide valuable insights into the model's effectiveness in binary classification:")

st.write("1. **Accuracy:** Provides an overall measure of correct classifications. The closer to 1, the better, although it can be misleading in imbalanced datasets.")
st.write("2. **Precision (Positive Predictive Value):** Indicates the accuracy of positive predictions, crucial for minimizing false positives. A higher precision is desirable.")
st.write("3. **Recall (Sensitivity, True Positive Rate):** Captures the ability to correctly identify positive instances, essential for minimizing false negatives. A higher recall is desirable.")
st.write("4. **F1 Score:** Balances precision and recall, particularly useful when there is an uneven class distribution. The closer to 1, the better, indicating a balance between precision and recall.")
st.write("5. **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** Assesses the model's ability to discriminate between positive and negative instances across different probability thresholds. An AUC-ROC value closer to 1 indicates better discrimination ability.")
st.write("6. **Area Under the Precision-Recall Curve (AUC-PR):** Evaluates the model's performance in terms of precision and recall across different probability thresholds. An AUC-PR value closer to 1 indicates better precision and recall trade-off.")

st.write("In the context of schizophrenia detection, it is crucial to minimize false negatives to ensure early detection and intervention. Precision is essential to minimize false positives, as misdiagnosing a healthy individual as having schizophrenia can have serious consequences. Achieving a balance between precision and recall is often necessary, emphasizing the importance of a comprehensive evaluation approach.")


# Load your own dataset (replace 'your_dataset.csv' with your file path)
dataset_path = 'Frequency Analysis Dataset/Dataset.csv'
df = pd.read_csv(dataset_path)

# Assume the last column is the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
pr_auc = average_precision_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])

# Streamlit application
st.title("Random Forest Classifier Metrics")

# Display evaluation metrics
st.subheader("Evaluation Metrics:")
st.write(f"Accuracy: {accuracy:.4f}")
st.write(f"Precision: {precision:.4f}")
st.write(f"Recall: {recall:.4f}")
st.write(f"F1 Score: {f1:.4f}")
st.write(f"AUC-ROC: {roc_auc:.4f}")
st.write(f"AUC-PR: {pr_auc:.4f}")

# Optionally, you can display other information or visualizations here