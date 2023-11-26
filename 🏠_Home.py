import streamlit as st

st.set_page_config(page_title="Home")

st.write(
    """
    # Project Overview
    ## Early Detection of Schizophrenia using EEG Data
    This project aims to develop a machine learning model for the early detection of schizophrenia based on Electroencephalogram (EEG) data. Schizophrenia is a complex mental health disorder, and early diagnosis is crucial for effective intervention and treatment. EEG data, which measures electrical activity in the brain, provides a valuable source of information for understanding neurophysiological patterns associated with schizophrenia.

    ### Objectives
    - Build a machine learning model capable of detecting early signs of schizophrenia from EEG data.
    - Utilize advanced data analysis techniques to extract meaningful features from EEG recordings.

    ### Methodology
    1. **Data Collection:** Gather EEG data from individuals with and without schizophrenia.
    2. **Data Preprocessing:** Clean, preprocess, and normalize the EEG data to enhance model performance.
    3. **Feature Extraction:** Utilize signal processing techniques and feature engineering to extract relevant features.
    4. **Model Development:** Train a machine learning model using a variety of algorithms to identify patterns associated with schizophrenia.

    ### Importance
    Early detection of schizophrenia can significantly improve patient outcomes by enabling timely intervention and treatment. This project contributes to the field of mental health by providing a tool that may assist healthcare professionals in the early identification of individuals at risk of developing schizophrenia.

    ### Technologies Used
    - Python
    - Machine Learning (MLflow for model tracking)
    - Streamlit (for the user interface)
    - EEG Data Processing Libraries

    *Disclaimer: This project is for research and educational purposes. If you or someone you know is experiencing mental health concerns, please consult with a qualified healthcare professional.*

    """
)
