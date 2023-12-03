import streamlit as st

st.set_page_config(page_title="Home")

st.write(
    """
    # Project Overview
    ## Early Detection of Schizophrenia using EEG Data
    This project aims to develop a machine learning model for the early detection of schizophrenia based on Electroencephalogram (EEG) data. Schizophrenia is a complex mental health disorder, and early diagnosis is crucial for effective intervention and treatment. EEG data, which measures electrical activity in the brain, provides a valuable source of information for understanding neurophysiological patterns associated with schizophrenia.

    ### Objectives
    - Utilize advanced data analysis techniques to extract meaningful features from EEG recordings
    - Build a machine learning model capable of detecting early signs of schizophrenia from EEG data.

    ### Pages
    1. **[Data Collection](https://ai-driven-eeg-schizophrenia-diagnosis.streamlit.app/Data_Collection):** Contains EEG data from individuals with and without schizophrenia i.e. patients and control, how the data was collected and various distributions in the data.
    2. **[Data Preprocessing](https://ai-driven-eeg-schizophrenia-diagnosis.streamlit.app/Data_Preprocessing):** Shows processes involved in cleaning, preprocessing and normalising the EEG data.
    3. **[Feature Extraction](https://ai-driven-eeg-schizophrenia-diagnosis.streamlit.app/Feature_Extraction):** Details signal processing techniques and feature engineering steps to extract relevant features.
    4. **[Model Development](https://ai-driven-eeg-schizophrenia-diagnosis.streamlit.app/Machine_Learning):** Logs machine learning models and parameters using a variety of algorithms to identify patterns associated with schizophrenia.

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
