import streamlit as st
import pandas as pd
from pyedflib import highlevel

## EEG Features for Diagnosis Section
st.header("EEG Features for Diagnosis")

### Frequency Analysis
st.subheader("Frequency Analysis")
st.write(
    """
The brain's electrical activity is often categorized into distinct frequency bands, including delta, theta, alpha, beta, and gamma. These bands represent different neural oscillations, each associated with specific cognitive and mental states.
"""
)

#### Delta (0.5–4.0 Hz)
st.subheader("Delta (0.5–4.0 Hz)")
st.write(
    """
Delta activity, characterized by a spectral bandwidth of 0.5–4.0 Hz, is primarily associated with slow waves observed during states of unconsciousness, such as sleep and anesthesia. Additionally, delta band synchronization in response to external stimulation plays a role in motivational, emotional, and cognitive functions. In subjects with chronic schizophrenia, resting-state EEG data consistently show increased delta power. However, findings regarding delta band abnormalities during resting state in the early phases of schizophrenia are controversial, with some studies reporting no significant differences, while others observe variations in power, amplitude, and synchronization within this band.
"""
)
# Data for the Frequency Bands, Frequency Ranges, and Brain States
data = {
    "Frequency Band": [
        "Delta (0.5–4.0 Hz)",
        "Theta (4–8 Hz)",
        "Alpha (8–13 Hz)",
        "Beta (12–30 Hz)",
        "Gamma (30–100 Hz)",
    ],
    "Frequency Range": ["0.5–4.0 Hz", "4–8 Hz", "8–13 Hz", "12–30 Hz", "30–100 Hz"],
    "Location": [
        "Deep brain structures, thalamus, brainstem",
        "Limbic system, hippocampus, parahippocampal gyrus",
        "Occipital, parietal, and temporal lobes",
        "Sensorimotor regions, central sulcus",
        "Frontal and temporal lobes",
    ],
    "Brain State/Condition": [
        "During states of unconsciousness, e.g., sleep and anesthesia",
        "Implicated in various cognitive processes",
        "Associated with cognition, consciousness, sensorimotor, and emotional processes",
        "Primarily studied in relation to sensorimotor behavior and cognitive processes",
        "Linked to cognitive and perceptual integration processes",
    ],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the table in the Streamlit app
st.title("Frequency Bands and Brain States")
st.table(df)
