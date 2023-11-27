import streamlit as st
import pandas as pd

## EEG Features for Diagnosis Section
st.header("EEG Features for Diagnosis")

### Frequency Analysis
st.subheader("Frequency Analysis")
st.write(
    """
The brain's electrical activity is often categorized into distinct frequency bands, including delta, theta, alpha, beta, and gamma. These bands represent different neural oscillations, each associated with specific cognitive and mental states.
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
    "Electrodes": [
        ["Fp1[1]", "Fp2[2]", "Cz[19]"],
        ["F3[3]", "F4[4]", "C3[5]", "C4[6]", "T3[13]", "T4[14]"],
        ["O1[9]", "O2[10]", "P3[7]", "P4[8]", "Fz[17]"],
        ["F7[11]", "F8[12]", "T5[15]", "T6[16]", "Pz[18]"],
        ["Fz[17]", "Pz[18]", "O1[9]", "O2[10]", "Cz[19]"],
    ],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the table in the Streamlit app
st.title("Frequency Bands and Brain States")
st.table(df)
