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
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Gamma",
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

st.title("EEG Patterns During Different Brain States and Cognitive Tasks")

# Rest
st.header("Rest")
st.write(
    "During rest, the EEG is typically dominated by alpha waves. However, there may also be some theta waves and delta waves present. The amplitude of the alpha waves is higher in the occipital lobe, which is the part of the brain that is responsible for vision."
)

# Performing arithmetic tasks
st.header("Performing Arithmetic Tasks")
st.write(
    "When performing arithmetic tasks, there is an increase in gamma waves and beta waves in the prefrontal cortex and parietal cortex. These areas of the brain are involved in planning, decision-making, and memory."
)

# Performing auditory tasks
st.header("Performing Auditory Tasks")
st.write(
    "When performing auditory tasks, there is an increase in gamma waves and beta waves in the auditory cortex. This area of the brain is responsible for processing sound."
)

import streamlit as st

st.title("EEG Wave Differences: Normal vs. Schizophrenia")

# Alpha Waves
st.header("Alpha Waves")
st.write(
    "In normal individuals, alpha waves are typically the dominant frequency band during rest. They are characterized by a rhythmic oscillation of 8-13 Hz and are most prominent in the occipital and parietal lobes. Alpha waves are associated with a state of relaxation and decreased cognitive activity."
)
st.write(
    "In individuals with schizophrenia, alpha waves exhibit abnormalities in both amplitude and frequency. Studies have shown that alpha power is often reduced in individuals with schizophrenia, particularly in the frontal and temporal lobes. Additionally, the frequency of alpha waves may be shifted to a lower range in schizophrenia. These abnormalities in alpha waves are thought to reflect underlying disruptions in neural connectivity and synchronization in individuals with schizophrenia."
)

# Beta Waves
st.header("Beta Waves")
st.write(
    "Beta waves, characterized by a frequency range of 12-30 Hz, are associated with alertness, focused attention, and sensorimotor activity. In normal individuals, beta waves increase in amplitude during tasks that require mental effort, such as performing arithmetic or auditory tasks."
)
st.write(
    "In individuals with schizophrenia, beta waves exhibit abnormalities in both amplitude and connectivity. Studies have shown that beta power may be increased or decreased in individuals with schizophrenia, depending on the specific task and brain region being examined. Additionally, beta activity may be more disorganized and less synchronized in schizophrenia. These abnormalities in beta waves are thought to reflect underlying disruptions in sensorimotor integration and cognitive control in individuals with schizophrenia."
)

# Gamma Waves
st.header("Gamma Waves")
st.write(
    "Gamma waves, the fastest frequency band of EEG oscillations (30-100 Hz), are associated with high-level cognitive processing, such as perception, memory, and attention. In normal individuals, gamma waves increase in amplitude during tasks that require these cognitive processes."
)
st.write(
    "In individuals with schizophrenia, gamma waves exhibit abnormalities in both amplitude, frequency, and connectivity. Studies have shown that gamma power may be increased or decreased in individuals with schizophrenia, depending on the specific task and brain region being examined. Additionally, gamma activity may be less synchronized and more phase-dispersed in schizophrenia. These abnormalities in gamma waves are thought to reflect underlying disruptions in neural communication and information integration in individuals with schizophrenia."
)

st.write(
    "Overall, EEG wave abnormalities in schizophrenia are thought to reflect underlying disruptions in brain connectivity, synchronization, and information processing. These abnormalities may contribute to the cognitive and behavioral symptoms of schizophrenia."
)

