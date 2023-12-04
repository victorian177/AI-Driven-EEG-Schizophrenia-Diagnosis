import streamlit as st
import pandas as pd
import numpy as np
import os

## EEG Features for Diagnosis Section
st.header("EEG Features for Diagnosis")

st.write(
    "When working with EEG data, extracting meaningful features is crucial for gaining insights into brain activity. "
    "There are three primary ways to generate features from EEG recordings:"
)

st.write(
    "1. **Frequency Analysis:** This method involves analyzing the different frequency components of the EEG signal. "
    "Common frequency bands include Alpha, Beta, and Gamma. These bands provide information about the oscillatory patterns "
    "of brain activity and are often associated with different cognitive states."
)

st.write(
    "2. **Statistical Characteristics:** Extracting statistical features such as the mean, standard deviation, skewness, "
    "and kurtosis can provide a quantitative description of the EEG signal. These measures offer insights into the central "
    "tendency, variability, and shape of the signal distribution."
)

st.write(
    "3. **Event-Related Potentials (ERPs):** ERPs represent the brain's electrical response to specific stimuli or events. "
    "Components like N100 and P300 are examples of ERPs that are time-locked to particular events. Analyzing ERPs allows "
    "researchers to understand how the brain processes information in response to external stimuli."
)

st.write(
    "These three approaches offer complementary perspectives on EEG data, enabling researchers to capture both the "
    "temporal and spectral characteristics of brain activity."
)

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
    "Frequency Range": ["0.5-4.0 Hz", "4-8 Hz", "8-13 Hz", "12-30 Hz", "30-100 Hz"],
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

patients = [10, 14, 15, 16, 18, 2, 20, 21, 22, 23, 25, 3, 4, 5, 6, 7, 8, 9]

a_cntrls = {}
a_ptnts = {}

b_cntrls = {}
b_ptnts = {}

g_cntrls = {}
g_ptnts = {}


def calculate_power(data, freq_range):
    lower, upper = freq_range
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), 1 / 100)

    # Filter frequencies within the range specified
    mask = (frequencies >= lower) & (frequencies <= upper)
    filtered_fft_result = fft_result[mask]
    # Calculate power spectrum
    power_spectrum = (np.abs(filtered_fft_result) ** 2) / len(filtered_fft_result) ** 2

    return power_spectrum


alpha_electrodes = ["O1[9]", "O2[10]", "P3[7]", "P4[8]", "Fz[17]"]

for dir in os.listdir("Output EEG Data"):
    rest0_phase_file = f"Output EEG Data/{dir}/Phase 1.csv"
    rest0_phase = pd.read_csv(rest0_phase_file)

    for electrode in alpha_electrodes:
        data = rest0_phase[electrode]

        if int(dir) not in patients:
            a_cntrls[dir] = {}
            a_cntrls[dir][electrode] = calculate_power(data, (8, 13))
        else:
            a_ptnts[dir] = {}
            a_ptnts[dir][electrode] = calculate_power(data, (8, 13))

st.write("# Alpha waves")
st.subheader("Power spectrum of control")
st.line_chart(a_cntrls["1"][alpha_electrodes[-1]])

st.subheader("Power spectrum of patient")
st.line_chart(a_ptnts["10"][alpha_electrodes[-1]])


beta_electrodes = ["F7[11]", "F8[12]", "T5[15]", "T6[16]", "Pz[18]"]
gamma_electrodes = ["Fz[17]", "Pz[18]", "O1[9]", "O2[10]", "Cz[19]"]

for dir in os.listdir("Output EEG Data"):
    arith_phase_file = f"Output EEG Data/{dir}/Phase 2.csv"
    arith_phase = pd.read_csv(arith_phase_file)

    for electrode in beta_electrodes:
        data = arith_phase[electrode]

        if int(dir) not in patients:
            b_cntrls[dir] = {}
            b_cntrls[dir][electrode] = calculate_power(data, (12, 30))
            g_cntrls[dir] = {}
            g_cntrls[dir][electrode] = calculate_power(data, (30, 100))

        else:
            b_ptnts[dir] = {}
            b_ptnts[dir][electrode] = calculate_power(data, (12, 30))
            g_ptnts[dir] = {}
            g_ptnts[dir][electrode] = calculate_power(data, (30, 100))

st.write("# Beta waves")
st.subheader("Power spectrum of control")
st.line_chart(b_cntrls["1"][beta_electrodes[-1]])

st.subheader("Power spectrum of patient")
st.line_chart(b_ptnts["10"][beta_electrodes[-1]])

st.write("# Gamma waves")
st.subheader("Power spectrum of control")
st.line_chart(g_cntrls["1"][gamma_electrodes[1]])

st.subheader("Power spectrum of patient")
st.line_chart(g_ptnts["10"][gamma_electrodes[1]])
