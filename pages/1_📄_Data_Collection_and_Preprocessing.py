import pandas as pd
import streamlit as st


st.write(
    """
        # Data Collection
        Here is the Data Collection Report for the project on the "Early Detection of Schizophrenia using EEG Data." This report provides a detailed account of the collection process for EEG data obtained from individuals both with and without schizophrenia, categorized as patients and controls, respectively. The purpose of this report is to elucidate the methodologies employed in data collection, participant demographics, and essential characteristics of the collected EEG data.

        ## Scope of the Report
        1. **Participants:** A detailed overview of the individuals involved in the study, including patients diagnosed with schizophrenia and the control group.
        2. **Data Collection Methodology:** An in-depth explanation of the procedures, tools, and protocols utilized for gathering EEG data.
        3. **Data Distributions:** Presentation and analysis of various distributions within the collected data, highlighting patterns and differences between patient and control groups.

        The insights derived from this comprehensive data collection process are pivotal for the subsequent stages of our project, contributing to the development of a machine learning model for the early detection of schizophrenia.
        """
)

st.write(" ### EEG Recording Information")
st.write("The EEG recording was conducted using the Generis Software.")
st.write("Recording Method: Referential")
st.write(
    "In referential recording, the electrical activity at each electrode is measured "
    "with respect to a common reference point, usually the ground electrode."
)
st.write(
    "This provides information about the activity at individual electrodes relative to the reference."
)
st.write("Phases of the EEG Recording:")
st.write("1. Rest Phase")
st.write("2. Arithmetic Task Phase")
st.write("3. Another Rest Phase")
st.write("4. Auditory Task Phase")

st.write("Data Format: European Data Format (EDF)")
st.write(
    "The EEG data was stored in the European Data Format, commonly used for biomedical signals such as EEG."
)


participants = pd.read_csv("participant_info.csv", index_col=False)
patients = participants[participants["category"] == "Patient"]
controls = participants[participants["category"] == "Control"]

st.write("### Participants")
st.dataframe(participants)

st.write(
    f"""
#### Demographic Information:

- **Age (Mean ± Standard Deviation):**
  - Patients: {patients["age"].mean():.2f} ± {patients["age"].std():.2f} years
  - Controls: {controls["age"].mean():.2f} ± {controls["age"].std():.2f} years

- **Gender Distribution (Male /Female):**
  - Patients: [{len(patients[patients["sex"] == 'M'])} / {len(patients[patients["sex"] == 'F'])}]
  - Controls: [{len(controls[controls["sex"] == 'M'])} / {len(controls[controls["sex"] == 'F'])}]

#### Sample Size:
- Total number of participants: {len(patients)} (Schizophrenia) + {len(controls)} = {len(participants)}

#### Participant IDs (Anonymization):
- The names of each of the participants was removed to ensure privacy.

"""
)
