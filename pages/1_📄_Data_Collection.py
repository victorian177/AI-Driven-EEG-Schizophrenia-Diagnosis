import pandas as pd
import streamlit as st


st.write(
    """
        # Data Collection Report
        Here is the Data Collection Report for the project on the "Early Detection of Schizophrenia using EEG Data." This report provides a detailed account of the collection process for EEG data obtained from individuals both with and without schizophrenia, categorized as patients and controls, respectively. The purpose of this report is to elucidate the methodologies employed in data collection, participant demographics, and essential characteristics of the collected EEG data.

        ## Scope of the Report
        1. **Participants:** A detailed overview of the individuals involved in the study, including patients diagnosed with schizophrenia and the control group.
        2. **Data Collection Methodology:** An in-depth explanation of the procedures, tools, and protocols utilized for gathering EEG data.
        3. **Data Distributions:** Presentation and analysis of various distributions within the collected data, highlighting patterns and differences between patient and control groups.

        The insights derived from this comprehensive data collection process are pivotal for the subsequent stages of our project, contributing to the development of a machine learning model for the early detection of schizophrenia.
        """
)

participants = pd.read_csv("participant_info.csv", index_col=False)
st.dataframe(participants)

st.write(
    f"""
### Participants

#### Demographic Information:

- **Age:**
  - Patients: Mean age
  - Controls: Mean age ± SD

- **Gender Distribution:**
  - Patients: [Number of Males / Number of Females]
  - Controls: [Number of Males / Number of Females]

- **Ethnicity (if applicable):**
  - Provide a breakdown of the ethnic composition of the participants.

#### Inclusion and Exclusion Criteria:

- Clearly defined inclusion criteria for both patient and control groups.
- Clearly defined exclusion criteria for both patient and control groups.

#### Sample Size:

- Total number of participants: [Number of Patients] (Schizophrenia) + [Number of Controls]

#### Participant IDs (Anonymization):

- Briefly explain the use of participant IDs to ensure anonymity.

#### Description of Control Group (if applicable):

- Provide information on how the control group was selected and any matching criteria used.

"""
)
