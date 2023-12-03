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

st.write(
    """
### Participants

#### Demographic Information:

- **Age:**
  - Patients: Mean age ± Standard Deviation (SD)
  - Controls: Mean age ± SD

- **Gender Distribution:**
  - Patients: [Number of Males / Number of Females / Other]
  - Controls: [Number of Males / Number of Females / Other]

- **Ethnicity (if applicable):**
  - Provide a breakdown of the ethnic composition of the participants.

#### Clinical Characteristics (for patients):

- **Duration of Illness:**
  - Average duration of schizophrenia ± SD (if applicable)

- **Clinical Assessments:**
  - Mention any relevant clinical assessments conducted for patients.

#### Inclusion and Exclusion Criteria:

- Clearly defined inclusion criteria for both patient and control groups.
- Clearly defined exclusion criteria for both patient and control groups.

#### Participant Recruitment:

- Describe the methods used for participant recruitment, such as clinical settings, advertisements, etc.

#### Sample Size:

- Total number of participants: [Number of Patients] (Schizophrenia) + [Number of Controls]

#### Ethical Considerations:

- Emphasize that the study adhered to ethical guidelines, including participant consent and privacy protection.

#### Participant IDs (Anonymization):

- Briefly explain the use of participant IDs to ensure anonymity.

#### Participant Compensation (if applicable):

- State if participants received compensation and provide details.

#### Voluntary Participation:

- Reiterate that participation was voluntary, and participants had the right to withdraw without consequences.

#### Description of Control Group (if applicable):

- Provide information on how the control group was selected and any matching criteria used.

"""
)
