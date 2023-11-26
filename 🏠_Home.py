import streamlit as st
from pages import data_preprocessing

# Create a dictionary to map page names to functions
pages = {"Data Input": data_preprocessing.app}

# Create a sidebar to select pages
selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))

# Run the selected page
pages[selected_page]()
