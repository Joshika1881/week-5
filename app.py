import streamlit as st

from apputil import *


# Load Titanic dataset
df = load_data()


st.write(
    """
# Titanic Visualization 1

How did survival rates differ by age group, sex, and passenger class?
"""
)

# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)


st.write(
    """
# Titanic Visualization 2

How did average fares differ by family size and passenger class?
"""
)

# Generate and display the figure
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)


# Compare last names with family sizes in the dataset
names = last_names()

st.write(
    """
### Family Group Findings

Several passengers shared the same last name. Andersson was the most
common last name with 9 passengers, followed by Sage with 7. This
suggests that some larger family groups may represent passengers
traveling with relatives who shared a last name.
"""
)
