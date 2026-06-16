import streamlit as st

# Configure the page layout and appearance
st.set_page_config(
    page_title="Churn Intelligence",  # Browser tab title
    page_icon="📉",  # Browser tab icon
    layout="wide",  # Use full width layout
    initial_sidebar_state="expanded",  # Keep sidebar open by default
)


single_page = st.Page("pages/single.py", title="Single Prediction", icon="📊")
batch_page = st.Page("pages/batch.py", title="Batch Prediction", icon="📂")
history_page = st.Page("pages/history.py", title="History",           icon="🕐")

# Create navigation menu with both pages
pg = st.navigation([single_page, batch_page, history_page])

# Run the selected page (single or batch)
pg.run()
