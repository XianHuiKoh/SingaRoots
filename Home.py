import streamlit as st

from utility import check_password

st.set_page_config(
    page_title="SingaRoots", layout="wide", initial_sidebar_state="expanded"
)

if not check_password():
    st.stop()

st.title("SingaRoots", anchor=False)

st.markdown(
    """    
    Welcome to **SingaRoots**, the dedicated platform for accessing information on 
    **Singapore's 25 Key Historical Figures** and **25 Intangible Cultural Heritage (ICH) Elements**. 
    """
)

st.header("🚀 Proceed to Data Domains", anchor=False)

st.markdown(
    """
To begin your inquiry, please use the **Sidebar Navigation** on the left to select your domain. 
Each domain corresponds to a dedicated page that provides dual access modes: **Question-Based Retrieval** (for precise QA) and **Exploratory Retrieval** (for structured, infographic-style profiling).
"""
)

# Use columns to present the two domains clearly without buttons
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("🗿 Historical Figures", anchor=False)
        st.markdown(
            """
            Select **Historical Figures** in the sidebar.
            
            *Focus: Biographies, roles, contributions, and tenure of key individuals.*
            """
        )

with col2:
    with st.container(border=True):
        st.subheader("🎭 ICH Elements", anchor=False)
        st.markdown(
            """
            Select **Intangible Cultural Heritage (ICH)** in the sidebar.
            
            *Focus: Traditions, practices, arts, festivals, and social customs.*
            """
        )

with st.expander("Disclaimer:"):
    st.markdown(
        """
        _IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. \
        The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, 
        especially those related to financial, legal, or healthcare matters._

        _Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. \
        You assume full responsibility for how you use any generated output._

        _Always consult with qualified professionals for accurate and personalized advice._
        """
    )
