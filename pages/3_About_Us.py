import streamlit as st

from utility import check_password

st.set_page_config(page_title="About SingaRoots", layout="wide")

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

st.title("About SingaRoots", anchor=False)

st.subheader("📐 Project Scope", anchor=False)
st.write(
    """
    SingaRoots is a platform designed to help users explore the rich historical and cultural heritage of Singapore. \
    It focuses specifically on 25 key historical figures and 25 intangible cultural heritage (ICH) elements, \
    presenting essential information in a way that is engaging and easy to navigate. \
    While Singapore's heritage ecosystem spans many areas - including places, events and artefacts - \
    SingaRoots intentionally concentrates on these curated entities to provide a focused and accessible experience.
    """
)

st.subheader("🎯 Objectives", anchor=False)
st.write(
    """
    SingaRoots aims to make heritage content more discoverable, understandable, and enjoyable for casual users, students, and educators.
    """
)
st.markdown(
    """
    Its key objectives are to:
    - Spark curiosity about Singapore's history and culture.
    - Reduce the effort needed to find specific information across multiple long-form articles and static webpages.
    - Provide clear, concise, and engaging summaries of heritage figures and ICH elements.
    - Showcase that heritage content can be presented in a dynamic and digestible way - not always "boring and wordy".
    """
)

st.subheader("🗄️ Data Source", anchor=False)
st.markdown(
    """
    The platform draws its information from reputable open-source heritage databases, primarily:
    - **Singapore Infopedia** (by the National Library Board), e.g.:
        - [Lim Kim San (Historical Figure)](https://www.nlb.gov.sg/main/article-detail?cmsuuid=0c994f60-ca57-4d67-8240-44eb77c5fd23)
        - [Chingay (ICH Element)](https://www.nlb.gov.sg/main/article-detail?cmsuuid=c5b0f751-ad72-47d1-ad4a-8ff51d0607e4)
    - **Roots.sg** (by the National Heritage Board), e.g.:
        - [Lim Kim San (Historical Figure)](https://www.roots.gov.sg/stories-landing/stories/lim-kim-san/story)
        - [Chingay (ICH Element)](https://www.roots.gov.sg/ich-landing/ich/chingay)
    """
)
st.write(
    """
    These sources contain rice and authoritative content on Singapore's cultural identity. \
    SingaRoots organises and presents this information in a more user-friendly format, making it easier for the public to explore and understand.
    """
)

st.subheader("✨ Features: Domains and Retrieval Modes", anchor=False)
st.markdown(
    """
    SingaRoots provides access to two primary **Data Domains** (Historical Figures and ICH Elements), with each domain supporting two distinct **Retrieval Modes**.
    
    This matrix clearly outlines the structure of the application:
    """
)

# Use Streamlit's native table or data frame function for clarity
st.markdown(
    """
    | Domain | Content Type | Retrieval Mode 1: Question-Based | Retrieval Mode 2: Exploratory |
    | :--- | :--- | :--- | :--- |
    | **Historical Figures** | 25 Key Individuals | Ask specific questions (e.g., "What was Lim Kim San's first role?"). | Browse a list and view a structured, infographic profile. |
    | **ICH Elements** | 25 Cultural Practices | Ask specific questions (e.g., "When is Chingay usually celebrated?"). | Browse a list and view a structured, infographic profile. |
    """
)

st.markdown("---")

st.subheader("🔍 Detailed Retrieval Mode Descriptions", anchor=False)
st.markdown(
    """
    Each of the two domain pages provides the following distinct retrieval experiences:

    * **1. Question-Based Retrieval (Directed QA):**
        Users can ask specific questions relevant to the selected domain (Historical Figures **or** ICH Elements). SingaRoots retrieves relevant information \
        directly from its curated database of articles for that domain. This allows users to get quick, precise answers without needing to search through \
        long-form content across multiple platforms.

    * **2. Exploratory Retrieval (Structured Profiling):**
        Users can browse a list of all available entities within the selected domain. \
        Selecting an entity displays an infographic-style profile that provides clear, concise, and visually engaging details—making it easy to learn at a glance.
    """
)
