import streamlit as st
import json
from logics.ich_query_handler import (
    get_search_response,
    get_element_infographics,
)
from helper_functions.llm import ICH_INFO_DICT

from utility import check_password

st.set_page_config(page_title="SingaRoots - ICH Elements", layout="wide")

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

SESSION_STATE_KEYS_DEFAULT_DICT = {
    "ich_elements": {},
    "is_element_selected": False,
    "ich_user_query": {},
    "ich_current_query": None,
    "ich_show_warning": False,
}

for key in SESSION_STATE_KEYS_DEFAULT_DICT.keys():
    if key not in st.session_state:
        st.session_state[key] = SESSION_STATE_KEYS_DEFAULT_DICT[key]

ICH_CATEGORIES_DICT = {
    info_dict["formatted_name"]: info_dict["categories"]
    for _, info_dict in ICH_INFO_DICT.items()
}
BADGE_COLOURS_DICT = {
    "Performing Arts": "#B71C1C",
    "Food Heritage": "#FF8F00",
    "Social Practices, Rituals and Festive Events": "#FFC107",
    "Oral Traditions and Expressions": "#3F51B5",
    "Traditional Craftsmanship": "#2E7D32",
    "Knowledge and Practices concerning Nature and Universe": "#6D4C41",
}


def handle_search_submit():
    search_term = st.session_state.search_term.strip()
    if search_term:
        st.session_state["is_element_selected"] = False
        st.session_state["ich_current_query"] = search_term
        st.session_state["ich_show_warning"] = False
    else:
        st.session_state["ich_show_warning"] = True


def handle_selection(element):
    st.session_state["is_element_selected"] = True
    st.session_state["ich_current_query"] = element


@st.cache_data(show_spinner=False)
def get_llm_response(input_value, is_element_selected):
    if is_element_selected:
        return get_element_infographics(input_value)
    else:
        return get_search_response(input_value)


def get_cached_response(query, is_element_selected):
    key = "ich_elements" if is_element_selected else "ich_user_query"
    processed_query = query.lower().strip()
    if processed_query not in st.session_state[key]:
        with st.spinner(f"Waiting response for: {query}..."):
            st.session_state[key][processed_query] = get_llm_response(
                query, is_element_selected
            )

    return st.session_state[key][processed_query]


def render_header():
    st.header("🎭 Intangible Cultural Heritage (ICH) Elements", anchor=False)
    st.caption(
        """
        Focuses on traditions, practices, arts, festivals, and social customs.
        
        **You can ask a question of explore ICH elements by category below.**
        """
    )

    st.divider()

    st.subheader("💬 Ask a Question", anchor=False)
    st.markdown(
        """
        💡 **Need ideas? Try these sample questions:**
        * Which cultural traditions are associated with the Performing Arts?
        * What is the historical background and purpose of the Chingay parade?
        * What are the essential ingredients and steps for preparing authentic Singapore chicken rice?
    """
    )

    with st.form(key="ich_search_form", clear_on_submit=False):
        st.text_area(
            "Ask about a ICH element of Singapore",
            key="search_term",
            placeholder="Please enter a question on ICH elements here.",
            label_visibility="collapsed",
        )

        _, col2 = st.columns([6, 1])
        with col2:
            st.form_submit_button(
                "Search 🔍", on_click=handle_search_submit, use_container_width=True
            )

        if st.session_state.get("ich_show_warning"):
            st.warning("Please enter a question.")
    st.divider()


def render_explore_tabs():
    st.subheader("🧭 Explore ICH Elements by Category", anchor=False)

    tabs = st.tabs(list(BADGE_COLOURS_DICT.keys()))
    for category, tab in zip(BADGE_COLOURS_DICT.keys(), tabs):
        with tab:
            st.caption(f"Elements under **{category}**")
            with st.container(height=300, border=False):
                category_elements = [
                    f for f, c in ICH_CATEGORIES_DICT.items() if category in c
                ]

                num_cols = 3
                rows = [
                    category_elements[i : i + num_cols]
                    for i in range(0, len(category_elements), num_cols)
                ]
                for row in rows:
                    cols = st.columns(num_cols)
                    for col, element in zip(cols, row):
                        with col:
                            with st.container(border=True, height=120):
                                st.markdown(f"**{element}**")
                                st.button(
                                    "View Element",
                                    key=f"btn_{category}_{element.replace(' ', '_')}",
                                    on_click=handle_selection,
                                    args=(element,),
                                    use_container_width=True,
                                )


def render_results_panel():
    input_value = st.session_state.get("ich_current_query")
    is_element_selected = st.session_state.get("is_element_selected")

    st.divider()
    st.subheader("📜 Results", anchor=False)

    if not input_value:
        st.info("Search for something or select an element to see results.")
        return

    response = get_cached_response(input_value, is_element_selected)

    if is_element_selected:
        if isinstance(response, str):
            element_details = json.loads(response)
        else:
            element_details = response

        categories = ICH_CATEGORIES_DICT[input_value]

        border_color = BADGE_COLOURS_DICT[categories[0]]

        st.markdown(
            """
            <style>
            .quick-fact-label {
                font-weight: bold;
                color: #333333;
                margin-bottom: 0.25rem;
            }

            .quick-fact-value {
                color: #555555;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<h1 style="margin:0; color:#222; font-size: 2.25rem;">{input_value}</h1>',
            unsafe_allow_html=True,
        )

        category_html = ""
        for cat in categories:
            badge_color = BADGE_COLOURS_DICT[cat]
            category_html += f"""
            <span style="background-color:{badge_color}; color:white; padding:6px 12px; border-radius:16px;
                font-weight:bold; font-size:0.9rem; margin-right:8px; white-space:nowrap; display:inline-block;
                text-align:center;">{cat}</span>
            """
        st.markdown(
            f"""
            <div style="margin-top: 10px; margin-bottom: 10px; display: flex; flex-wrap: wrap;">
                {category_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="
                border-top: 2px solid {border_color}; /* Thinned the border */
                margin-top: 10px; 
                margin-bottom: 25px;
            "></div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.subheader("📚 Quick Facts", anchor=False)
            ref_no, inclusion_date, _ = st.columns(3)
            ref_no.markdown(
                f'<p class="quick-fact-label">Reference No.</p><p class="quick-fact-value">{element_details["ref_no"]}</p>',
                unsafe_allow_html=True,
            )

            inclusion_date.markdown(
                f'<p class="quick-fact-label">Date of Inclusion</p><p class="quick-fact-value">{element_details["date_of_inclusion"]}</p>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            col1.markdown(
                f'<p class="quick-fact-label">Other Name</p><p class="quick-fact-value">{"-" if element_details["chinese_or_other_name"].lower().startswith("not available") else element_details["chinese_or_other_name"]}</p>',
                unsafe_allow_html=True,
            )
            col2.markdown(
                f'<p class="quick-fact-label">Communities Involved</p><p class="quick-fact-value">{element_details["communities_involved"]}</p>',
                unsafe_allow_html=True,
            )
            col3.markdown(
                f'<p class="quick-fact-label">Location</p><p class="quick-fact-value">{element_details["geographic_location"]}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown(
            f"""
            <div style="border-left:4px solid {border_color}; padding-left:1rem; margin-bottom:1.5rem;">
                <h3 style="margin-bottom:0.5rem;">🕰️ Origins & History</h3>
                <p style="margin:0; font-size:1.05rem;">{element_details["origins_and_historical_evolution"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛠️ Core Practice & Elements", anchor=False)
            st.write(element_details["description_of_practice"])

        with col2:
            st.subheader("👥 Associated Social Practices", anchor=False)
            practices_list = "\n".join(
                [f"- {p}" for p in element_details["associated_practices_and_rituals"]]
            )
            st.markdown(practices_list)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🔄 Transmission & Safeguarding", anchor=False)
            transmission_list = "\n".join(
                [
                    f"- {t}"
                    for t in element_details["transmission_and_safeguarding_efforts"]
                ]
            )
            st.markdown(transmission_list)

        with col4:
            st.subheader("✨ Current Significance", anchor=False)
            st.markdown(
                f"""
                <div style="
                    background-color:#fff5e5; /* Light, warm background */
                    padding:0.75rem 1rem;
                    border-radius:8px;
                    border-left:4px solid #ff9900; /* Orange highlight */
                ">
                    {element_details["current_significance_in_sg"]}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.subheader("🤖 You asked:", anchor=False)
        st.markdown(f"**“{input_value}”**")
        st.write(response)


render_header()
render_explore_tabs()

if st.session_state["ich_current_query"]:
    render_results_panel()
