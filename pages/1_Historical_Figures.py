import streamlit as st
import json
from logics.historical_figures_query_handler import (
    get_search_response,
    get_figure_infographics,
)
from helper_functions.llm import FIGURE_INFO_DICT
from utility import check_password

st.set_page_config(page_title="SingaRoots - Historical Figures", layout="wide")

# Do not continue if check_password is not True.
if not check_password():
    st.stop()

SESSION_STATE_KEYS_DEFAULT_DICT = {
    "figures_details": {},
    "is_profile_selected": False,
    "user_query": {},
    "current_query": None,
    "show_warning": False,
}

for key in SESSION_STATE_KEYS_DEFAULT_DICT.keys():
    if key not in st.session_state:
        st.session_state[key] = SESSION_STATE_KEYS_DEFAULT_DICT[key]

FIGURES_DICT = {
    info_dict["formatted_name"]: info_dict["category"]
    for _, info_dict in FIGURE_INFO_DICT.items()
}

BADGE_COLOURS_DICT = {
    "Political Leaders & Nation Builders": "#1F4788",
    "Cultural & Artistic Pioneers": "#F39C12",
    "Social & Community Pioneers": "#27AE60",
    "War Heroes & Resistance Fighters": "#C0392B",
    "Philanthropists & Business Pioneers": "#8E44AD",
    "Colonial Figures": "#7F8C8D",
}


def handle_search_submit():
    search_term = st.session_state.search_term.strip()
    if search_term:
        st.session_state["is_profile_selected"] = False
        st.session_state["current_query"] = search_term
        st.session_state["show_warning"] = False
    else:
        st.session_state["show_warning"] = True


def handle_selection(figure):
    st.session_state["is_profile_selected"] = True
    st.session_state["current_query"] = figure


@st.cache_data(show_spinner=False)
def get_llm_response(input_value, is_profile_selected):
    if is_profile_selected:
        return get_figure_infographics(input_value)
    else:
        return get_search_response(input_value)


def get_cached_response(query, is_profile_selected):
    key = "figures_details" if is_profile_selected else "user_query"
    processed_query = query.lower().strip()
    if processed_query not in st.session_state[key]:
        with st.spinner(f"Waiting response for: {query}..."):
            st.session_state[key][processed_query] = get_llm_response(
                query, is_profile_selected
            )

    return st.session_state[key][processed_query]


def render_header():
    st.header("🗿 Historical Figures", anchor=False)
    st.caption(
        """
        Focuses on biographies, roles, contributions, and tenure of key individuals.

        **You can ask a question or explore figures by category below.**
        """
    )

    st.divider()

    st.subheader("💬 Ask a Question", anchor=False)

    st.markdown(
        """
        💡 **Need ideas? Try these sample questions:**
        * What was Lee Kuan Yew's main vision for Singapore's future after separation from Malaysia?
        * Who is Elizabeth Choy, and why is she an important figure in Singapore's wartime history?
        * Which pioneers are associated with the financial or business development of Singapore?
        """
    )
    with st.form(key="figures_search_form", clear_on_submit=False):
        st.text_area(
            "Ask about a figure or event",
            key="search_term",
            placeholder="Please enter a question on Historical Figures here.",
            label_visibility="collapsed",
        )

        _, col2 = st.columns([6, 1])
        with col2:
            st.form_submit_button(
                "Search 🔍", on_click=handle_search_submit, use_container_width=True
            )

        if st.session_state.get("show_warning"):
            st.warning("Please enter a question.")

    st.divider()


def render_explore_tabs():
    st.subheader("🧭 Explore Historical Figures by Category", anchor=False)

    tabs = st.tabs(list(BADGE_COLOURS_DICT.keys()))
    for category, tab in zip(BADGE_COLOURS_DICT.keys(), tabs):
        with tab:
            st.caption(f"Figures under **{category}**")
            with st.container(height=300, border=False):
                category_figures = [f for f, c in FIGURES_DICT.items() if c == category]

                num_cols = 4
                rows = [
                    category_figures[i : i + num_cols]
                    for i in range(0, len(category_figures), num_cols)
                ]
                for row in rows:
                    cols = st.columns(num_cols)
                    for col, figure in zip(cols, row):
                        with col:
                            with st.container(border=True, height=120):
                                st.markdown(f"**{figure}**")
                                st.button(
                                    "View Profile",
                                    key=f"btn_{figure.replace(' ', '_')}",
                                    on_click=handle_selection,
                                    args=(figure,),
                                    use_container_width=True,
                                )


def render_results_panel():
    input_value = st.session_state.get("current_query")
    is_profile_selected = st.session_state.get("is_profile_selected")

    st.divider()
    st.subheader("📜 Results", anchor=False)

    if not input_value:
        st.info("Search for something or select a figure to see results.")
        return

    response = get_cached_response(input_value, is_profile_selected)

    if is_profile_selected:
        if isinstance(response, str):
            figure_details = json.loads(response)
        else:
            figure_details = response

        category = FIGURES_DICT[input_value]
        badge_color = BADGE_COLOURS_DICT[category]

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
            f"""
            <div style="
                display: flex; align-items: center; justify-content: space-between;
                border-bottom: 3px solid {badge_color}; margin-bottom: 0.5rem; padding-bottom: 0.5rem;
            ">
                <h1 style="margin:0; color:#222; font-size: 2.25rem;">{input_value}</h1>
                <span style="
                    background-color:{badge_color};
                    color:white; padding:6px 12px; border-radius:16px;
                    font-weight:bold; font-size:0.9rem;">
                    {category}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f'Role(s): {figure_details["occupation_or_roles"]}')

        st.markdown("#### 🏅 Known For")
        st.markdown(f"**{figure_details['known_for']}**")

        st.markdown("---")

        st.subheader("🗓️ Quick Facts", anchor=False)

        born, died = st.columns(2)
        born.markdown(
            f'<p class="quick-fact-label">🎂 Born</p><p class="quick-fact-value">{figure_details["birth_date"]}</p>',
            unsafe_allow_html=True,
        )

        died.markdown(
            f'<p class="quick-fact-label">🕊️ Died</p><p class="quick-fact-value">{figure_details["death_date"]}</p>',
            unsafe_allow_html=True,
        )

        birth_place, death_place = st.columns(2)
        birth_place.markdown(
            f'<p class="quick-fact-label">👶 Birth Place</p><p class="quick-fact-value">{figure_details["birth_place"]}</p>',
            unsafe_allow_html=True,
        )

        death_place.markdown(
            f'<p class="quick-fact-label">📍 Death Place</p><p class="quick-fact-value">{figure_details["death_place"]}</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.subheader("📜 Biography", anchor=False)
        st.markdown(
            f"""
            <div style="padding-left:0; margin-bottom:1.5rem;">
                <p style="margin:0; font-size:1.0rem; line-height:1.6;">{figure_details["biography"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.subheader("🌟 Key Contributions & Achievements", anchor=False)

        badge_html = ""
        for c in figure_details["key_contributions"]:
            badge_html += f"""
            <span style="
                background-color: #FAFAFA; /* Very light pink background */
                color: #222; 
                border: 1px solid {badge_color}; /* Border matches the badge_color */
                border-radius: 18px; /* Slightly more rounded for pill shape */
                padding: 6px 12px;
                margin: 6px 6px 6px 0;
                display: inline-block;
                font-size: 0.95rem; 
                font-weight: 500;
            ">
                ✨ {c} 
            </span>
            """
        st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("🏛️ Historical Significance", anchor=False)
        st.markdown(
            f"""
            <div style="
                padding: 10px 15px;
                background-color: #F5F5F5;
                border-radius: 5px;
                border-left: 4px solid #9E9E9E;
                margin-bottom: 1.5rem;
            ">
                <p style="margin:0; font-size:1.0rem;">
                    {figure_details["historical_significance"]}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("🤝 Related Figures", anchor=False)

        if figure_details["related_figures"]:
            st.caption(
                f"These figures are crucial to understanding {input_value}'s context and legacy in Singapore."
            )

            st.markdown(
                """
                <div style="display: grid; grid-template-columns: 1fr 1.5fr; gap: 10px 20px; margin-top: 10px;">
                    <div style="font-weight: bold; border-bottom: 1px solid #ddd; padding-bottom: 5px; color:#444;">Figure</div>
                    <div style="font-weight: bold; border-bottom: 1px solid #ddd; padding-bottom: 5px; color:#444;">Connection</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            for figure in figure_details["related_figures"]:
                st.markdown(
                    f"""
                    <div style="display: grid; grid-template-columns: 1fr 1.5fr; gap: 10px 20px; padding-top: 8px; padding-bottom: 2px;">
                        <div style="font-weight: 500;">{figure['name']}</div>
                        <div style="font-style: italic; color: #555;">{figure['relationship']}</div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No related figures listed.")
    else:
        st.subheader("🤖 You asked:", anchor=False)
        st.markdown(f"**“{input_value}”**")
        st.markdown(response.replace("$", "\$"))


render_header()
render_explore_tabs()

if st.session_state["current_query"]:
    render_results_panel()
