import streamlit as st

DEFAULT_WELCOME = (
    "Hi! I’m your support assistant.\n\n"
    "Tell me what product you have (monitor/printer/etc.) and what issue you see."
)

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": DEFAULT_WELCOME}
        ]
    if "debug_events" not in st.session_state:
        st.session_state.debug_events = []

def reset_chat() -> None:
    st.session_state.messages = [
        {"role": "assistant", "content": DEFAULT_WELCOME}
    ]
    st.session_state.debug_events = []