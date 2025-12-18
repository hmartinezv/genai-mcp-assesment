import streamlit as st
from app.state import init_state, reset_chat
from app.ui import render_sidebar, render_header, render_messages
from app.chat_engine import chat_reply, tools_available, tool_command_guide

@st.cache_resource
def preload_tools_and_guide() -> str:
    # Forces MCP tool discovery once per process
    tools_available()
    return tool_command_guide()

def main() -> None:
    st.set_page_config(
        page_title="Support Chatbot Prototype",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    init_state()

    # Preload tools before the first prompt + keep it cached
    guide = preload_tools_and_guide()

    # Your sidebar
    render_sidebar(on_reset=reset_chat, debug_events=st.session_state.debug_events)

    # Show supported commands (prevents hallucinated /tickets etc.)
    with st.sidebar.expander("Supported commands", expanded=True):
        st.code(guide, language="text")

    render_header()
    render_messages(st.session_state.messages)

    user_text = st.chat_input("Describe your issue (e.g., 'printer offline', 'monitor no signal')")

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user"):
            st.markdown(user_text)

        try:
            with st.status("Thinking...", expanded=False) as status:
                assistant_text, debug_event = chat_reply(st.session_state.messages)
                status.update(label="Done", state="complete")
        except Exception:
            with st.spinner("Thinking..."):
                assistant_text, debug_event = chat_reply(st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.debug_events.append(debug_event)

        st.rerun()

if __name__ == "__main__":
    main()