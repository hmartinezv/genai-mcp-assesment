import streamlit as st
from app.state import init_state, reset_chat
from app.ui import render_sidebar, render_header, render_messages

def placeholder_bot_reply(user_text: str) -> str:
    return (
        "Thanks — I can help with that.\n\n"
        f"You said: “{user_text}”\n\n"
        "For now this is a placeholder reply. Next we will connect the LLM and MCP tools."
    )

def main() -> None:
    st.set_page_config(
        page_title="Support Chatbot Prototype",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    init_state()
    render_sidebar(on_reset=reset_chat, debug_events=st.session_state.debug_events)
    render_header()
    render_messages(st.session_state.messages)

    user_text = st.chat_input("Describe your issue (e.g., 'printer offline', 'monitor no signal')")

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

        reply = placeholder_bot_reply(user_text)
        st.session_state.messages.append({"role": "assistant", "content": reply})

        st.session_state.debug_events.append({
            "type": "placeholder",
            "message": "No tool calls yet. LLM/MCP integration comes next.",
        })

        st.rerun()

if __name__ == "__main__":
    main()