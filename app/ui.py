import json
import html
import streamlit as st
from typing import Callable, List, Dict, Any

def render_header() -> None:
    st.title("💬 Customer Support Chatbot (Prototype)")
    st.caption("Streamlit UI scaffold — next step: Groq LLM + MCP tool calls.")

def render_messages(messages: List[Dict[str, str]]) -> None:
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)

def render_sidebar(on_reset: Callable[[], None], debug_events: List[Dict[str, Any]]) -> None:
    with st.sidebar:
        st.header("Controls")

        if st.button("🧹 Reset chat", use_container_width=True):
            on_reset()
            st.rerun()

        st.checkbox("Show debug panel", key="show_debug", value=True)

        st.divider()
        st.subheader("About")
        st.write("Current: Groq + (optional) LangChain + MCP tools.")

    # Render the debug panel pinned at the bottom-left INSIDE the sidebar
    if not st.session_state.get("show_debug", True):
        return

    events = debug_events[-30:] if debug_events else []
    text = "No debug events yet." if not events else json.dumps(events, ensure_ascii=False, indent=2)
    safe_text = html.escape(text)

    st.markdown(
        """
        <style>
          /* Make room so sidebar content doesn't get covered by the fixed panel */
          section[data-testid="stSidebar"] .stSidebarContent {
            padding-bottom: 300px !important;
          }

          section[data-testid="stSidebar"] .debug-panel {
            position: fixed;
            bottom: 16px;
            left: 16px;
            width: calc(21rem - 32px); /* matches default sidebar width */
            max-height: 260px;
            overflow-y: auto;
            background: rgba(255,255,255,0.95);
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px 12px;
            z-index: 999999;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 12px;
            line-height: 1.3;
          }

          section[data-testid="stSidebar"] .debug-title {
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 6px;
          }

          section[data-testid="stSidebar"] .debug-pre {
            white-space: pre-wrap;
            margin: 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Important: inject into sidebar area (it will still be fixed, but now visible)
    st.sidebar.markdown(
        f"""
        <div class="debug-panel">
          <div class="debug-title">Debug (latest events)</div>
          <pre class="debug-pre">{safe_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )