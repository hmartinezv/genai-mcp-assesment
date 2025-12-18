from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Tuple
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
try:
    # LangChain v1
    from langchain.agents import create_agent
    _AGENT_FACTORY = "langchain.create_agent"
except Exception:
    # Fallback
    from langgraph.prebuilt import create_react_agent
    create_agent = None
    _AGENT_FACTORY = "langgraph.create_react_agent"

from app.config import settings

SYSTEM_PROMPT = """You are a helpful customer support agent for a company that sells computer products (monitors, printers, etc.).

Rules:
- Ask 1-2 short questions if needed (model, symptoms, error message).
- Prefer using tools when available (MCP tools) instead of guessing.
- Give step-by-step instructions.
- If you cannot solve it, suggest escalation and what info to include.
- Keep answers concise and practical.
"""

def _run_async(coro):
    """
    Streamlit runs sync code. MCP + agent calls are async.
    This helper runs a coroutine safely in typical Streamlit environments.
    """
    try:
        asyncio.get_running_loop()
        # If a loop is already running, use a new event loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(coro)

@lru_cache(maxsize=1)
def _build_agent_and_tools() -> Tuple[Any, List[str]]:
    """
    Create Groq model + load MCP tools once per process.
    Cached so Streamlit reruns don't reload tools every time.
    """
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    if not settings.mcp_server_url:
        raise RuntimeError("MCP_SERVER_URL not set")

    # Imports here so your app still boots even if deps aren't installed yet

    llm = ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        max_retries=2,
    )

    # MCP Streamable HTTP: langchain-mcp-adapters uses transport "http" (aka streamable_http)
    client = MultiServerMCPClient(
        {
            "support": {
                "transport": "http",
                "url": settings.mcp_server_url,
            }
        },
        tool_name_prefix=True,  # helps avoid name collisions later
    )

    tools = _run_async(client.get_tools())
    tool_names = [t.name for t in tools]

    if create_agent is not None:
        agent = create_agent(
                llm,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
            )
    else:
        agent = create_react_agent(
            llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )
    return agent, tool_names

def _extract_tool_traces(agent_messages: List[Any]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for m in agent_messages:
        cls = m.__class__.__name__
        # Tool messages usually come back as ToolMessage
        if cls == "ToolMessage":
            traces.append({
                "type": "tool_message",
                "tool_name": getattr(m, "name", None),
                "content": getattr(m, "content", ""),
            })
    return traces

def chat_reply(messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any]]:
    """
    Called by Streamlit.
    `messages` are your st.session_state messages in dict form:
      [{"role":"user"/"assistant", "content":"..."}]
    """
    user_text = messages[-1]["content"] if messages else ""

    # If keys missing, keep current behavior
    if not settings.groq_api_key or not settings.mcp_server_url:
        return (
            "✅ UI is working.\n\n"
            f'You said: “{user_text}”.\n\n'
            "Set GROQ_API_KEY and MCP_SERVER_URL to enable real answers + tool calls.",
            {
                "type": "config",
                "level": "warning",
                "message": "Missing GROQ_API_KEY or MCP_SERVER_URL — using placeholder.",
            },
        )

    try:
        agent, tool_names = _build_agent_and_tools()

        # LangGraph agents accept the messages in dict format like:
        # {"messages": [{"role":"user","content":"..."}, ...]}
        result = _run_async(agent.ainvoke({"messages": messages}))

        agent_messages = result.get("messages", [])
        assistant_text = agent_messages[-1].content if agent_messages else "Sorry, I got no response."

        tool_traces = _extract_tool_traces(agent_messages)

        debug_event = {
            "type": "agent",
            "model": settings.groq_model,
            "mcp_server_url": settings.mcp_server_url,
            "tools_available": tool_names,
            "tool_traces": tool_traces,
        }
        return assistant_text, debug_event

    except Exception as e:
        return (
            "I hit an internal error while generating the answer. Please try again.",
            {
                "type": "error",
                "message": str(e),
            },
        )