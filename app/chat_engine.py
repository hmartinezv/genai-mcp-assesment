from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from langchain.agents import create_agent
    _AGENT_FACTORY = "langchain.create_agent"
except Exception:
    from langgraph.prebuilt import create_react_agent
    create_agent = None
    _AGENT_FACTORY = "langgraph.create_react_agent"

from app.config import settings

USE_AGENT = os.getenv("USE_AGENT", "true").lower() in {"1", "true", "yes"}
PRELOAD_TOOLS = os.getenv("PRELOAD_TOOLS", "true").lower() in {"1", "true", "yes"}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PIN_RE = re.compile(r"\b(\d{4,8})\b")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)

# Tools where we want FULL output in UI (no LLM summary)
FULL_OUTPUT_TOOLS = {"support_list_products", "support_search_products"}

SYSTEM_PROMPT_BASE = """You are a helpful customer support agent for a company that sells computer products (monitors, printers, etc.).

Rules:
- NEVER show tool/function syntax like <function=...> or JSON tool calls to the user.
- ONLY use the tools and commands listed in the "Available tools & commands" section.
- If the user asks for something not supported (example: support tickets), say it's not available in this demo and offer alternatives (products or orders).
- Be concise and practical.
"""


def _run_async(coro):
    try:
        asyncio.get_running_loop()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(coro)


def _trim_messages(messages: List[Dict[str, str]], max_pairs: int = 3) -> List[Dict[str, str]]:
    keep = 2 * max_pairs
    return messages if len(messages) <= keep else messages[-keep:]


def _safe_preview(obj: Any, max_chars: int = 3500) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return s if len(s) <= max_chars else s[:max_chars] + "\n... (truncated)"


def extract_tool_text(tool_output: Any) -> str:
    if tool_output is None:
        return ""
    if isinstance(tool_output, list):
        parts: List[str] = []
        for item in tool_output:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(tool_output, dict):
        if isinstance(tool_output.get("text"), str):
            return tool_output["text"].strip()
        return json.dumps(tool_output, ensure_ascii=False, indent=2)
    return str(tool_output).strip()


def clean_tool_syntax(text: str) -> str:
    return re.sub(r"<function=.*?</function>", "", text, flags=re.DOTALL).strip()


def _tool_args_fields(tool: Any) -> List[str]:
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            js = schema.model_json_schema()  # pydantic v2
            props = js.get("properties", {}) or {}
            return list(props.keys())
        except Exception:
            try:
                js = schema.schema()  # pydantic v1
                props = js.get("properties", {}) or {}
                return list(props.keys())
            except Exception:
                pass
    args = getattr(tool, "args", None)
    if isinstance(args, dict):
        return list(args.keys())
    return []


@lru_cache(maxsize=1)
def _mcp_tools_map() -> Dict[str, Any]:
    if not settings.mcp_server_url:
        raise RuntimeError("MCP_SERVER_URL not set")
    client = MultiServerMCPClient(
        {"support": {"transport": "http", "url": settings.mcp_server_url}},
        tool_name_prefix=True,
    )
    tools = _run_async(client.get_tools())
    return {t.name: t for t in tools}


def tools_available() -> List[str]:
    try:
        return sorted(_mcp_tools_map().keys())
    except Exception:
        return []


def _ensure_tools_preloaded():
    if not PRELOAD_TOOLS:
        return
    if not settings.mcp_server_url:
        return
    try:
        _mcp_tools_map()
        tool_command_guide()
    except Exception:
        pass


# --- Commands (now aligned with your schemas) ---
COMMAND_DEFS: Dict[str, Dict[str, str]] = {
    "support_list_products": {
        "usage": "/products [category=<category>] [is_active=true|false]",
        "example": "/products category=Monitors is_active=true",
        "desc": "List products (optional filters)",
    },
    "support_search_products": {
        "usage": "/search <keyword>",
        "example": "/search monitor",
        "desc": "Search products by keyword",
    },
    "support_get_product": {
        "usage": "/product <sku>",
        "example": "/product MON-0088",
        "desc": "Get product details by SKU",
    },
    "support_get_customer": {
        "usage": "/customer <customer_id>",
        "example": "/customer 2f3b7c3e-0f7e-4b6b-9c4a-2a8d1d3f6a11",
        "desc": "Get customer details by customer_id (UUID)",
    },
    "support_verify_customer_pin": {
        "usage": "/verify email=<email> pin=<pin>",
        "example": "/verify email=donaldgarcia@example.net pin=1234",
        "desc": "Verify identity using email + PIN",
    },
    "support_list_orders": {
        "usage": "/orders [customer_id=<uuid>] [status=<status>]",
        "example": "/orders customer_id=2f3b7c3e-0f7e-4b6b-9c4a-2a8d1d3f6a11 status=shipped",
        "desc": "List orders (optional filters)",
    },
    "support_get_order": {
        "usage": "/order <order_id>",
        "example": "/order ORD-10293",
        "desc": "Get order details by order_id",
    },
    "support_create_order": {
        # Items is array of objects; we provide a convenience form for demo:
        "usage": "/create_order customer_id=<uuid> sku=<sku> qty=<int>",
        "example": "/create_order customer_id=2f3b7c3e-0f7e-4b6b-9c4a-2a8d1d3f6a11 sku=MON-0088 qty=1",
        "desc": "Create an order (demo-friendly shorthand)",
    },
}


@lru_cache(maxsize=1)
def tool_command_guide() -> str:
    available = tools_available()
    lines = ["Available tools & commands (ONLY these are supported in this demo):"]
    for t in available:
        d = COMMAND_DEFS.get(t)
        if d:
            lines.append(f"- {d['desc']}: {d['usage']} (example: {d['example']})")
        else:
            fields = _tool_args_fields(_mcp_tools_map()[t])
            if fields:
                lines.append(f"- {t}: args: " + ", ".join(fields))
            else:
                lines.append(f"- {t}: no args")
    lines.append("")
    lines.append("If a user asks for something not listed (like tickets), say it's not available.")
    return "\n".join(lines)


def parse_kv_args(s: str) -> Dict[str, Any]:
    """
    Parse: a=1 b='two words' c=xyz
    Keep values as strings by default (avoid pin becoming int).
    Convert booleans true/false.
    """
    pattern = r"""(\w+)=(".*?"|'.*?'|[^\s]+)"""
    out: Dict[str, Any] = {}
    for k, v in re.findall(pattern, s):
        v = v.strip().strip('"').strip("'")

        lv = v.lower()
        if lv in {"true", "false"}:
            out[k] = (lv == "true")
        else:
            # keep as string
            out[k] = v

    return out


def _validate_uuid_maybe(v: str) -> bool:
    return bool(UUID_RE.match(v.strip()))


def call_tool(tool_name: str, args: dict) -> Any:
    tools_map = _mcp_tools_map()
    if tool_name not in tools_map:
        raise KeyError(f"Unknown tool: {tool_name}")
    return _run_async(tools_map[tool_name].ainvoke(args))


def _extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_pin(text: str) -> Optional[str]:
    m = PIN_RE.search(text)
    return m.group(1) if m else None


def _extract_customer_id_from_tool_output(out: Any) -> Optional[str]:
    """
    Best effort:
    - try JSON parse
    - try UUID regex in text
    """
    # 1) raw dict with customer_id
    if isinstance(out, dict) and isinstance(out.get("customer_id"), str):
        return out["customer_id"]

    # 2) list blocks -> text
    txt = extract_tool_text(out)

    # try to find a UUID anywhere
    m = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", txt)
    if m:
        return m.group(0)

    # 3) JSON embedded in text
    try:
        js = json.loads(txt)
        if isinstance(js, dict) and isinstance(js.get("customer_id"), str):
            return js["customer_id"]
    except Exception:
        pass

    return None


# -----------------------
# Slash command routing
# -----------------------
def route_intent(user_text: str) -> Optional[Dict[str, Any]]:
    raw = user_text.strip()
    t = raw.lower().strip()

    if t in {"/help", "/commands"}:
        return {"tool": "__help__", "args": {}}

    if t.startswith("/products"):
        args = parse_kv_args(raw.replace("/products", "", 1).strip())
        # Valid args: category, is_active (both optional)
        return {"tool": "support_list_products", "args": {k: v for k, v in args.items() if k in {"category", "is_active"}}}

    m = re.match(r"^/search\s+(.+)$", raw, flags=re.I)
    if m:
        return {"tool": "support_search_products", "args": {"query": m.group(1).strip()}}

    m = re.match(r"^/product\s+(.+)$", raw, flags=re.I)
    if m:
        return {"tool": "support_get_product", "args": {"sku": m.group(1).strip()}}

    m = re.match(r"^/customer\s+(.+)$", raw, flags=re.I)
    if m:
        cid = m.group(1).strip()
        return {"tool": "support_get_customer", "args": {"customer_id": cid}}

    if t.startswith("/verify "):
        args = parse_kv_args(raw.replace("/verify", "", 1).strip())
        return {"tool": "support_verify_customer_pin", "args": {k: v for k, v in args.items() if k in {"email", "pin"}}}

    if t.startswith("/orders"):
        args_part = raw.replace("/orders", "", 1).strip()
        args = parse_kv_args(args_part)

        # allow "/orders <uuid>" shorthand:
        if not args and args_part:
            args = {"customer_id": args_part}

        filtered = {k: v for k, v in args.items() if k in {"customer_id", "status"}}
        return {"tool": "support_list_orders", "args": filtered}

    m = re.match(r"^/order\s+(.+)$", raw, flags=re.I)
    if m:
        return {"tool": "support_get_order", "args": {"order_id": m.group(1).strip()}}

    if t.startswith("/create_order "):
        args = parse_kv_args(raw.replace("/create_order", "", 1).strip())

        # demo-friendly shorthand -> real schema (customer_id + items[])
        customer_id = args.get("customer_id")
        sku = args.get("sku")
        qty_raw = args.get("qty", "1")
        try:
            qty = int(qty_raw)
        except Exception:
            qty = 1

        if customer_id and sku:
            items = [{"sku": sku, "qty": qty}]
            return {"tool": "support_create_order", "args": {"customer_id": customer_id, "items": items}}

        # allow advanced: user provides items_json='[{"sku":"...","qty":1}]'
        items_json = args.get("items_json")
        if customer_id and items_json:
            try:
                items = json.loads(items_json)
                return {"tool": "support_create_order", "args": {"customer_id": customer_id, "items": items}}
            except Exception:
                return {"tool": "__error__", "args": {"message": "items_json is not valid JSON"}}

        return {"tool": "__error__", "args": {"message": "Missing customer_id or sku. Use /create_order customer_id=<uuid> sku=<sku> qty=1"}}

    return None


# -----------------------
# Non-slash auto-routing (tool-first)
# -----------------------
TOOL_RULES: List[Tuple[str, str, List[str]]] = [
    ("support_list_products", "none", ["all products", "products available", "list products", "show products", "catalog", "what products"]),
    ("support_search_products", "query", ["search products", "find products", "find me", "look for", "search for"]),
    ("support_get_product", "sku", ["get product", "product details", "product info", "sku"]),
    ("support_list_orders", "orders_flow", ["my orders", "order status", "order history", "check my orders", "status of my orders"]),
]

def auto_route_non_slash(user_text: str) -> Optional[Dict[str, Any]]:
    available = set(tools_available())
    t = user_text.lower()

    for tool_name, mode, keywords in TOOL_RULES:
        if tool_name in available and any(k in t for k in keywords):
            if mode == "none":
                return {"tool": "support_list_products", "args": {}}

            if mode == "query":
                # naive query extraction
                q = user_text
                q = re.sub(r"(?i)search( for)?", "", q).strip()
                q = re.sub(r"(?i)find( me)?( a| an)?", "", q).strip()
                return {"tool": "support_search_products", "args": {"query": q or "printer"}}

            if mode == "sku":
                # look for something like "SKU MON-0088"
                m = re.search(r"(?i)\bsku\b[:\s]*([A-Z]{3}-\d{4})", user_text)
                if m:
                    return {"tool": "support_get_product", "args": {"sku": m.group(1)}}
                return None

            if mode == "orders_flow":
                # If they provide email+pin -> verify then list_orders
                email = _extract_email(user_text)
                pin = _extract_pin(user_text)

                if email and pin and "support_verify_customer_pin" in available:
                    return {"tool": "__verify_then_orders__", "args": {"email": email, "pin": pin}}

                # If they provide customer_id -> list orders
                m = re.search(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}", user_text)
                if m:
                    return {"tool": "support_list_orders", "args": {"customer_id": m.group(0)}}

                # Otherwise ask for email+pin (since verify tool supports that)
                if "support_verify_customer_pin" in available:
                    return {"tool": "__need_email_pin_for_orders__", "args": {}}

                return {"tool": "__need_customer_id_for_orders__", "args": {}}

    return None


def select_tools_for_text(text: str) -> List[str]:
    available = set(tools_available())
    t = text.lower()

    products = ["support_search_products", "support_list_products", "support_get_product"]
    orders = ["support_verify_customer_pin", "support_list_orders", "support_get_order", "support_create_order"]
    customer = ["support_get_customer"]  # get_customer is separate; verify is email-based

    if any(k in t for k in ["product", "products", "monitor", "printer", "catalog", "spec", "manual", "model", "buy"]):
        return [x for x in products if x in available]

    if any(k in t for k in ["order", "orders", "shipping", "delivery", "status", "invoice", "purchase"]):
        return [x for x in orders if x in available]

    if any(k in t for k in ["customer", "account", "profile"]):
        return [x for x in customer if x in available]

    return []


@lru_cache(maxsize=16)
def _build_agent_for_tool_subset(tool_subset_key: Tuple[str, ...]) -> Tuple[Any, List[str]]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    if not settings.mcp_server_url:
        raise RuntimeError("MCP_SERVER_URL not set")

    llm = ChatGroq(
        model=settings.groq_model,
        temperature=settings.temperature,
        max_tokens=256,
        max_retries=2,
    )

    tools_map = _mcp_tools_map()
    selected_names = list(tool_subset_key)
    selected_tools = [tools_map[name] for name in selected_names if name in tools_map]

    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + tool_command_guide()

    if create_agent is not None:
        agent = create_agent(llm, tools=selected_tools, system_prompt=system_prompt)
    else:
        agent = create_react_agent(llm, tools=selected_tools, prompt=system_prompt)

    return agent, selected_names


def explain(tool_name: str, tool_output: Any) -> str:
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0,
        max_tokens=256,
        max_retries=2,
    )
    preview = _safe_preview(tool_output, max_chars=3000)
    prompt = (
        "You are a customer support assistant.\n"
        f"Tool used: {tool_name}\n"
        "Tool output:\n"
        f"{preview}\n\n"
        "Write a short, clear answer for the user. Use bullet points if helpful."
    )
    resp = llm.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


def _extract_tool_traces(agent_messages: List[Any]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for m in agent_messages:
        cls = m.__class__.__name__
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                traces.append({"type": "tool_call", "tool_name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")})
        if cls == "ToolMessage":
            traces.append({"type": "tool_result", "tool_name": getattr(m, "name", None), "content": getattr(m, "content", "")})
    return traces


def chat_reply(messages: List[Dict[str, str]]) -> tuple[str, Dict[str, Any]]:
    user_text = messages[-1]["content"] if messages else ""

    _ensure_tools_preloaded()

    # Tickets are not supported by these tools
    if "ticket" in user_text.lower():
        return (
            "Support tickets are not available in this demo.\n\n"
            "I can help with products and orders.\n"
            "Type /help to see supported commands.",
            {"type": "not_supported", "topic": "tickets", "tools_available": tools_available()},
        )

    # Slash commands => tool-only
    if user_text.strip().startswith("/"):
        route = route_intent(user_text)

        if route and route["tool"] == "__help__":
            return tool_command_guide(), {"type": "help", "tools_available": tools_available()}

        if route and route["tool"] == "__error__":
            return route["args"]["message"], {"type": "error", "message": route["args"]["message"]}

        if route:
            tool = route["tool"]
            args = route["args"]

            # Friendly validation
            if tool in {"support_get_customer", "support_list_orders", "support_create_order"}:
                cid = (args.get("customer_id") or "").strip()
                if cid and not _validate_uuid_maybe(cid):
                    d = COMMAND_DEFS.get(tool)
                    return (
                        "That customer_id does not look like a UUID.\n\n"
                        f"Use:\n{d['usage']}\nExample:\n{d['example']}",
                        {"type": "validation_error", "field": "customer_id", "value": cid},
                    )

            if tool == "support_verify_customer_pin":
                email = (args.get("email") or "").strip()
                pin = str(args.get("pin") or "").strip()
                if not email or not pin:
                    d = COMMAND_DEFS.get(tool)
                    return (
                        f"Missing email or pin.\n\nUse:\n{d['usage']}\nExample:\n{d['example']}",
                        {"type": "validation_error", "field": "email/pin", "value": str(args)},
                    )

            try:
                out = call_tool(tool, args)
            except Exception as e:
                d = COMMAND_DEFS.get(tool)
                hint = f"\n\nTry:\n{d['usage']}\nExample:\n{d['example']}" if d else ""
                return (
                    f"I couldn't run that request: {str(e)}{hint}",
                    {"type": "tool_error", "tool": tool, "error": str(e)},
                )

            # Full output tools
            if tool in FULL_OUTPUT_TOOLS:
                full_text = extract_tool_text(out)
                return (
                    f"```text\n{full_text}\n```",
                    {"type": "tool_only", "route": route, "full_output": True, "tool_output_preview": _safe_preview(out, 800)},
                )

            assistant_text = explain(tool, out)
            return assistant_text, {"type": "tool_only", "route": route, "full_output": False, "tool_output_preview": _safe_preview(out, 800)}

        return (
            "Unknown command. Type /help to see supported commands.",
            {"type": "tool_only", "error": "unknown_command", "tools_available": tools_available()},
        )

    # Non-slash auto-routing (tool-first)
    auto = auto_route_non_slash(user_text)
    if auto:
        # Special flows
        if auto["tool"] == "__need_email_pin_for_orders__":
            d = COMMAND_DEFS["support_verify_customer_pin"]
            return (
                "To check your orders, please verify with your email and PIN.\n\n"
                f"Use:\n{d['usage']}\nExample:\n{d['example']}",
                {"type": "need_email_pin", "tools_available": tools_available()},
            )

        if auto["tool"] == "__need_customer_id_for_orders__":
            d = COMMAND_DEFS["support_list_orders"]
            return (
                "To check your orders, I need your customer_id (UUID).\n\n"
                f"Use:\n{d['usage']}\nExample:\n{d['example']}",
                {"type": "need_customer_id", "tools_available": tools_available()},
            )

        if auto["tool"] == "__verify_then_orders__":
            email = auto["args"]["email"]
            pin = auto["args"]["pin"]

            try:
                verify_out = call_tool("support_verify_customer_pin", {"email": email, "pin": pin})
            except Exception as e:
                d = COMMAND_DEFS["support_verify_customer_pin"]
                return (
                    f"I couldn't verify that email and PIN: {str(e)}\n\nUse:\n{d['usage']}\nExample:\n{d['example']}",
                    {"type": "tool_error", "tool": "support_verify_customer_pin", "error": str(e)},
                )

            customer_id = _extract_customer_id_from_tool_output(verify_out)
            if not customer_id:
                return (
                    "Your email and PIN were checked, but I could not find a customer_id in the response.\n\n"
                    "In this demo, I need a customer_id (UUID) to list orders.",
                    {"type": "verify_ok_but_no_customer_id", "verify_preview": _safe_preview(verify_out, 800)},
                )

            # Now list orders
            try:
                orders_out = call_tool("support_list_orders", {"customer_id": customer_id})
            except Exception as e:
                return (
                    f"Verified. But I couldn't fetch orders: {str(e)}",
                    {"type": "tool_error", "tool": "support_list_orders", "error": str(e)},
                )

            full_text = extract_tool_text(orders_out)
            return (
                f"✅ Verified. Here are your orders:\n\n```text\n{full_text}\n```",
                {
                    "type": "verify_then_orders",
                    "customer_id": customer_id,
                    "verify_preview": _safe_preview(verify_out, 800),
                    "orders_preview": _safe_preview(orders_out, 800),
                },
            )

        # Normal single tool
        tool = auto["tool"]
        args = auto["args"]

        # Validate UUID if needed
        if tool in {"support_get_customer", "support_list_orders", "support_create_order"}:
            cid = (args.get("customer_id") or "").strip()
            if cid and not _validate_uuid_maybe(cid):
                d = COMMAND_DEFS.get(tool)
                return (
                    "That customer_id does not look like a UUID.\n\n"
                    f"Use:\n{d['usage']}\nExample:\n{d['example']}",
                    {"type": "validation_error", "field": "customer_id", "value": cid},
                )

        try:
            out = call_tool(tool, args)
        except Exception as e:
            d = COMMAND_DEFS.get(tool)
            hint = f"\n\nTry:\n{d['usage']}\nExample:\n{d['example']}" if d else ""
            return (
                f"I couldn't run that request: {str(e)}{hint}",
                {"type": "tool_error", "tool": tool, "error": str(e)},
            )

        if tool in FULL_OUTPUT_TOOLS:
            full_text = extract_tool_text(out)
            return (
                f"```text\n{full_text}\n```",
                {"type": "tool_only", "route": auto, "full_output": True, "tool_output_preview": _safe_preview(out, 800)},
            )

        assistant_text = explain(tool, out)
        return assistant_text, {"type": "tool_only", "route": auto, "full_output": False, "tool_output_preview": _safe_preview(out, 800)}

    # Agent mode (non-slash general questions)
    if not settings.groq_api_key or not settings.mcp_server_url:
        return (
            "Set GROQ_API_KEY and MCP_SERVER_URL to enable real answers + tool calls.",
            {"type": "config", "level": "warning", "message": "Missing GROQ_API_KEY or MCP_SERVER_URL."},
        )

    if not USE_AGENT:
        return (
            "Agent mode is disabled. Type /help to see supported commands.",
            {"type": "config", "level": "info", "message": "USE_AGENT is false."},
        )

    try:
        tool_subset = tuple(select_tools_for_text(user_text))
        agent, tool_names = _build_agent_for_tool_subset(tool_subset)

        trimmed = _trim_messages(messages, max_pairs=3)
        result = _run_async(agent.ainvoke({"messages": trimmed}))

        agent_messages = result.get("messages", [])
        assistant_text = agent_messages[-1].content if agent_messages else "Sorry, I got no response."
        assistant_text = clean_tool_syntax(assistant_text)

        tool_traces = _extract_tool_traces(agent_messages)

        debug_event = {
            "type": "agent",
            "agent_factory": _AGENT_FACTORY,
            "model": settings.groq_model,
            "tool_subset_used": list(tool_subset),
            "tools_available_in_subset": tool_names,
            "tool_traces": tool_traces,
            "tools_available": tools_available(),
        }
        return assistant_text, debug_event

    except Exception as e:
        return (
            "I hit an internal error while generating the answer. Please try again.",
            {"type": "error", "message": str(e), "tools_available": tools_available()},
        )