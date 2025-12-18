import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    mcp_server_url: str | None = os.getenv("MCP_SERVER_URL")

    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature: float = float(os.getenv("TEMPERATURE", "0"))

settings = Settings()