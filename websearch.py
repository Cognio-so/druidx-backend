# websearch.py
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from llm_model import LLMManager, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class WebSearchConfig:
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    analysis_model: str = field(default_factory=lambda: os.getenv("WEBS_ANALYSIS_MODEL", "gpt-4o-mini"))
    max_results: int = int(os.getenv("WEBS_MAX_RESULTS", "5"))
    include_raw_content: bool = True
    search_depth: str = os.getenv("WEBS_DEPTH", "advanced")

    @classmethod
    def from_env(cls) -> "WebSearchConfig":
        return cls()


# websearch.py

class WebSearch:
    def __init__(self, config: WebSearchConfig, llm_manager: Optional[LLMManager] = None):
        self.cfg = config
        self.tavily = AsyncTavilyClient(api_key=self.cfg.tavily_api_key) if (self.cfg.tavily_api_key and TAVILY_AVAILABLE) else None
        self.llm = llm_manager or LLMManager(LLMConfig.from_env())

    def is_available(self) -> bool:
        return self.tavily is not None

    async def smart_search(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """Performs a web search and returns a simple list of Document objects."""
        if not self.tavily:
            return []
        try:
            params = {
                "query": query,
                "search_depth": self.cfg.search_depth,
                "max_results": max_results or self.cfg.max_results,
                "include_raw_content": self.cfg.include_raw_content,
                "include_answer": False 
            }
            resp = await self.tavily.search(**params)
            
            docs = []
            for r in resp.get("results", []):
                content = r.get("content") or r.get("raw_content") or ""
                # Truncate content to a reasonable size for the context window
                content = content[:3000]

                meta = {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "source": "web_search",
                }
                if content:
                    docs.append(Document(page_content=content, metadata=meta))
            return docs
            
        except Exception as e:
            logger.warning(f"web search failed: {e}")
            return []

    async def cleanup(self):
        return