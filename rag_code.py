# rag_code.py
import os
import re
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Optional: PDF/text/HTML parsing (fallbacks kept lightweight)
try:
    import pdfplumber
except Exception:
    pdfplumber = None
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

QDRANT_VECTOR_PARAMS = VectorParams(size=1536, distance=Distance.COSINE)
CONTENT_PAYLOAD_KEY = "page_content"
METADATA_PAYLOAD_KEY = "metadata"


def build_date_anchors(tz_name: str = None) -> str:
    tz = ZoneInfo(tz_name or os.getenv("LOCAL_TZ", "Asia/Kolkata"))
    today = datetime.now(tz).date()
    anchors = {
        "today": today,
        "tomorrow": today + timedelta(days=1),
        "in_7_days": today + timedelta(days=7),
        "in_14_days": today + timedelta(days=14),
        "in_30_days": today + timedelta(days=30),
        "start_next_week": today + timedelta(days=(7 - today.weekday()) % 7 or 7),
    }
    fmt = lambda d: d.strftime("%d %b %Y")
    lines = "\n".join(f"{k}: {fmt(v)}" for k, v in anchors.items())
    return f"DATE ANCHORS ({tz.key}):\n{lines}"


@dataclass
class RAGConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rag_kb"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "6"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    temp_processing_path: str = field(default_factory=lambda: os.getenv("TEMP_PROCESSING_PATH", "local_rag_data/temp"))

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls()


class VectorStoreManager:
    def __init__(self, config: RAGConfig, qdrant: Optional[QdrantClient] = None):
        self.config = config
        self.client = qdrant or QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30.0)
        self.embeddings = OpenAIEmbeddings(api_key=config.openai_api_key, model=config.embedding_model)

    async def ensure_collection(self, collection: str):
        cols = await asyncio.to_thread(self.client.get_collections)
        names = [c.name for c in cols.collections]
        if collection not in names:
            await asyncio.to_thread(self.client.create_collection, collection, QDRANT_VECTOR_PARAMS)

    def store(self, collection: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            embedding=self.embeddings,
            content_payload_key=CONTENT_PAYLOAD_KEY,
            metadata_payload_key=METADATA_PAYLOAD_KEY
        )

    async def add_documents(self, docs: List[Document], collection: str) -> List[str]:
        await self.ensure_collection(collection)
        return await self.store(collection).aadd_documents(docs)

    async def clear(self, collection: str):
        try:
            await asyncio.to_thread(self.client.delete_collection, collection_name=collection)
        except Exception:
            pass
        await self.ensure_collection(collection)

    async def count(self, collection: str) -> int:
        """Return number of points in a collection. 0 if missing."""
        try:
            info = await asyncio.to_thread(self.client.get_collection, collection)
            # Some client versions expose points_count, others only via dict
            if hasattr(info, "points_count") and info.points_count is not None:
                return int(info.points_count)
            if isinstance(info, dict):
                return int(info.get("points_count", 0))
            # Last resort: try a cheap scroll limit=0 (but we avoid this for performance)
            return 0
        except Exception:
            return 0

    def retriever(self, collection: str, k: int) -> Any:
        return self.store(collection).as_retriever(search_kwargs={"k": k})


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        os.makedirs(config.temp_processing_path, exist_ok=True)

    async def docs_from_memory(self, file_bytes: bytes, file_name: str) -> List[Document]:
        ext = os.path.splitext(file_name)[1].lower()
        meta = {"source": file_name}
        try:
            if ext in [".txt", ".md", ".log", ".rst"]:
                return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"), metadata=meta)]
            if ext in [".html", ".htm", ".xhtml"]:
                soup = BeautifulSoup(file_bytes, "html.parser")
                return [Document(page_content=soup.get_text(separator="\n", strip=True), metadata=meta)]
            if ext in [".json", ".jsonl"]:
                try:
                    txt = file_bytes.decode("utf-8", errors="ignore")
                    data = json.loads(txt)
                    return [Document(page_content=json.dumps(data, indent=2, ensure_ascii=False), metadata=meta)]
                except Exception:
                    return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"), metadata=meta)]
            if ext == ".pdf":
                if pdfplumber is None:
                    return [Document(page_content="[PDF parsing not available]", metadata=meta)]
                try:
                    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                        text = "".join(page.extract_text(x_tolerance=1, y_tolerance=1) or "" for page in pdf.pages)
                    return [Document(page_content=text, metadata=meta)]
                except Exception as e:
                    return [Document(page_content=f"[PDF not parsed: {e}]", metadata=meta)]
            # default: try utf-8
            return [Document(page_content=file_bytes.decode("utf-8", errors="ignore"), metadata=meta)]
        except Exception as e:
            logger.error(f"doc parse failed '{file_name}': {e}")
            return []

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)


class RAGPipeline:
    def __init__(self, config: RAGConfig, qdrant: Optional[QdrantClient] = None):
        self.config = config
        self.vsm = VectorStoreManager(config, qdrant)
        self.dp = DocumentProcessor(config)

    async def initialize(self):
        await self.vsm.ensure_collection(self.config.qdrant_collection_name)

    def session_collection_name(self, session_id: str) -> str:
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', session_id).strip('_') or "default"
        return f"{self.config.qdrant_collection_name}__sess__{safe}"

    async def has_session_documents(self, session_id: str) -> bool:
        """
        Check if the session has any uploaded documents.
        """
        if not session_id:
            return False
        
        try:
            session_collection = self.session_collection_name(session_id)
            count = await self.vsm.count(session_collection)
            return count > 0
        except Exception as e:
            logger.warning(f"Failed to check session documents: {e}")
            return False

    async def get_session_document_list(self, session_id: str) -> List[str]:
        """
        Get a list of source files in the session.
        """
        if not session_id:
            return []
        
        try:
            session_collection = self.session_collection_name(session_id)
            if await self.vsm.count(session_collection) == 0:
                return []
            
            # Get some documents to extract source names
            store = self.vsm.store(session_collection)
            docs = await store.asimilarity_search("", k=20)  # Get many docs to find all sources
            
            sources = set()
            for doc in docs:
                source = (doc.metadata or {}).get("source", "unknown")
                if source != "unknown":
                    sources.add(source)
            
            return list(sources)
        except Exception as e:
            logger.warning(f"Failed to get session document list: {e}")
            return []

    async def gather_evidence(self, query: str, session_id: Optional[str], enable_web_search: bool) -> str:
        """
        Gathers evidence with strong priority for session documents.
        """
        session_hits = []
        kb_hits = []
        web_hits = []
        
        # Helper to search a collection
        async def _search_collection(collection_name: str, min_results: int = 3):
            try:
                if await self.vsm.count(collection_name) > 0:
                    store = self.vsm.store(collection_name)
                    # Get more results initially, then filter
                    results_with_scores = await store.asimilarity_search_with_relevance_scores(query, k=8)
                    return results_with_scores
            except Exception as e:
                logger.warning(f"RAG search failed for '{collection_name}': {e}")
            return []

        # 1. PRIORITY: Search Session documents first (if session exists)
        if session_id:
            session_collection = self.session_collection_name(session_id)
            session_results = await _search_collection(session_collection)
            
            # For session documents, use a lower threshold (0.6) and ensure we get at least some results
            session_threshold = 0.6
            for doc, score in session_results:
                if score >= session_threshold or len(session_hits) < 2:  # Always include top 2 session results
                    session_hits.append((doc, score))
                    if len(session_hits) >= 5:  # Limit session results
                        break

        # 2. Search KB documents (only if we have few session results)
        if len(session_hits) < 3:  # Only search KB if session results are limited
            kb_results = await _search_collection(self.config.qdrant_collection_name)
            kb_threshold = 0.7  # Higher threshold for KB since session has priority
            
            for doc, score in kb_results:
                if score >= kb_threshold:
                    kb_hits.append((doc, score))
                    if len(kb_hits) >= 3:  # Limit KB results when session exists
                        break

        # 3. Web search fallback (only if insufficient local results)
        total_local_hits = len(session_hits) + len(kb_hits)
        if total_local_hits < 2 and enable_web_search:
            try:
                from websearch import WebSearch, WebSearchConfig
                ws = WebSearch(config=WebSearchConfig.from_env())
                web_docs = await ws.smart_search(query)
                web_hits = [(doc, 1.0) for doc in web_docs[:3]]  # Assume web results are relevant
            except Exception as e:
                logger.warning(f"Web fallback failed: {e}")

        # 4. Format evidence block with clear source priority
        parts = []
        seen_content = set()
        
        # Format session documents FIRST and PROMINENTLY
        if session_hits:
            parts.append("=== YOUR UPLOADED DOCUMENTS (PRIORITY) ===")
            for i, (doc, score) in enumerate(session_hits):
                header = (doc.metadata or {}).get("source", "uploaded_document")
                body = (doc.page_content or "").strip()
                
                # Avoid duplicates
                sig = body[:200]
                if sig in seen_content:
                    continue
                seen_content.add(sig)
                
                parts.append(f"[{len(parts)}] (SESSION) {header} (relevance: {score:.2f})\n{body}")
        
        # Then KB documents (if any)
        if kb_hits:
            if session_hits:  # Add separator if we have session docs
                parts.append("\n=== KNOWLEDGE BASE DOCUMENTS ===")
            for i, (doc, score) in enumerate(kb_hits):
                header = (doc.metadata or {}).get("source", "knowledge_base")
                body = (doc.page_content or "").strip()
                
                sig = body[:200]
                if sig in seen_content:
                    continue
                seen_content.add(sig)
                
                parts.append(f"[{len(parts)}] (KB) {header} (relevance: {score:.2f})\n{body}")
        
        # Finally web results (if any)
        if web_hits:
            if session_hits or kb_hits:
                parts.append("\n=== WEB SEARCH RESULTS ===")
            for i, (doc, score) in enumerate(web_hits):
                source_url = (doc.metadata or {}).get("url", "web_source")
                title = (doc.metadata or {}).get("title", "Web Result")
                body = (doc.page_content or "").strip()
                
                sig = body[:200]
                if sig in seen_content:
                    continue
                seen_content.add(sig)
                
                parts.append(f"[{len(parts)}] (WEB) {title} - {source_url}\n{body}")

        # 5. Return evidence block
        if not parts:
            return "=== EVIDENCE START ===\n(No relevant information found in your documents, knowledge base, or web.)\n=== EVIDENCE END ==="
        
        evidence_content = "\n\n---\n\n".join(parts)
        return f"=== EVIDENCE START ===\n{evidence_content}\n=== EVIDENCE END ==="

    async def simple_collection_hits(self, query: str, collection: str) -> str:
        """Simple search within a specific collection."""
        try:
            count = await self.vsm.count(collection)
            if count == 0:
                return ""
            
            store = self.vsm.store(collection)
            docs = await store.asimilarity_search(query, k=5)
            
            if not docs:
                return ""
            
            parts = []
            for i, doc in enumerate(docs):
                source = (doc.metadata or {}).get("source", "unknown")
                content = (doc.page_content or "").strip()
                parts.append(f"[{i+1}] {source}\n{content}")
            
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.warning(f"Collection search failed for '{collection}': {e}")
            return ""

    async def build_evidence_block(self, query: str, session_id: Optional[str], enable_web_search: bool) -> str:
        """Legacy method - redirects to gather_evidence for compatibility."""
        return await self.gather_evidence(query, session_id, enable_web_search)