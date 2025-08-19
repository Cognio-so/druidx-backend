# agent.py
import os
import re
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Annotated, Tuple

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool

from rag_code import RAGPipeline, RAGConfig, build_date_anchors
from websearch import WebSearch, WebSearchConfig
from llm_model import LLMManager, LLMConfig
from storage import CloudflareR2Storage

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(name: str): # type: ignore
        return lambda f: f
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Tracing will be disabled. To enable, run: pip install langsmith")


logger = logging.getLogger(__name__)

@dataclass
class ChatbotConfig:
    """Configuration for the Chatbot (OpenRouter key removed)."""
    # LLM API Keys
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))

    # Model and Prompt Settings
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o"))
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
    default_system_prompt: Optional[str] = field(default_factory=lambda: os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a careful research assistant. Use evidence from tools; cite [1], [2], ... when referencing facts."
    ))
    local_tz: str = field(default_factory=lambda: os.getenv("LOCAL_TZ", "Asia/Kolkata"))

    # Web search
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Qdrant / RAG
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "rag_kb"))

    @classmethod
    def from_env(cls) -> "ChatbotConfig":
        return cls()
        
    def to_llm_config(self) -> "LLMConfig":
        """Helper to create an LLMConfig from this agent config."""
        return LLMConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            google_api_key=self.google_api_key,
            groq_api_key=self.groq_api_key,
            default_model=self.default_model,
            temperature=self.temperature,
        )


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ChatbotAgent:
    """FastAPI Agent using LangGraph with multi-model support."""

    CORE_BEHAVIOR = (
        "CORE BEHAVIOR:\n"
        "- PRIORITY ORDER: 1) USER UPLOADED DOCUMENTS (SESSION) - ALWAYS CHECK FIRST, 2) Knowledge Base, 3) Web search.\n"
        "- If user asks about 'uploaded doc', 'my document', 'session file', 'CV', or similar, ONLY use SESSION documents.\n"
        "- If SESSION documents exist and are relevant, answer PRIMARILY from them unless user specifically asks for KB or web info.\n"
        "- When SESSION documents are found, prominently mention 'Based on your uploaded document:' in your response.\n"
        "- Never say 'no session file loaded' if any SESSION documents exist in the evidence block.\n"
        "- If user asks for document summary without specifying source, prioritize SESSION documents over KB.\n"
        "- Only use KB documents when: a) No relevant SESSION docs exist, or b) User specifically asks for KB info.\n"
        "- Only use web search when: a) No relevant local documents exist, or b) User specifically requests current/web info.\n"
        "- When multiple sources exist, clearly distinguish: 'From your uploaded documents:', 'From knowledge base:', 'From web:'\n"
        "\n"
        "OUTPUT STYLE:\n"
        "- Always start by identifying the source: 'Based on your uploaded document:', 'From the knowledge base:', etc.\n"
        "- If using 'web search' for answering user query, output the answer in more than [2000] words, with proper heading and subheading\n"
        "- Cite non-trivial facts with [1], [2], ... referencing items in the EVIDENCE block.\n"
        "- At the end of your response, you MUST include a '## Sources' section and list the full source for each citation you used.\n"
        "- If a field truly isn't present, write 'Not stated in documents' (do not guess or hallucinate).\n"
        "- Always mention the tool/source used to answer the query at the top of your response.\n"
    )
    DATE_POLICY = (
        "DATE RULES:\n"
        "- Convert relative phrases (today/tomorrow/next week/in N days) to absolute dates using DATE ANCHORS.\n"
        "- Use 'DD Mon YYYY' or month ranges like 'Est. Dec 2025 – Jan 2026'.\n"
    )

    def __init__(
        self,
        config: ChatbotConfig,
        storage_client: Optional[CloudflareR2Storage] = None,
        llm_manager: Optional[LLMManager] = None,
        qdrant_client: Any = None,
    ):
        self.config = config
        self.llm_manager = llm_manager or LLMManager(self.config.to_llm_config())
        self.qdrant_client = qdrant_client
        self.storage = storage_client
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.web: Optional[WebSearch] = None

        if LANGSMITH_AVAILABLE and self.config.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.config.langsmith_api_key
            project_name = os.getenv("LANGCHAIN_PROJECT", "My-Chatbot-Agent")
            os.environ["LANGCHAIN_PROJECT"] = project_name
            logger.info(f"✅ LangSmith tracing enabled for project: '{project_name}'")


    async def initialize(self):
        rconf = RAGConfig.from_env()
        rconf.openai_api_key = self.config.openai_api_key
        rconf.qdrant_url = self.config.qdrant_url
        rconf.qdrant_api_key = self.config.qdrant_api_key
        rconf.qdrant_collection_name = self.config.qdrant_collection_name

        self.rag_pipeline = RAGPipeline(rconf, qdrant=self.qdrant_client)
        await self.rag_pipeline.initialize()

        self.web = WebSearch(WebSearchConfig.from_env(), llm_manager=self.llm_manager)

    async def _fetch_bytes(self, url_or_key: str) -> Tuple[Optional[bytes], str]:
        try:
            if re.match(r"^https?://", url_or_key, re.I):
                import httpx
                async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                    r = await client.get(url_or_key)
                    r.raise_for_status()
                    from urllib.parse import urlparse
                    fn = os.path.basename(urlparse(url_or_key).path) or "downloaded.data"
                    return r.content, fn
            if self.storage:
                content = self.storage.get_file_content_bytes(url_or_key)
                fn = os.path.basename(url_or_key)
                return content, (fn or "file.data")
        except Exception as e:
            logger.warning(f"_fetch_bytes failed for {url_or_key}: {e}")
        return None, "file.data"

    async def add_documents_to_knowledge_base(self, urls_or_keys: List[str]):
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized.")
        docs_all: List = []
        for item in urls_or_keys:
            data, fname = await self._fetch_bytes(item)
            if not data:
                logger.warning(f"Skipping (no data): {item}")
                continue
            docs = await self.rag_pipeline.dp.docs_from_memory(data, fname)
            docs_all.extend(docs)
        if not docs_all:
            return
        splits = self.rag_pipeline.dp.split(docs_all)
        await self.rag_pipeline.vsm.add_documents(splits, self.rag_pipeline.config.qdrant_collection_name)

    async def add_user_documents_for_session(self, session_id: str, urls_or_keys: List[str]):
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized.")
        target = self.rag_pipeline.session_collection_name(session_id)
        docs_all: List = []
        for item in urls_or_keys:
            data, fname = await self._fetch_bytes(item)
            if not data:
                logger.warning(f"Skipping (no data): {item}")
                continue
            docs = await self.rag_pipeline.dp.docs_from_memory(data, fname)
            for d in docs:
                meta = d.metadata or {}
                meta["origin"] = "SESSION"
                meta["source"] = meta.get("source") or fname
                d.metadata = meta
            docs_all.extend(docs)
        if not docs_all:
            return
        splits = self.rag_pipeline.dp.split(docs_all)
        await self.rag_pipeline.vsm.add_documents(splits, target)

    async def _summarize_messages(self, messages_to_summarize: Sequence[BaseMessage]):
        if not messages_to_summarize:
            return None
        
        logger.info(f"Summarizing {len(messages_to_summarize)} old messages...")
        
        conversation_str = ""
        for msg in messages_to_summarize:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation_str += f"{role}: {msg.content}\n"
            
        summarization_prompt = (
            "You are a conversation summarizer. Create a concise, third-person summary of the following chat history.\n\n"
            f"--- CONVERSATION TRANSCRIPT ---\n{conversation_str.strip()}"
            "\n--- END OF TRANSCRIPT ---\n\n"
            "Provide your summary:"
        )
        
        try:
            summary_content = await self.llm_manager.generate_response(
                messages=[{"role": "user", "content": summarization_prompt}],
                model="gpt-4o-mini",
                stream=False
            )
            return SystemMessage(content=f"Summary of previous conversation: {summary_content}")
        except Exception as e:
            logger.error(f"Could not summarize conversation: {e}")
            return SystemMessage(content="A summary of the previous conversation is unavailable due to an error.")

    def _make_tools(self, session_id: Optional[str], enable_web: bool):
        rag = self.rag_pipeline
        web = self.web

        async def retrieve_auto(query: str) -> str:
            return await rag.gather_evidence(
                query=query, session_id=session_id, enable_web_search=enable_web
            )

        async def knowledge_base_retriever_tool(query: str) -> str:
            result = await rag.simple_collection_hits(
                query=query, collection=rag.config.qdrant_collection_name
            )
            if result.strip():
                return f"KNOWLEDGE BASE RESULTS:\n{result}"
            return "No relevant knowledge base documents found."

        async def session_retriever_tool(query: str) -> str:
            if not session_id:
                return "No session ID provided."
            
            # Check if session has documents
            session_collection = rag.session_collection_name(session_id)
            try:
                doc_count = await rag.vsm.count(session_collection)
                if doc_count == 0:
                    return "No documents have been uploaded to this session."
                
                # Get session document list
                sources = await rag.get_session_document_list(session_id)
                
                result = await rag.simple_collection_hits(
                    query=query, collection=session_collection
                )
                
                if result.strip():
                    source_list = ", ".join(sources) if sources else "uploaded documents"
                    return f"SESSION DOCUMENTS ({source_list}):\n{result}"
                else:
                    source_list = ", ".join(sources) if sources else "your uploaded documents"
                    return f"No relevant content found in {source_list} for this query."
                    
            except Exception as e:
                logger.error(f"Session retriever error: {e}")
                return f"Error accessing session documents: {str(e)}"

        tools: List[StructuredTool] = [
            StructuredTool.from_function(
                func=knowledge_base_retriever_tool,
                name="knowledge_base_retriever_tool",
                description="Search the knowledge base documents. Use this when user asks specifically about KB or when no session documents are relevant.",
            ),
            StructuredTool.from_function(
                func=session_retriever_tool,
                name="session_retriever_tool",
                description="Search documents uploaded by the user in this session. Use this FIRST when user asks about 'uploaded docs', 'my documents', 'CV', or document summaries.",
            ),
        ]

        if enable_web and web and web.is_available():
            async def web_search_tool(query: str) -> str:
                result = await web.smart_search(query)
                if isinstance(result, list) and result:
                    # Format as string if it's a list of documents
                    formatted_results = []
                    for i, doc in enumerate(result):
                        title = doc.metadata.get("title", "Web Result")
                        url = doc.metadata.get("url", "")
                        content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        formatted_results.append(f"[{i+1}] {title}\nURL: {url}\n{content}")
                    return "WEB SEARCH RESULTS:\n" + "\n\n".join(formatted_results)
                elif isinstance(result, dict):
                    return result.get("search_results", "No web results found.")
                return "No web results found."

            tools.append(
                StructuredTool.from_function(
                    func=web_search_tool,
                    name="web_search_tool",
                    description="Search the web for current information. Use only when local documents don't have the needed information.",
                )
            )
        
        logging.info(f"Tools configured: {[tool.name for tool in tools]}")
        return tools

    def _compile_graph(self, system_prompt: str, model_name: str, temperature: float, tools):
        norm_model = model_name or self.config.default_model
        fallback_model = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

        def build_llm(m: str) -> BaseChatModel:
            llm = self.llm_manager.get_llm(
                model_name=m,
                temperature=temperature
            )
            return llm.bind_tools(tools)

        llm = build_llm(norm_model)

        def model_call(state: AgentState):
            msgs = list(state["messages"])
            prompt_messages = [
                SystemMessage(content=self.CORE_BEHAVIOR),
                SystemMessage(content=self.DATE_POLICY),
                SystemMessage(content=system_prompt or ""),
            ] + msgs
            try:
                resp = llm.invoke(prompt_messages)
                return {"messages": [resp]}
            except Exception as e:
                msg = str(e).lower()
                logger.warning(f"Model call for '{norm_model}' failed with error: {e}. Trying fallback model '{fallback_model}'.")
                if any(err in msg for err in ["model_not_found", "does not exist", "unknown model", "rate limit", "overloaded", "invalid api key"]):
                    fb_llm = build_llm(fallback_model)
                    resp = fb_llm.invoke(prompt_messages)
                    return {"messages": [resp]}
                raise

        def decide_next_step(state: AgentState):
            last = state["messages"][-1]
            return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

        gb = StateGraph(AgentState)
        gb.add_node("llm_decision", model_call)
        gb.add_node("tools", ToolNode(tools))
        gb.set_entry_point("llm_decision")
        gb.add_conditional_edges("llm_decision", decide_next_step, {"tools": "tools", END: END})
        gb.add_edge("tools", "llm_decision")
        return gb.compile()

    @traceable(name="agent_query")
    async def query(
        self,
        session_id: str,
        query: str,
        history: List[BaseMessage],
        enable_web_search: bool = False,
        model_override: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        date_block = build_date_anchors(self.config.local_tz)
        evidence_block = await self.rag_pipeline.gather_evidence(query, session_id, enable_web_search)

        processed_history = []
        if len(history) > 8:
            summary_message = await self._summarize_messages(history[:-4])
            if summary_message:
                processed_history.append(summary_message)
            processed_history.extend(history[-4:])
        else:
            processed_history = history

        run_messages: List[BaseMessage] = [
            SystemMessage(content=date_block),
            SystemMessage(content=evidence_block),
            *processed_history,
            HumanMessage(content=query)
        ]

        tools = self._make_tools(session_id, enable_web_search)
        app = self._compile_graph(
            system_prompt_override or (self.config.default_system_prompt or ""),
            model_override or self.config.default_model,
            self.config.temperature,
            tools
        )

        result = await app.ainvoke({"messages": run_messages}, config={"recursion_limit": 25})
        final_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        answer = final_ai.content if final_ai else "Could not determine an answer."

        return {"answer": answer, "evidence": evidence_block}

    @traceable(name="agent_query_stream")
    async def query_stream(
        self,
        session_id: str,
        query: str,
        history: List[BaseMessage],
        enable_web_search: bool = False,
        model_override: Optional[str] = None,
        system_prompt_override: Optional[str] = None,
    ):
        date_block = build_date_anchors(self.config.local_tz)
        evidence_block = await self.rag_pipeline.gather_evidence(query, session_id, enable_web_search)
        
        processed_history = []
        if len(history) > 10:
            summary_message = await self._summarize_messages(history[:-4])
            if summary_message:
                processed_history.append(summary_message)
            processed_history.extend(history[-4:])
        else:
            processed_history = history

        run_messages: List[BaseMessage] = [
            SystemMessage(content=date_block),
            SystemMessage(content=evidence_block),
            *processed_history,
            HumanMessage(content=query)
        ]

        tools = self._make_tools(session_id, enable_web_search)
        app = self._compile_graph(
            system_prompt_override or (self.config.default_system_prompt or ""),
            model_override or self.config.default_model,
            self.config.temperature,
            tools
        )
        
        streamed_anything = False
        final_answer = None

        async for ev in app.astream_events({"messages": run_messages}, version="v2", config={"recursion_limit": 25}):
            if ev["event"] == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                if chunk and (tok := getattr(chunk, "content", None)) is not None:
                    streamed_anything = True
                    yield {"type": "delta", "data": tok}
            
            elif ev["event"] == "on_tool_start":
                tool_name = ev["data"].get("name")
                logger.info(f"--- 🛠️ TOOL CALL: `{tool_name}` ---")
                yield {"type": "tool_call", "data": {"name": tool_name}}

            elif ev["event"] == "on_chain_end" and not streamed_anything:
                output = ev["data"].get("output", {})
                if output and "messages" in output:
                    final_ai = next((m for m in reversed(output["messages"]) if isinstance(m, AIMessage)), None)
                    if final_ai:
                        final_answer = final_ai.content
        
        if final_answer:
            yield {"type": "delta", "data": final_answer}
            
    async def cleanup(self):
        return