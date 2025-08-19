# main_app.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
import logging
import signal
import sys
import re

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Core imports ---
try:
    from agent import ChatbotAgent, ChatbotConfig
    AGENT_AVAILABLE = True
    logger.info("✅ ChatbotAgent imported successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    logger.error(f"❌ Failed to import ChatbotAgent: {e}. Ensure agent.py is present.")

try:
    from storage import CloudflareR2Storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.error("❌ storage.py not found.")

try:
    from llm_model import LLMManager, LLMConfig
    from qdrant_client import QdrantClient
    from rag_code import RAGConfig
    LLM_MODEL_AVAILABLE = True
    QDRANT_AVAILABLE = True
except ImportError as e:
    LLM_MODEL_AVAILABLE = False
    QDRANT_AVAILABLE = False
    logger.error(f"❌ Failed to import core clients: {e}")

# --- Globals ---
active_agents: Dict[str, ChatbotAgent] = {}
agents_lock = asyncio.Lock()
r2_storage: Optional[CloudflareR2Storage] = None
global_llm_manager: Optional[LLMManager] = None
global_qdrant_client: Optional[QdrantClient] = None

# --- Helpers ---
def get_session_id(user_id: str, gpt_id: str) -> str:
    safe_user = user_id.replace("@", "_").replace(".", "_")
    return f"user_{safe_user}_gpt_{gpt_id}"

def sanitize_for_collection_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    return f"gpt_collection_{s or 'default'}"

async def get_or_create_agent(
    user_id: str,
    gpt_id: str,
    gpt_name: Optional[str] = "default_gpt",
    api_keys: Optional[Dict[str, str]] = None,
    force_recreate: bool = False,
    **config_overrides
) -> ChatbotAgent:
    """Singleton-per (user,gpt) with shared R2/LLM/Qdrant clients."""
    global r2_storage, global_llm_manager, global_qdrant_client

    qdrant_collection_name = sanitize_for_collection_name(gpt_id)
    agent_key = f"{user_id}_{gpt_id}"

    existing = active_agents.get(agent_key)
    if existing and existing.rag_pipeline.config.qdrant_collection_name != qdrant_collection_name:
        force_recreate = True

    async with agents_lock:
        if agent_key in active_agents and not force_recreate:
            return active_agents[agent_key]
        if agent_key in active_agents and force_recreate:
            old = active_agents.pop(agent_key)
            asyncio.create_task(old.cleanup())

    try:
        cfg = ChatbotConfig.from_env()
        if api_keys:
            cfg.openai_api_key = api_keys.get("openai", cfg.openai_api_key)

        # allow overrides
        config_overrides["qdrant_collection_name"] = qdrant_collection_name
        for k, v in config_overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

        agent = ChatbotAgent(
            config=cfg,
            storage_client=r2_storage,
            llm_manager=global_llm_manager,
            qdrant_client=global_qdrant_client,
        )
        await agent.initialize()

        async with agents_lock:
            active_agents[agent_key] = agent
        return agent
    except Exception as e:
        logger.error(f"❌ Failed to create ChatbotAgent ({agent_key}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create chatbot agent: {str(e)}")

async def process_uploaded_file_to_r2(file: UploadFile, is_user_doc: bool) -> "FileUploadInfoResponse":
    global r2_storage
    if not r2_storage or r2_storage.use_local_fallback:
        msg = "R2 storage is not available or in fallback mode."
        return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=msg)

    try:
        content = await file.read()
        success, url_or_err = await asyncio.to_thread(
            r2_storage.upload_file,
            file_data=content,
            filename=file.filename,
            is_user_doc=is_user_doc
        )
        if success:
            return FileUploadInfoResponse(filename=file.filename, stored_url_or_key=url_or_err, status="success")
        return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=url_or_err)
    except Exception as e:
        return FileUploadInfoResponse(filename=file.filename, stored_url_or_key="", status="failure", error_message=str(e))

# --- Lifecycle ---
async def cleanup_r2_expired_files():
    global r2_storage
    logger.info("🧹 R2 cleanup tick...")
    if r2_storage and not r2_storage.use_local_fallback:
        try:
            deleted = await asyncio.to_thread(r2_storage.check_and_delete_expired_files)
            logger.info(f"R2 cleanup: deleted={deleted}")
        except Exception as e:
            logger.error(f"R2 cleanup error: {e}")
    else:
        logger.info("R2 not available or in fallback; skipping cleanup.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global r2_storage, global_llm_manager, global_qdrant_client
    logger.info("🚀 Starting FastAPI...")

    # R2
    if STORAGE_AVAILABLE:
        try:
            r2_storage = CloudflareR2Storage()
            if r2_storage.use_local_fallback:
                logger.warning("R2 initialized in local fallback mode.")
            else:
                logger.info("R2 initialized.")
        except Exception as e:
            logger.error(f"R2 init failed: {e}")
            r2_storage = None

    # LLM
    if LLM_MODEL_AVAILABLE:
        try:
            # --- THIS IS THE FIX ---
            # Instantiate LLMConfig directly instead of using .from_env()
            global_llm_manager = LLMManager(LLMConfig())
            logger.info("Global LLM Manager ready.")
        except Exception as e:
            logger.error(f"LLM Manager init failed: {e}")
    # Qdrant
    if QDRANT_AVAILABLE:
        try:
            rag_cfg = RAGConfig.from_env()
            global_qdrant_client = QdrantClient(
                url=rag_cfg.qdrant_url,
                api_key=rag_cfg.qdrant_api_key,
                timeout=30.0
            )
            await asyncio.to_thread(global_qdrant_client.get_collections)
            logger.info("Qdrant warmed.")
        except Exception as e:
            logger.error(f"Qdrant init failed: {e}")

    # Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(cleanup_r2_expired_files, "interval", hours=6)
    scheduler.start()
    logger.info("⏰ Scheduler started (6h)")

    yield

    logger.info("🛑 Shutting down...")
    scheduler.shutdown(wait=False)
    if global_llm_manager:
        await global_llm_manager.cleanup()
    if global_qdrant_client:
        global_qdrant_client = None

    async with agents_lock:
        tasks = [a.cleanup() for a in active_agents.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        active_agents.clear()
        logger.info("Agents cleaned.")

def setup_signal_handlers():
    def _h(signum, frame):
        logger.info(f"🛑 Received signal {signum}, exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _h)
    signal.signal(signal.SIGTERM, _h)

# --- App ---
app = FastAPI(
    title="Enhanced Chatbot API (Qdrant RAG + LangGraph)",
    description="FastAPI layer over ChatbotAgent using rag_code/websearch/storage stack and LangGraph orchestration.",
    version="2.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://my-gpt-frontend.vercel.app",
        "https://www.druidx.co",
        "https://www.mygpt.work",
        "https://www.EMSA.co",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

setup_signal_handlers()

# --- Schemas ---
class BaseAgentRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: Optional[str] = "default_gpt"

class ChatPayload(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    user_document_keys: Optional[List[str]] = Field([], alias="user_documents")  # deprecated
    use_hybrid_search: Optional[bool] = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    web_search_enabled: Optional[bool] = False
    mcp_enabled: Optional[bool] = False
    mcp_schema: Optional[str] = None
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

class ChatStreamRequest(BaseAgentRequest, ChatPayload): ...
class ChatRequest(BaseAgentRequest, ChatPayload): ...

class GptContextSetupRequest(BaseAgentRequest):
    kb_document_urls: Optional[List[str]] = []
    default_model: Optional[str] = None
    default_system_prompt: Optional[str] = None
    default_use_hybrid_search: Optional[bool] = False
    mcp_enabled_config: Optional[bool] = Field(None, alias="mcpEnabled")
    mcp_schema_config: Optional[str] = Field(None, alias="mcpSchema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

class FileUploadInfoResponse(BaseModel):
    filename: str
    stored_url_or_key: str
    status: str
    error_message: Optional[str] = None

class GptOpenedRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: str
    file_urls: List[str] = []
    use_hybrid_search: bool = False
    web_search_enabled: bool = True
    config_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

# --- Routes ---
@app.get("/", include_in_schema=False)
async def root_redirect():
    return JSONResponse(content={"message": "Enhanced Chatbot API is running. Visit /docs for details."})

@app.get("/health", tags=["Monitoring"])
async def health_check():
    storage_healthy = bool(r2_storage and not r2_storage.use_local_fallback)
    return {"status": "healthy", "timestamp": time.time(), "agent_available": AGENT_AVAILABLE, "storage_healthy": storage_healthy}

@app.post("/setup-gpt-context", tags=["Agent Setup"])
async def setup_gpt_context_endpoint(request: GptContextSetupRequest, background_tasks: BackgroundTasks):
    logger.info(f"🔧 Setup for GPT {request.gpt_id}")
    try:
        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=True,
            default_model=request.default_model,
            default_system_prompt=request.default_system_prompt,
        )

        async def _proc_kb(urls: List[str], a: ChatbotAgent):
            if urls:
                logger.info(f"📚 Indexing {len(urls)} KB URLs in background...")
                await a.add_documents_to_knowledge_base(urls)
                logger.info("✅ KB update finished.")

        if request.kb_document_urls:
            background_tasks.add_task(_proc_kb, request.kb_document_urls, agent)
            msg = "Agent context update initiated in background."
        else:
            msg = "Agent context initialized/updated."

        collection_name = agent.rag_pipeline.config.qdrant_collection_name if agent.rag_pipeline else None
        return JSONResponse(content={"success": True, "message": msg, "collection_name": collection_name})
    except Exception as e:
        logger.error(f"setup-gpt-context failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to setup GPT context: {str(e)}")

@app.post("/upload-documents", tags=["Documents"])
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    is_user_document: str = Form("false"),
):
    is_user = is_user_document.lower() == "true"
    tasks = [process_uploaded_file_to_r2(f, is_user) for f in files]
    results = await asyncio.gather(*tasks)
    urls = [r.stored_url_or_key for r in results if r.status == "success"]

    if not urls:
        return JSONResponse(status_code=400, content={"message": "No files were successfully uploaded.", "upload_results": [r.model_dump() for r in results]})

    agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)
    session_id = get_session_id(user_id, gpt_id)

    async def _index(a: ChatbotAgent, u: List[str], is_user_doc: bool, s_id: str):
        if is_user_doc:
            logger.info(f"👤 Indexing {len(u)} user docs for session {s_id}...")
            await a.add_user_documents_for_session(s_id, u)
        else:
            logger.info(f"📚 Indexing {len(u)} KB docs...")
            await a.add_documents_to_knowledge_base(u)

    background_tasks.add_task(_index, agent, urls, is_user, session_id)
    msg = f"{len(urls)} {'user' if is_user else 'KB'} files accepted and are being indexed."
    return JSONResponse(status_code=202, content={"message": msg, "upload_results": [r.model_dump() for r in results]})

# main_app.py -> inside the chat_stream endpoint

@app.post("/chat-stream", summary="Streaming chat with agent", tags=["Chat"])
async def chat_stream(request: ChatStreamRequest):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=bool(request.api_keys),
        )

        # --- ADD THIS BLOCK TO PROCESS HISTORY ---
        from langchain_core.messages import HumanMessage, AIMessage
        history_messages = []
        if request.history:
            for msg in request.history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    history_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                     history_messages.append(AIMessage(content=content))
        # --- END OF BLOCK ---

        async def generate():
            try:
                # Update the agent.query_stream call to pass the history
                async for chunk in agent.query_stream(
                    session_id=session_id,
                    query=request.message,
                    history=history_messages,  # <-- PASS THE PROCESSED HISTORY
                    enable_web_search=request.web_search_enabled,
                    model_override=request.model,
                    system_prompt_override=request.system_prompt
                ):
                    if chunk and isinstance(chunk, dict):
                        sse_payload = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield sse_payload
            except Exception as e:
                logger.error(f"❌ Error during stream generation: {e}", exc_info=True)
                error_payload = {"type": "error", "data": str(e)}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)

    except Exception as e:
        logger.error(f"❌ Error in chat stream endpoint: {e}", exc_info=True)
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'data': f'Failed to start chat stream: {str(e)}'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)
    
@app.post("/chat", tags=["Chat"])
async def chat_endpoint(request: "ChatRequest"):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        agent = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys,
            force_recreate=bool(request.api_keys)
        )
        resp = await agent.query(
            session_id=session_id,
            query=request.message,
            enable_web_search=request.web_search_enabled,
            model_override=request.model,
            system_prompt_override=request.system_prompt,
            # mcp_enabled=request.mcp_enabled,
            # mcp_schema=request.mcp_schema,
            # api_keys=request.api_keys
        )
        return JSONResponse(content=resp)
    except Exception as e:
        logger.error(f"chat failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {str(e)}"})

@app.post("/gpt-opened", tags=["Init"])
async def gpt_opened_endpoint(request: "GptOpenedRequest", background_tasks: BackgroundTasks):
    session_id = get_session_id(request.user_id, request.gpt_id)
    try:
        agent_instance = await get_or_create_agent(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=(request.config_schema or {}).get("model"),
            default_system_prompt=(request.config_schema or {}).get("instructions"),
            api_keys=request.api_keys,
            force_recreate=bool(request.api_keys)
        )

        if request.file_urls:
            async def _proc(a: ChatbotAgent, urls: List[str]):
                try:
                    await a.add_documents_to_knowledge_base(urls)
                except Exception as e:
                    logger.error(f"KB doc processing failed: {e}", exc_info=True)
            background_tasks.add_task(_proc, agent_instance, request.file_urls)

        collection_name = agent_instance.rag_pipeline.config.qdrant_collection_name if agent_instance.rag_pipeline else None
        mcp_enabled = (request.config_schema or {}).get("mcpEnabled", False)
        mcp_schema = (request.config_schema or {}).get("mcpSchema")

        return JSONResponse(content={
            "success": True,
            "message": f"GPT '{request.gpt_name}' context ready.",
            "collection_name": collection_name,
            "session_id": session_id,
            "mcp_config_loaded": {"enabled": mcp_enabled, "schema_present": bool(mcp_schema)}
        })
    except Exception as e:
        logger.error(f"gpt-opened failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to handle GPT opened event: {str(e)}")

@app.post("/upload-chat-files", tags=["Documents"])
async def upload_chat_files_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    tasks = [process_uploaded_file_to_r2(f, is_user_doc=True) for f in files]
    results = await asyncio.gather(*tasks)
    urls = [r.stored_url_or_key for r in results if r.status == "success"]

    if not urls:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "No files were successfully uploaded.",
            "file_urls": [],
            "processing_results": [r.model_dump() for r in results]
        })

    agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)
    session_id = get_session_id(user_id, gpt_id)

    async def _index(a: ChatbotAgent, s_id: str, u: List[str]):
        await a.add_user_documents_for_session(s_id, u)

    background_tasks.add_task(_index, agent, session_id, urls)
    return JSONResponse(status_code=202, content={
        "success": True,
        "message": f"Processed {len(urls)} files. Indexing in background for this session.",
        "file_urls": urls,
        "processing_results": [r.model_dump() for r in results]
    })

@app.post("/index-knowledge", tags=["Documents"])
async def index_knowledge_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        gpt_id = data.get("gpt_id")
        user_id = data.get("user_id", "default_user")
        file_urls = data.get("file_urls", [])
        if not gpt_id or not file_urls:
            raise HTTPException(status_code=400, detail="gpt_id and file_urls are required.")
        agent = await get_or_create_agent(user_id=user_id, gpt_id=gpt_id)

        async def _proc(urls: List[str], a: ChatbotAgent):
            await a.add_documents_to_knowledge_base(urls)

        background_tasks.add_task(_proc, file_urls, agent)
        return {"success": True, "message": f"Indexing started for {len(file_urls)} files."}
    except Exception as e:
        logger.error(f"index-knowledge failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/dev/reset-gpt-context", tags=["Development"])
async def dev_reset_gpt_context_endpoint(gpt_id: str = Form(...), user_id: str = Form(...)):
    if os.getenv("ENVIRONMENT_TYPE", "production").lower() != "development":
        raise HTTPException(status_code=403, detail="Endpoint only available in development.")
    agent_key = f"{user_id}_{gpt_id}"
    async with agents_lock:
        if agent_key in active_agents:
            agent = active_agents.pop(agent_key)
            await agent.cleanup()
            return {"status": "success", "message": f"Agent context for '{agent_key}' cleared from memory."}
        return JSONResponse(status_code=404, content={"status": "not_found", "message": f"No active agent context for '{agent_key}'."})

@app.post("/maintenance/cleanup-r2", tags=["Maintenance"])
async def manual_cleanup_r2():
    try:
        await cleanup_r2_expired_files()
        return {"status": "success", "message": "R2 cleanup triggered."}
    except Exception as e:
        logger.error(f"manual cleanup failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    if not AGENT_AVAILABLE:
        logger.critical("❌ ChatbotAgent is not available. Cannot start.")
        sys.exit(1)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=os.getenv("ENVIRONMENT_TYPE", "production").lower() == "development"
    )
