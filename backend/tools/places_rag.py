import os
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Disable ChromaDB telemetry to prevent DNS hang on Render/cloud
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"

# Paths
project_root = Path(__file__).resolve().parent.parent.parent
db_path      = str(project_root / "backend" / "data" / "chroma_db")

# Lazy singleton — loaded ONLY when first tool call happens
# NOT at import time, so startup RAM stays low on Render free tier (512MB)
_vectorstore = None
 
def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
 
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        _vectorstore = Chroma(
            client=chroma_client,
            collection_name="langchain",
            embedding_function=embeddings,
        )
    return _vectorstore
 


class RAGSchema(BaseModel):
    query: str = Field(description="Destination or attraction to look up.")


@tool(args_schema=RAGSchema)
def search_local_guides(query: str) -> str:
    """Search offline database for guides and hidden gems."""
    try:
        vectorstore = _get_vectorstore()
        # Token Optimisation: Cap retrieval at top 2 chunks
        docs = vectorstore.similarity_search(query, k=2)
        if not docs:
            return "No offline guide information found."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"RAG Retrieval Error: {str(e)}"
