import os
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

# Disable ChromaDB telemetry to prevent DNS hang on Render/cloud
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"]     = "false"

# Paths
project_root = Path(__file__).resolve().parent.parent.parent
db_path      = str(project_root / "backend" / "data" / "chroma_db")

# Embeddings (CPU-only, no GPU required on cloud) 
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ChromaDB client with telemetry disabled
chroma_client = chromadb.PersistentClient(
    path=db_path,
    settings=Settings(anonymized_telemetry=False),
)

vectorstore = Chroma(
    client=chroma_client,
    collection_name="langchain",
    embedding_function=embeddings,
)


class RAGSchema(BaseModel):
    query: str = Field(description="Destination or attraction to look up.")


@tool(args_schema=RAGSchema)
def search_local_guides(query: str) -> str:
    """Search offline database for guides and hidden gems."""
    try:
        # Token Optimisation: Cap retrieval at top 2 chunks
        docs = vectorstore.similarity_search(query, k=2)
        if not docs:
            return "No offline guide information found."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"RAG Retrieval Error: {str(e)}"
