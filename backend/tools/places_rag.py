import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Connect to the local Chroma database
project_root = Path(__file__).resolve().parent.parent.parent
db_path = str(project_root / "backend" / "data" / "chroma_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="langchain")

class RAGSchema(BaseModel):
    query: str = Field(description="Destination or attraction to look up.")

@tool(args_schema=RAGSchema)
def search_local_guides(query: str) -> str:
    """Search offline database for guides and hidden gems."""
    try:
        # Token Optimization: Cap retrieval at top 2 chunks
        docs = vectorstore.similarity_search(query, k=2)
        if not docs:
            return "No offline guide information found."
        return "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"RAG Retrieval Error: {str(e)}"