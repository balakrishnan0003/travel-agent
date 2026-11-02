
# backend/compass.py

import os
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def ingest_youtube_video(video_id: str, video_title: str):
    """Extract a YouTube transcript, split it into chunks, and store it in ChromaDB."""

    try:
        # 1. Fetch transcript
        print(f"Fetching transcript for video: {video_id}")

        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        transcript_list = transcript.to_raw_data()

        full_text = " ".join(
            entry["text"] for entry in transcript_list
        )

        if not full_text.strip():
            raise ValueError("The YouTube transcript is empty.")

        # 2. Add source metadata to the document
        document_text = (
            f"SOURCE: YouTube Video - {video_title}\n"
            f"VIDEO_ID: {video_id}\n"
            f"CONTENT: {full_text}"
        )

        # 3. Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        chunks = splitter.split_text(document_text)

        if not chunks:
            raise ValueError("No text chunks were generated.")

        # 4. Set an absolute database path based on the project root
        project_root = Path(__file__).resolve().parent.parent
        db_path = str(project_root / "backend" / "data" / "chroma_db")

        # 5. Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 6. Connect to the existing ChromaDB
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_name="langchain",
        )

        # 7. Store chunks with source metadata
        metadatas = [
            {
                "source": "youtube",
                "video_id": video_id,
                "video_title": video_title,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
        )

        print(
            f"Successfully added {len(chunks)} chunks "
            f"from '{video_title}' to ChromaDB."
        )

    except Exception as e:
        print(f"Compass Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Replace this placeholder with a real YouTube video ID.
    ingest_youtube_video(
        "o71LTCpCLzc",
        "The Ultimate 3-Day Kyoto, Japan Travel Guide? ",
        # "YOUR_VIDEO_ID",
        # "Ultimate Kyoto Travel Guide",
    )