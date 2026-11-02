from langchain_core.tools import tool
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS


class SearchSchema(BaseModel):
    query: str = Field(description="Search query for hotels and events.")


@tool(args_schema=SearchSchema)
def search_local_events_and_hotels(query: str) -> str:
    """Search web for events and hotels."""
    try:
        with DDGS() as ddgs:
            # Strictly limit to top 3 results
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."

            formatted = []
            for r in results:
                # Token Optimization: Truncate snippet to 150 characters
                snippet = r['body'][:150] + "..." if len(r['body']) > 150 else r['body']
                formatted.append(f"Title: {r['title']}\nSnippet: {snippet}\nSource: {r['href']}")

            return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"