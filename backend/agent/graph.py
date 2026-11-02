import time
import mlflow
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.config import GROQ_API_KEY, LLM_MODEL
from backend.tools.weather import get_weather_forecast
from backend.tools.search import search_local_events_and_hotels
from backend.tools.places_rag import search_local_guides
from backend.agent.state import AgentState

tools = [get_weather_forecast, search_local_events_and_hotels, search_local_guides]

llm = ChatGroq(
    model_name=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
).bind_tools(tools)

SYSTEM_PROMPT = """You are an expert AI Travel Concierge.

PHASE 1: RESEARCH
You must use your tools to gather real-world data:
1. `search_local_guides` (param: query)
2. `get_weather_forecast` (param: city)
3. `search_local_events_and_hotels` (param: query)

PHASE 2: SYNTHESIS
Once you receive the tool responses, you MUST STOP calling tools and write the final response to the user.
Your final response MUST be a rich, beautifully formatted Markdown blog post (use ## headers, bullet points, emojis).
NEVER output JSON in this final phase. Just write normal, readable Markdown text.

CITATION RULES:
Cite your sources inline for every recommendation:
- YouTube/Compass tips -> [Compass RAG]
- Web search results -> [Live Web]
- Weather data -> [Live Weather]
"""

MAX_TOOL_ITERATIONS = 6


def call_model(state: AgentState):
    messages = list(state["messages"])

    if not any(m.type == "system" for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # Count how many tool-call rounds have already happened.
    # If the agent is stuck in a loop, inject a nudge to force synthesis.
    tool_rounds = sum(1 for m in messages if m.type == "tool")
    if tool_rounds >= MAX_TOOL_ITERATIONS:
        messages.append(
            SystemMessage(
                content=(
                    "You have already called enough tools. "
                    "DO NOT call any more tools. "
                    "Write the final Markdown travel itinerary NOW using the data you already have."
                )
            )
        )

    # ── MLflow: log each LLM hop only when a run is active (set by main.py) ──
    start = time.time()
    response = llm.invoke(messages)
    latency = round(time.time() - start, 3)

    if mlflow.active_run():
        tool_calls = getattr(response, "tool_calls", None) or []
        mlflow.log_metrics({
            "llm_call_latency_s": latency,
            "tool_rounds":        tool_rounds,
        })
        if tool_rounds >= MAX_TOOL_ITERATIONS:
            mlflow.set_tag("forced_synthesis", "true")
        if tool_calls:
            mlflow.set_tag(
                "tools_called",
                ", ".join(t["name"] for t in tool_calls)
            )

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # Normalise: Groq sometimes sets tool_calls=None instead of []
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "agent")

travel_agent_app = workflow.compile()