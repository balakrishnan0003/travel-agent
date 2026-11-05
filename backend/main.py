import time
try:
    import mlflow
    MLFLOW_ENABLED = True
except ImportError:
    MLFLOW_ENABLED = False

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from backend.agent.graph import travel_agent_app
from backend.config import LLM_MODEL, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# ── MLflow experiment setup ───────────────────────────────────────────────────
if MLFLOW_ENABLED:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

app = FastAPI(title="AI Travel Agent Backend")


class PlanRequest(BaseModel):
    destination: str
    days: int = 3
    budget: str = "Moderate"
    interests: list[str] = ["Sightseeing", "Food"]
    travel_dates: str = "Next month"
    restrictions: str = ""


def extract_text(msg) -> str:
    """Safely extract plain text regardless of Groq content shape (str or list-of-blocks)."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in msg.content
        )
    return ""


def is_final_ai(msg) -> bool:
    """True only for AI messages that carry the final answer (no pending tool calls)."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    return msg.type == "ai" and not tool_calls


@app.post("/plan")
async def generate_itinerary(req: PlanRequest):
    user_prompt = (
        f"Plan a {req.days}-day trip to {req.destination}. "
        f"Travel Dates: {req.travel_dates}. Budget: {req.budget}. "
        f"Interests: {', '.join(req.interests)}. "
        f"Please check weather, recommend hotels, find local events, "
        f"and give a structured day-by-day plan."
    )
    if req.restrictions.strip():
        user_prompt += f"\nCRITICAL RESTRICTIONS TO ENFORCE: {req.restrictions}"

    if MLFLOW_ENABLED:
        with mlflow.start_run(run_name=f"plan_{req.destination}"):
            return await _run_agent(req, user_prompt)
    else:
        return await _run_agent(req, user_prompt)


async def _run_agent(req: PlanRequest, user_prompt: str):
    if MLFLOW_ENABLED:
        mlflow.log_params({
            "destination": req.destination,
            "days": req.days,
            "budget": req.budget,
            "interests": ", ".join(req.interests),
            "travel_dates": req.travel_dates,
            "has_restrictions": bool(req.restrictions.strip()),
            "model": LLM_MODEL,
        })

    start = time.time()
    try:
        result = travel_agent_app.invoke(
            {"messages": [HumanMessage(content=user_prompt)]}
        )
    except Exception as e:
        if MLFLOW_ENABLED:
            mlflow.set_tag("status", "agent_error")
            mlflow.log_param("error_message", str(e)[:250])
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    total_latency = round(time.time() - start, 3)
    all_messages = result.get("messages", [])

    tool_messages = [m for m in all_messages if m.type == "tool"]
    final_ai_msgs = [m for m in all_messages if is_final_ai(m)]
    tools_used = sorted({m.name for m in tool_messages})

    if MLFLOW_ENABLED:
        mlflow.log_metrics({
            "total_latency_s": total_latency,
            "total_messages": len(all_messages),
            "tool_calls_made": len(tool_messages),
        })
        mlflow.set_tags({
            "tools_used": ", ".join(tools_used) if tools_used else "none",
            "status": "success" if final_ai_msgs else "empty_response",
        })

    if not final_ai_msgs:
        if MLFLOW_ENABLED:
            mlflow.set_tag("status", "empty_response")
        raise HTTPException(
            status_code=500,
            detail="Agent completed but produced no final text response.",
        )

    itinerary = extract_text(final_ai_msgs[-1]).strip()

    if not itinerary:
        if MLFLOW_ENABLED:
            mlflow.set_tag("status", "empty_itinerary")
        raise HTTPException(
            status_code=500,
            detail="Agent returned an empty itinerary. Try adjusting your prompt.",
        )

    if MLFLOW_ENABLED:
        mlflow.log_metric("itinerary_char_count", len(itinerary))
        mlflow.log_text(itinerary, artifact_file="itinerary.md")

    return {"itinerary": itinerary}
