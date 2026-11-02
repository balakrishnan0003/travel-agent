import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage
from pathlib import Path

# Get the Travel_agent project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to Python's import path
sys.path.insert(0, str(PROJECT_ROOT))
from backend.agent.graph import travel_agent_app

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

st.title("✈️ Autonomous AI Travel Agent")

st.caption("Powered by LangGraph, Groq Llama-3.3, and Open-Meteo")

# --- SIDEBAR: INPUT PARAMETER SPACE ---
with st.sidebar:
    st.header("Trip Preferences")

    destination = st.text_input("Destination City", value="Kyoto")
    days = st.slider("Duration (Days)", min_value=1, max_value=10, value=4)
    budget = st.selectbox("Budget", ["Backpacker / Budget", "Moderate", "Luxury"])
    travel_dates = st.text_input("Travel Dates", value="November 2026")

    interests = st.multiselect(
        "Interests",
        ["Historical & Cultural", "Food & Gastronomy", "Nature & Scenery", "Adventure", "Festivals", "Relaxation"],
        default=["Historical & Cultural", "Food & Gastronomy"]
    )

    restrictions = st.text_area(
        "Travel Restrictions & Needs",
        placeholder="e.g., Vegan only, Wheelchair accessible, No flights over $500"
    )

    submit_btn = st.button("Generate Smart Itinerary", type="primary", use_container_width=True)

# --- MAIN AGENT EXECUTION ---
if submit_btn:
    if not destination.strip():
        st.warning("Please provide a destination.")
    else:
        interests_str = ", ".join(interests) if interests else "General sightseeing"

        query = (
            f"Plan a {days}-day trip to {destination}. "
            f"Dates: {travel_dates}. Budget: {budget}. Interests: {interests_str}. "
            f"Check weather, hotels, and local events."
        )

        if restrictions:
            query += f"\nCRITICAL RESTRICTIONS TO ENFORCE: {restrictions}"

        try:
            status_box = st.empty()
            status_box.info("🚀 Agent initialized...")

            inputs = {"messages": [HumanMessage(content=query)]}
            full_history = []

            for step in travel_agent_app.stream(inputs, stream_mode="values"):
                full_history = step["messages"]
                last_msg = full_history[-1]

                if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tools_being_called = ", ".join([t["name"] for t in last_msg.tool_calls])
                    status_box.warning(f"🔧 Agent is querying: {tools_being_called}...")
                elif last_msg.type == "tool":
                    status_box.info("🧠 AI is analyzing the retrieved data...")
                elif last_msg.type == "ai" and not getattr(last_msg, "tool_calls", []):
                    status_box.success("✅ Itinerary Complete!")

            # --- NEW: SAFE AI MESSAGE EXTRACTION ---
            # Filter specifically for AI messages that contain text and no tool calls
            ai_messages = [msg for msg in full_history if msg.type == "ai" and not getattr(msg, "tool_calls", [])]

            if ai_messages and ai_messages[-1].content.strip():
                final_message = ai_messages[-1].content
            else:
                final_message = "⚠️ *Agent completed the task but returned an empty response. Please adjust your prompt parameters and try again.*"

            # Extract tools used from the ENTIRE history
            used_tools = {msg.name for msg in full_history if msg.type == "tool"}

            st.session_state["itinerary"] = final_message
            st.session_state["destination"] = destination
            st.session_state["used_tools"] = used_tools

        except Exception as e:
            st.error(f"Error processing plan: {e}")

# --- DISPLAY & EMOTION CAPTURE ---
if "itinerary" in st.session_state:
    st.markdown(f"### 🗺️ Your Tailored Itinerary: {st.session_state['destination']}")

    labels = {
        "search_local_guides": "🧭 Compass (YouTube)",
        "search_local_events_and_hotels": "🌐 Live Web",
        "get_weather_forecast": "🌤️ Live Weather"
    }
    sources = [labels[t] for t in st.session_state.get("used_tools", []) if t in labels]

    if sources:
        st.caption(f"**Data Sources Used:** {' | '.join(sources)}")

    st.divider()

    st.markdown(st.session_state["itinerary"])

    st.divider()
    st.write("### How did we do?")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("👍 Love it"):
            st.success("Feedback recorded: Positive. We will use this to improve future Compass recommendations!")
    with col2:
        if st.button("👎 Dislike"):
            st.error("Feedback recorded: Negative. We will adjust the AI retrieval parameters.")