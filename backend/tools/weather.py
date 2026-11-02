import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherSchema(BaseModel):
    city: str = Field(description="Destination city name.")


@tool(args_schema=WeatherSchema)
def get_weather_forecast(city: str) -> str:
    """Fetch 7-day weather forecast."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=10).json()
        if not geo_res.get("results"):
            return f"Could not find coordinates for {city}."

        loc = geo_res["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        w_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto"
        )
        w_res = requests.get(w_url, timeout=10).json()
        daily = w_res.get("daily", {})

        summary = [f"Forecast for {loc.get('name')}:"]
        dates = daily.get("time", [])
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        rain_prob = daily.get("precipitation_probability_max", [])

        for i in range(min(len(dates), 7)):
            summary.append(
                f"- {dates[i]}: High {t_max[i]}°C, Low {t_min[i]}°C, Rain Prob: {rain_prob[i]}%"
            )
        return "\n".join(summary)
    except Exception as e:
        return f"Weather API error: {str(e)}"