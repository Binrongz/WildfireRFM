#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weather Data Retrieval Module
"""

import requests

# 替换为实时 API Key
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

def get_weather(lat, lon, lang="en"):
    """
    Retrieve real-time weather data
    :param lat: Latitude
    :param lon: Longitude
    :param units: "metric" = Celsius, "imperial" = Fahrenheit
    :return: dict Including weather information
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": lang
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        # Unit conversion
        temp_c = data["main"]["temp"]
        temp_f = temp_c * 9/5 + 32
        wind_mps = data["wind"]["speed"]
        wind_mph = wind_mps * 2.23694

        return {
            "temperature": {
                "celsius": round(temp_c, 2),
                "fahrenheit": round(temp_f, 2)
            },
            "humidity": data["main"]["humidity"],
            "wind_speed": {
                "mps": round(wind_mps, 2),
                "mph": round(wind_mph, 2)
            },
            "weather_description": data["weather"][0]["description"],
            "source": "OpenWeatherMap"
        }

    except Exception as e:
        print(f"❌ Weather query failed.: {e}")
        return {
            "temperature": {"celsius": None, "fahrenheit": None},
            "humidity": None,
            "wind_speed": {"mps": None, "mph": None},
            "weather_description": "Unavailable",
            "source": "OpenWeatherMap"
        }
        