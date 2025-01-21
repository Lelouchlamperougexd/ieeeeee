import requests, json

OPENWEATHER_API_KEY = "49ed84e283efd82dfc2239cb5afd36c8"

def get_weather_data(location):
    """Fetch current weather data for a given location."""
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat=51.1655&lon=71.4272&appid={"49ed84e283efd82dfc2239cb5afd36c8"}"
    params = {"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather = data.get("weather", [{}])[0].get("description", "No description available")
        temp = data.get("main", {}).get("temp", "N/A")
        return f"Weather: {weather}, Temperature: {temp}°C"
    else:
        return f"Failed to fetch weather data for {location}."
