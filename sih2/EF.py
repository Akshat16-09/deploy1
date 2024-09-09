import requests
import json
from datetime import datetime, timezone, timedelta

print("Weather Application")

# Function to convert UTC time to local time
def Time_Format_For_location(utc_with_tz, timezone_offset):
    local_time = datetime.fromtimestamp(utc_with_tz + timezone_offset, tz=timezone.utc)
    return local_time.strftime('%Y-%m-%d %H:%M:%S')

# Main function to show weather
def showWeather():
    api_key = "9eafd5e32ce6bdebbc2ff649cea24b1c"  # Your API key
    cityName = input("Enter the city name (default is London): ") or "London"  # Default to London if no input

    # Corrected weather URL with city name and API key
    weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={cityName}&appid={api_key}'

    try:
        # Requesting weather data
        response = requests.get(weather_url)
        response.raise_for_status()  # Check if the request was successful
        weather_info = response.json()

        # Check if the city is found
        if weather_info['cod'] == 200:
            kelvin_to_celsius = 273.15  # Conversion factor to Celsius
            temp = round(weather_info['main']['temp'] - kelvin_to_celsius, 2)
            pressure = weather_info['main']['pressure']
            humidity = weather_info['main']['humidity']
            sunrise = weather_info['sys']['sunrise']
            sunset = weather_info['sys']['sunset']
            timezone_offset = weather_info['timezone']
            cloudy = weather_info['clouds']['all']
            description = weather_info['weather'][0]['description']

            # Convert sunrise and sunset times to local times
            sunrise_time = Time_Format_For_location(sunrise, timezone_offset)
            sunset_time = Time_Format_For_location(sunset, timezone_offset)

            # Prepare the weather data output
            weather_data = f"""
            Weather for '{cityName}':
            Temperature (Celsius): {temp}°C
            Pressure: {pressure} hPa
            Humidity: {humidity}%
            Sunrise at: {sunrise_time}
            Sunset at: {sunset_time}
            Cloudiness: {cloudy}%
            Weather description: {description}
            """
            print(weather_data)
        else:
            # City not found case
            print(f"\nWeather for '{cityName}' is not found!\nPlease enter a valid city name.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Call the function
showWeather()
