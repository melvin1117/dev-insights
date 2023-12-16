import googlemaps
import json
from typing import Dict, Any, Union
from os import getenv
from log_config import LoggerConfig

# Initialize the logger for this module
logger = LoggerConfig(__name__).logger

GMAP_API_KEY = getenv('GMAP_API_KEY')
CACHE_FILE_PATH = "assets/geocode_cache.json"

class LocationGeocodingService:
    """
    A class for geocoding locations using the Google Maps API and caching the results.
    """

    def __init__(self) -> None:
        """
        Initializes the LocationGeocodingService.

        - gmaps: A Google Maps API client instance.
        - cache: A dictionary to store geocoding results.
        """
        self.gmaps = googlemaps.Client(key=GMAP_API_KEY)
        self.cache: Dict[str, Any] = {}
        self.load_cache_from_file()  # Load cache from file when an instance is created

    def geocode(self, address: str) -> Union[Dict[str, Any], None]:
        """
        Geo codes the given address using the Google Maps API.

        Args:
            address (str): The address to geocode.

        Returns:
            Union[Dict[str, Any], None]: Geo coded location information or None if geocoding fails.
        """
        if not address:
            return None
        
        if address in self.cache:
            logger.info(f"Using cached result for {address}")
            return self.cache[address]

        geocode_result = self.gmaps.geocode(address)

        if geocode_result and len(geocode_result):
            geocode_result = geocode_result[0]
            location = geocode_result['geometry']['location']
            formatted_address = geocode_result['formatted_address']
            location_obj = {
                'formatted_address': formatted_address,
                'address': address,
                **location
            }
            self.cache[address.replace(" ", "").lower()] = location_obj
            return location_obj
        else:
            logger.warning(f"Geocoding failed for {address}, setting default.")
            return None

    def save_cache_to_file(self) -> None:
        """
        Saves the geocoding cache to a file.
        """
        with open(CACHE_FILE_PATH, "w") as file:
            json.dump(self.cache, file)
        logger.info("Location Cache saved to file.")

    def load_cache_from_file(self) -> None:
        """
        Loads the geocoding cache from a file.
        """
        try:
            with open(CACHE_FILE_PATH, "r") as file:
                data = json.load(file)

            if data and isinstance(data, dict):
                self.cache = data
                logger.info("Location Cache loaded successfully.")
            else:
                logger.info("Location Cache file is empty or not in the expected format. Starting with an empty cache.")

        except FileNotFoundError:
            logger.warning("Location Cache file not found. Starting with an empty cache.")
        except json.JSONDecodeError:
            logger.warning("Error decoding Location Cache JSON. Starting with an empty cache.")

    def __del__(self) -> None:
        """
        Destructor. Saves the cache to a file when the object is destroyed.
        """
        self.save_cache_to_file()
