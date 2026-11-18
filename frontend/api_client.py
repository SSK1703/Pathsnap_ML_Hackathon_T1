import os

import requests
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv(
    "PATHSNAP_API_URL",
    "http://127.0.0.1:8000"
)


class PathSnapAPIClient:

    def __init__(self, base_url=API_BASE_URL):

        self.base_url = base_url.rstrip("/")

    def check_health(self):

        response = requests.get(
            f"{self.base_url}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    def get_trips(self):

        response = requests.get(
            f"{self.base_url}/api/v1/trips",
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    def predict(self, trip_id):

        response = requests.post(
            f"{self.base_url}/api/v1/predict",
            json={"trip_id": trip_id},
            timeout=120
        )

        response.raise_for_status()

        return response.json()
