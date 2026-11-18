from fastapi.testclient import TestClient

from pathsnap.api.main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == "Welcome to PathSnap API"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"

    assert data["service"] == "PathSnap API"


def test_api_documentation():

    response = client.get("/docs")

    assert response.status_code == 200
