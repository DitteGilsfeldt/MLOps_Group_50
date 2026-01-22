from fastapi.testclient import TestClient
from group50.api import app  # adjust if needed

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
