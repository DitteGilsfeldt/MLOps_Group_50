from fastapi.testclient import TestClient
from group50.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Emotion model inference API!"}

def test_read_item():
    response = client.get("/items/7")
    assert response.status_code == 200
    assert response.json() == {"item_id": 7}
