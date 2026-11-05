import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_menu_page(client):
    response = client.get("/menu")
    assert response.status_code == 200


def test_available_attacks(client):
    response = client.get("/api/attacks/available")
    assert response.status_code == 200
    assert response.get_json()
