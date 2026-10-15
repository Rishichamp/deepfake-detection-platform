"""API tests in simulation mode (no model weights or Gemini key needed)."""
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("GEMINI_API_KEY", None)
os.environ.setdefault("MODEL_WEIGHTS_DIR", str(ROOT / "tests" / "no-weights"))

from fastapi.testclient import TestClient  # noqa: E402

from backend import main  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["engine_mode"] == "simulation"
    assert body["agent_mode"] == "rule-based"


def test_detect_image_in_simulation_mode(client):
    r = client.post("/api/detect", files={"file": ("face.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["mode"] == "simulation"
    assert 0.0 <= body["confidence"] <= 1.0


def test_detect_rejects_bad_type_and_empty(client):
    assert client.post("/api/detect", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.post("/api/detect", files={"file": ("x.png", b"", "image/png")}).status_code == 400


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE", 1024)
    r = client.post("/api/detect", files={"file": ("big.png", b"0" * 4096, "image/png")})
    assert r.status_code == 413
    assert not any(main.UPLOAD_DIR.glob("*.png"))  # partial file removed


def test_chat_validation(client):
    assert client.post("/api/chat", json=["not", "an", "object"]).status_code == 400
    assert client.post("/api/chat", json={"message": ""}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 3000}).status_code == 413
    assert client.post("/api/chat", json={"message": "hi", "context": "nope"}).status_code == 400
    r = client.post("/api/chat", json={"message": "What does the verdict mean?"})
    assert r.status_code == 200 and r.json()["response"]


def test_cors_does_not_allow_credentials(client):
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-credentials") != "true"


def test_static_path_traversal_blocked(client):
    r = client.get("/..%2Fbackend%2Fmain.py")
    assert b"FastAPI" not in r.content


def test_gemini_path_uses_google_genai_client():
    from backend.agent import DeepfakeInvestigator

    calls = []

    class FakeModels:
        def generate_content(self, model, contents):
            calls.append((model, contents))
            return type("R", (), {"text": "enhanced answer"})()

    agent = DeepfakeInvestigator()
    agent._gemini_available = True
    agent._gemini_client = type("C", (), {"models": FakeModels()})()
    agent._gemini_model_name = "gemini-flash-latest"
    context = {"verdict": "FAKE", "confidence": 0.9, "per_model_scores": {}, "frame_analysis": []}
    assert agent.chat("why?", context) == "enhanced answer"
    assert calls[0][0] == "gemini-flash-latest" and "why?" in calls[0][1]
