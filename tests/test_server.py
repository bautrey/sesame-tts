"""Integration tests for the FastAPI server — requires model to be downloaded."""

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from server import app


@pytest_asyncio.fixture(scope="module")
async def client():
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"]["model_loaded"] is True
    assert data["ffmpeg"] is True
    assert len(data["voices"]) >= 2


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "csm-1b"


@pytest.mark.asyncio
async def test_speech_mp3(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "csm-1b", "input": "Hello.", "voice": "conversational"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_speech_wav(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "csm-1b",
            "input": "Testing the audio format output.",
            "voice": "conversational",
            "response_format": "wav",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


@pytest.mark.asyncio
async def test_speech_opus(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "csm-1b",
            "input": "Testing the audio format output.",
            "voice": "conversational",
            "response_format": "opus",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/opus"


@pytest.mark.asyncio
async def test_speech_flac(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "csm-1b",
            "input": "Testing the audio format output.",
            "voice": "conversational",
            "response_format": "flac",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/flac"


@pytest.mark.asyncio
async def test_speech_voice_alias(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "csm-1b", "input": "Hello.", "voice": "conversationalB"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_invalid_voice(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "csm-1b", "input": "Hello.", "voice": "nonexistent"},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "invalid_voice"


@pytest.mark.asyncio
async def test_invalid_format(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "model": "csm-1b",
            "input": "Hello.",
            "voice": "conversational",
            "response_format": "aac",
        },
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "invalid_format"


@pytest.mark.asyncio
async def test_empty_input(client):
    resp = await client.post(
        "/v1/audio/speech",
        json={"model": "csm-1b", "input": "", "voice": "conversational"},
    )
    assert resp.status_code == 400
