"""Integration tests for ElevenLabs API endpoints."""

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


# --- Voice management ---


@pytest.mark.asyncio
async def test_list_voices(client):
    resp = await client.get("/v1/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert len(data["voices"]) >= 2
    # Check ElevenLabs format fields
    voice = data["voices"][0]
    assert "voice_id" in voice
    assert "name" in voice
    assert "settings" in voice


@pytest.mark.asyncio
async def test_get_voice_by_alias(client):
    resp = await client.get("/v1/voices/conversationalB")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "conversational_b"


@pytest.mark.asyncio
async def test_get_voice_by_name(client):
    resp = await client.get("/v1/voices/conversational_b")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "conversational_b"


@pytest.mark.asyncio
async def test_get_voice_unknown_returns_404(client):
    resp = await client.get("/v1/voices/unknownVoice123")
    assert resp.status_code == 404
    data = resp.json()
    assert data["detail"]["status"] == "voice_not_found"


# --- Compatibility stubs ---


@pytest.mark.asyncio
async def test_user_subscription(client):
    resp = await client.get("/v1/user/subscription")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tier"] == "local"
    assert data["status"] == "active"
    assert data["character_limit"] == 999999999


@pytest.mark.asyncio
async def test_user_info(client):
    resp = await client.get("/v1/user")
    assert resp.status_code == 200
    data = resp.json()
    assert data["xi_api_key"] == "local"
    assert data["subscription"]["tier"] == "local"
    assert data["subscription"]["status"] == "active"


# --- Non-streaming TTS ---


@pytest.mark.asyncio
async def test_tts_non_stream_pcm(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB",
        json={"text": "Hello from the test.", "model_id": "csm-1b"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_tts_non_stream_with_elevenlabs_model(client):
    """ElevenLabs model IDs should be accepted and mapped to csm-1b."""
    resp = await client.post(
        "/v1/text-to-speech/conversationalB",
        json={"text": "Hello from the test.", "model_id": "eleven_turbo_v2"},
    )
    assert resp.status_code == 200
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_tts_non_stream_mp3_format(client):
    """Request mp3_44100_128 format via query param."""
    resp = await client.post(
        "/v1/text-to-speech/conversationalB?output_format=mp3_44100_128",
        json={"text": "Hello in mp3.", "model_id": "csm-1b"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0


# --- Streaming TTS ---


@pytest.mark.asyncio
async def test_tts_stream_pcm(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000",
        json={"text": "Hello from streaming test.", "model_id": "csm-1b"},
    )
    assert resp.status_code == 200
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_tts_stream_multi_sentence(client):
    """Multi-sentence input should use sentence-splitting pipeline."""
    resp = await client.post(
        "/v1/text-to-speech/conversationalB/stream?output_format=pcm_24000",
        json={
            "text": "First sentence here. Second sentence here.",
            "model_id": "csm-1b",
        },
    )
    assert resp.status_code == 200
    assert len(resp.content) > 0


# --- Error cases ---


@pytest.mark.asyncio
async def test_tts_empty_text(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB",
        json={"text": "", "model_id": "csm-1b"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_tts_deferred_format(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB/stream?output_format=pcm_16000",
        json={"text": "Test.", "model_id": "csm-1b"},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "not yet supported" in data["detail"]["message"]


@pytest.mark.asyncio
async def test_tts_invalid_format(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB?output_format=wav_48000",
        json={"text": "Test.", "model_id": "csm-1b"},
    )
    assert resp.status_code == 400
    data = resp.json()
    assert "Invalid output format" in data["detail"]["message"]


# --- Auth headers (accepted, not required) ---


@pytest.mark.asyncio
async def test_xi_api_key_accepted(client):
    resp = await client.post(
        "/v1/text-to-speech/conversationalB",
        json={"text": "Hello with auth.", "model_id": "csm-1b"},
        headers={"xi-api-key": "sk-test-key-12345"},
    )
    assert resp.status_code == 200
