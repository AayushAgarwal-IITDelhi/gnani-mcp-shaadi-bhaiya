"""Gnani STT/TTS exposed as an MCP server (Streamable HTTP at /mcp).

Tools:
  gnani_transcribe(audio_url, language_code)  -> transcript text
  gnani_voice_reply(text, language)           -> public audio_url to send on WhatsApp

Run:  uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import os
import time
import uuid
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")
STT_URL = os.environ.get("GNANI_STT_URL", "https://api.vachana.ai/stt/v3")
TTS_URL = os.environ.get("GNANI_TTS_URL", "https://api.vachana.ai/api/v1/tts/inference")
TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.environ.get("GNANI_TTS_VOICE", "Nalini")
# WhatsApp prefers ogg/opus. If Gnani rejects this combo, set these three to
# linear_pcm / wav / 48000 (the values from Gnani's docs example).
TTS_ENCODING = os.environ.get("GNANI_TTS_ENCODING", "oggopus")
TTS_CONTAINER = os.environ.get("GNANI_TTS_CONTAINER", "ogg")
TTS_SAMPLE_RATE = int(os.environ.get("GNANI_TTS_SAMPLE_RATE", "24000"))

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")  # e.g. https://gnani-mcp.onrender.com
MCP_BEARER_TOKEN = os.environ.get("MCP_BEARER_TOKEN", "")  # optional shared secret for /mcp
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
# Only these hosts may be fetched for STT audio (prevents the tool being used as an open proxy).
ALLOWED_AUDIO_HOSTS = {
    h.strip() for h in os.environ.get("ALLOWED_AUDIO_HOSTS", "api.twilio.com,media.twiliocdn.com").split(",") if h.strip()
}
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_TTL_SECONDS = 15 * 60

# audio_id -> (expires_at, bytes, mime)
_audio_store: dict[str, tuple[float, bytes, str]] = {}

# host="0.0.0.0" so the SDK does not enable localhost-only DNS-rebinding protection,
# which would reject requests arriving on the public hostname.
mcp = FastMCP("gnani-wedding", host="0.0.0.0", stateless_http=True, json_response=True)


def _require_key() -> None:
    if not GNANI_API_KEY:
        raise RuntimeError("GNANI_API_KEY is not configured on the server")


def _purge_expired() -> None:
    now = time.time()
    for k in [k for k, v in _audio_store.items() if v[0] < now]:
        _audio_store.pop(k, None)


@mcp.tool()
async def gnani_transcribe(audio_url: str, language_code: str = "hi-IN") -> dict:
    """Transcribe a short voice note (<= 60 s) to text using Gnani speech-to-text.

    audio_url: URL of the voice note (e.g. the Twilio WhatsApp MediaUrl0).
    language_code: BCP-47 code such as hi-IN or en-IN.
    Returns {"transcript": str}. Never guess missing details; if the transcript is
    unclear, report that back instead of filling gaps.
    """
    _require_key()
    parsed = urlparse(audio_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_AUDIO_HOSTS:
        raise ValueError(f"audio_url host not allowed: {parsed.hostname}")

    auth = (TWILIO_SID, TWILIO_TOKEN) if parsed.hostname == "api.twilio.com" and TWILIO_SID else None
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        dl = await client.get(audio_url, auth=auth)
        dl.raise_for_status()
        audio = dl.content
        if len(audio) > MAX_AUDIO_BYTES:
            raise ValueError("audio too large")
        mime = dl.headers.get("content-type", "audio/ogg").split(";")[0]

        resp = await client.post(
            STT_URL,
            headers={"X-API-Key-ID": GNANI_API_KEY},
            files={"audio_file": ("voice_note", audio, mime)},
            data={"language_code": language_code, "format": "transcribe"},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Gnani STT failed: HTTP {resp.status_code} {resp.text[:300]}")
    body = resp.json()
    if not body.get("success", True) or not body.get("transcript"):
        raise RuntimeError(f"Gnani STT returned no transcript: {str(body)[:300]}")
    return {"transcript": body["transcript"], "language_code": language_code}


@mcp.tool()
async def gnani_voice_reply(text: str, language: str = "hi-IN", voice: str = "") -> dict:
    """Convert text to speech with Gnani and return a public audio_url.

    Speaks exactly the text given. Send the returned audio_url as a WhatsApp media
    message. Only call this with text already approved to be sent to the family.
    """
    _require_key()
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured on the server")
    if not text.strip() or len(text) > 2000:
        raise ValueError("text must be 1-2000 characters")

    payload = {
        "text": text,
        "voice": voice or TTS_VOICE,
        "model": TTS_MODEL,
        "language": language,
        "speed": 1,
        "audio_config": {
            "encoding": TTS_ENCODING,
            "container": TTS_CONTAINER,
            "num_channels": 1,
            "sample_rate": TTS_SAMPLE_RATE,
            "sample_width": 2,
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(TTS_URL, headers={"X-API-Key-ID": GNANI_API_KEY}, json=payload)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"Gnani TTS failed: HTTP {resp.status_code} {resp.text[:300]}")

    mime = {"ogg": "audio/ogg", "wav": "audio/wav", "mp3": "audio/mpeg"}.get(TTS_CONTAINER, "application/octet-stream")
    _purge_expired()
    audio_id = uuid.uuid4().hex
    _audio_store[audio_id] = (time.time() + AUDIO_TTL_SECONDS, resp.content, mime)
    return {"audio_url": f"{PUBLIC_BASE_URL}/audio/{audio_id}", "mime_type": mime, "expires_in_seconds": AUDIO_TTL_SECONDS}


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "service": "gnani-wedding-mcp"})


@mcp.custom_route("/audio/{audio_id}", methods=["GET"])
async def get_audio(request: Request) -> Response:
    item = _audio_store.get(request.path_params["audio_id"])
    if not item or item[0] < time.time():
        return Response(status_code=404)
    return Response(content=item[1], media_type=item[2])


class BearerAuth(BaseHTTPMiddleware):
    """Protect /mcp with a shared secret when MCP_BEARER_TOKEN is set."""

    async def dispatch(self, request: Request, call_next):
        if MCP_BEARER_TOKEN and request.url.path.startswith("/mcp"):
            if request.headers.get("authorization") != f"Bearer {MCP_BEARER_TOKEN}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


app = mcp.streamable_http_app()
app.add_middleware(BearerAuth)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
