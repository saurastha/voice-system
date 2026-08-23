import io
import logging

import soundfile as sf
from fastapi import FastAPI, Response
from kokoro import KPipeline
from pydantic import BaseModel

logger = logging.getLogger("tts_service")
app = FastAPI()

logger.info("Loading Kokoro TTS model...")
pipeline = KPipeline(lang_code="a")
logger.info("Kokoro loaded successfully.")


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    """Receives text and returns a playable WAV audio file."""
    if not request.text.strip():
        return Response(status_code=204)

    try:
        # Generate audio array
        generator = pipeline(
            request.text, voice=request.voice, speed=1.0, split_pattern=r"\n+"
        )
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.extend(audio)

        if not audio_chunks:
            return Response(status_code=204)

        # Convert raw array to 24kHz PCM WAV bytes
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_chunks, 24000, format="WAV", subtype="PCM_16")

        # Return the audio file directly in the HTTP response
        return Response(content=wav_io.getvalue(), media_type="audio/wav")

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        return Response(status_code=500)
