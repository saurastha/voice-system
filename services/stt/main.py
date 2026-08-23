import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel

logger = logging.getLogger("stt_service")
app = FastAPI()

logger.info("Loading Faster-Whisper base.en model...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")
logger.info("Faster-Whisper loaded successfully.")


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...)):
    """Receives raw 16kHz PCM audio bytes and returns text."""
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        return {"text": ""}

    try:
        # Convert raw 16-bit PCM bytes to float32 for Whisper
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Transcribe the chunk
        segments, _ = model.transcribe(audio_float32, beam_size=1)
        text = " ".join(segment.text for segment in segments).strip()

        return {"text": text}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"text": ""}
