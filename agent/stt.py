import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
from agent.base import BaseSTT


class FasterWhisperSTT(BaseSTT):
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        logger.info(f"Loading Faster-Whisper '{model_size}'...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_float32, beam_size=1)
        return " ".join(segment.text for segment in segments).strip()
