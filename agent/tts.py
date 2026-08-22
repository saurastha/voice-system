import io
from loguru import logger
import soundfile as sf
from kokoro import KPipeline
from agent.base import BaseTTS


class KokoroTTS(BaseTTS):
    def __init__(self, voice="af_heart"):
        logger.info(f"Loading Kokoro TTS voice '{voice}'...")
        lang_code = voice[0]
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice

    def generate_audio(self, text: str) -> bytes:
        if not text.strip():
            return b""
        generator = self.pipeline(
            text, voice=self.voice, speed=1.0, split_pattern=r"\n+"
        )
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.extend(audio)
        if not audio_chunks:
            return b""
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_chunks, 24000, format="WAV", subtype="PCM_16")
        return wav_io.getvalue()
