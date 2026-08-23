from abc import ABC, abstractmethod


class BaseVAD(ABC):
    @abstractmethod
    def process_audio(self, audio_chunk: bytes) -> list:
        """Takes raw audio bytes and returns a list of events like [('StartOfTurn', None), ('EndOfTurn', audio_bytes)]"""
        pass


class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """Takes raw 16kHz PCM audio bytes and returns transcribed text."""
        pass


class BaseTTS(ABC):
    @abstractmethod
    def generate_audio(self, text: str) -> bytes:
        """Takes text and returns playable audio file bytes (like WAV or MP3)."""
        pass
