import torch
import numpy as np
from loguru import logger
from agent.base import BaseVAD


class SileroVAD(BaseVAD):
    def __init__(
        self,
        sample_rate=16000,
        silence_timeout_ms=800,
        speech_threshold=0.6,
        max_duration_seconds=6,
    ):
        self.sample_rate = sample_rate
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self.vad_iterator = utils[3](
            self.model,
            sampling_rate=sample_rate,
            min_silence_duration_ms=silence_timeout_ms,
            threshold=speech_threshold,
        )

        self.window_size_samples = 512
        self.audio_buffer = bytearray()

        self.is_speaking = False
        self.speech_buffer = bytearray()

        self.max_speech_bytes = max_duration_seconds * sample_rate * 2

    def process_audio(self, audio_chunk: bytes) -> list:
        self.audio_buffer.extend(audio_chunk)
        events = []
        bytes_per_window = self.window_size_samples * 2

        while len(self.audio_buffer) >= bytes_per_window:
            window_bytes = self.audio_buffer[:bytes_per_window]
            self.audio_buffer = self.audio_buffer[bytes_per_window:]

            if self.is_speaking:
                self.speech_buffer.extend(window_bytes)

                # SAFEGUARD: Hard cutoff if they've been speaking for too long
                if len(self.speech_buffer) >= self.max_speech_bytes:
                    logger.warning(
                        "Max speech duration reached. Forcefully cutting off VAD."
                    )
                    self.is_speaking = False
                    captured_speech = bytes(self.speech_buffer)
                    self.speech_buffer = bytearray()
                    self.vad_iterator.reset_states()
                    events.append(("EndOfTurn", captured_speech))
                    continue

            audio_np = (
                np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            audio_tensor = torch.from_numpy(audio_np)

            speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)

            if speech_dict:
                if "start" in speech_dict and not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = bytearray(window_bytes)
                    events.append(("StartOfTurn", None))

                if "end" in speech_dict and self.is_speaking:
                    self.is_speaking = False
                    captured_speech = bytes(self.speech_buffer)
                    self.speech_buffer = bytearray()
                    events.append(("EndOfTurn", captured_speech))

        return events
