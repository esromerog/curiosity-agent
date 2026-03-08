"""
Speech-to-text input using OpenAI Whisper (local, offline).

Records from the microphone until a configurable silence threshold is reached,
then transcribes the captured audio.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from loguru import logger


class STT:
    def __init__(
        self,
        model_name: str = "base.en",
        input_device: int | None = None,
        silence_threshold_sec: float = 2.0,
        sample_rate: int = 16000,
        max_duration_sec: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self._input_device = input_device
        self._silence_threshold = silence_threshold_sec
        self._sample_rate = sample_rate
        self._max_duration = max_duration_sec
        self._model: whisper.Whisper | None = None
        self._lock = asyncio.Lock()

    def _load_model(self) -> None:
        if self._model is None:
            logger.info(f"Loading Whisper model '{self._model_name}'…")
            self._model = whisper.load_model(self._model_name)
            logger.info("Whisper model ready.")

    async def listen(self, timeout_sec: float = 30.0) -> str | None:
        """
        Listen for speech, stopping after silence or timeout.
        Returns the transcribed text, or None if nothing was heard.
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            try:
                text = await asyncio.wait_for(
                    loop.run_in_executor(None, self._record_and_transcribe),
                    timeout=timeout_sec,
                )
                return text if text and text.strip() else None
            except asyncio.TimeoutError:
                logger.debug("STT listen timeout.")
                return None
            except Exception as exc:
                logger.error(f"STT error: {exc}")
                return None

    def _record_and_transcribe(self) -> str | None:
        self._load_model()

        chunk_duration = 0.1           # seconds per chunk
        chunk_samples = int(self._sample_rate * chunk_duration)
        silence_rms_threshold = 0.01   # tune for ambient noise
        silence_chunks_needed = int(self._silence_threshold / chunk_duration)
        max_chunks = int(self._max_duration / chunk_duration)

        recorded: list[np.ndarray] = []
        silence_count = 0
        speech_detected = False

        logger.debug("Listening…")

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._input_device,
            blocksize=chunk_samples,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > silence_rms_threshold:
                    speech_detected = True
                    silence_count = 0
                    recorded.append(chunk)
                elif speech_detected:
                    recorded.append(chunk)
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        break

        if not speech_detected or not recorded:
            return None

        audio = np.concatenate(recorded)
        # write to in-memory WAV
        buf = io.BytesIO()
        sf.write(buf, audio, self._sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        # transcribe
        result = self._model.transcribe(
            audio,
            language="en",
            fp16=False,
            condition_on_previous_text=False,
        )
        text = result.get("text", "").strip()
        logger.debug(f"Transcribed: {text!r}")
        return text or None
