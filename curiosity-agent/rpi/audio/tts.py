"""
Text-to-speech output.

Backends:
  - "edge-tts"  : Microsoft Edge TTS (online, high quality neural voices)
  - "pyttsx3"   : offline, system TTS (espeak on Linux)
  - "espeak"    : direct espeak subprocess call (always available on RPi)
"""

from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger


class TTS:
    def __init__(
        self,
        engine: Literal["edge-tts", "pyttsx3", "espeak"] = "edge-tts",
        voice: str = "en-US-GuyNeural",
        output_device: int | None = None,
    ) -> None:
        self._engine = engine
        self._voice = voice
        self._output_device = output_device
        self._lock = asyncio.Lock()

    async def speak(self, text: str) -> None:
        """Convert text to speech and play it. Blocks until audio finishes."""
        async with self._lock:
            if self._engine == "edge-tts":
                await self._speak_edge(text)
            elif self._engine == "pyttsx3":
                await self._speak_pyttsx3(text)
            else:
                await self._speak_espeak(text)

    # ------------------------------------------------------------------
    # edge-tts
    # ------------------------------------------------------------------

    async def _speak_edge(self, text: str) -> None:
        try:
            import edge_tts
            import sounddevice as sd
            import soundfile as sf

            communicate = edge_tts.Communicate(text, self._voice)
            audio_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.write(chunk["data"])

            audio_buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
                f.write(audio_buf.read())

            # play via sounddevice
            data, samplerate = sf.read(tmp_path)
            sd.play(data, samplerate, device=self._output_device)
            sd.wait()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"edge-tts failed: {exc}. Falling back to espeak.")
            await self._speak_espeak(text)

    # ------------------------------------------------------------------
    # pyttsx3  (blocking, run in executor)
    # ------------------------------------------------------------------

    async def _speak_pyttsx3(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._pyttsx3_sync, text)

    def _pyttsx3_sync(self, text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.error(f"pyttsx3 failed: {exc}")

    # ------------------------------------------------------------------
    # espeak  (subprocess, always available on Raspberry Pi OS)
    # ------------------------------------------------------------------

    async def _speak_espeak(self, text: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._espeak_sync, text)

    def _espeak_sync(self, text: str) -> None:
        try:
            subprocess.run(
                ["espeak", "-v", "en", "-s", "160", text],
                check=True,
                timeout=30,
            )
        except Exception as exc:
            logger.error(f"espeak failed: {exc}")
