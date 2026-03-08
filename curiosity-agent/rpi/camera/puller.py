"""
Camera puller — periodically fetches a JPEG frame from the ESP32
CameraWebServer's /capture endpoint and puts it into the frame queue.

Use this when the ESP32 is running stock firmware (pull mode) instead of
the custom POST-based firmware (push mode).
"""

from __future__ import annotations

import asyncio
import time

import aiohttp
from loguru import logger

from rpi.camera.receiver import Frame


class CameraPuller:
    def __init__(
        self,
        capture_url: str,
        frame_queue: asyncio.Queue,
        interval_sec: float = 5.0,
    ) -> None:
        self._url = capture_url
        self._queue = frame_queue
        self._interval = interval_sec
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Camera puller started → {self._url} every {self._interval}s")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(self._url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            frame = Frame(data=data, timestamp=time.time(), device_id="esp32-pull")
                            try:
                                self._queue.put_nowait(frame)
                            except asyncio.QueueFull:
                                try:
                                    self._queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                await self._queue.put(frame)
                        else:
                            logger.warning(f"Puller got HTTP {resp.status} from {self._url}")
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    logger.error(f"Puller error: {exc}")

                await asyncio.sleep(self._interval)
