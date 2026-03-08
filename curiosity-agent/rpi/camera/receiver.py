"""
HTTP server that accepts JPEG frames POSTed by the ESP32-CAM.

Frames are placed into an asyncio.Queue so the recognition engine
can consume them at its own pace without blocking the HTTP server.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from aiohttp import web
from loguru import logger


@dataclass
class Frame:
    data: bytes
    timestamp: float = field(default_factory=time.time)
    device_id: str = "unknown"


class CameraReceiver:
    def __init__(self, host: str, port: int, queue_maxsize: int = 8) -> None:
        self._host = host
        self._port = port
        self.frame_queue: asyncio.Queue[Frame] = asyncio.Queue(maxsize=queue_maxsize)
        self._app = web.Application()
        self._app.router.add_post("/frame", self._handle_frame)
        self._app.router.add_get("/health", self._handle_health)
        self._runner: web.AppRunner | None = None
        self._last_frame_at: float = 0.0

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info(f"Camera receiver listening on {self._host}:{self._port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle_frame(self, request: web.Request) -> web.Response:
        body = await request.read()
        if not body:
            return web.Response(status=400, text="empty body")

        device_id = request.headers.get("X-Device-ID", "unknown")
        frame = Frame(data=body, device_id=device_id)

        try:
            self.frame_queue.put_nowait(frame)
            self._last_frame_at = frame.timestamp
        except asyncio.QueueFull:
            # drop oldest, insert newest
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await self.frame_queue.put(frame)
            logger.debug("Frame queue was full; dropped oldest frame.")

        return web.Response(status=200, text="ok")

    async def _handle_health(self, request: web.Request) -> web.Response:
        age = time.time() - self._last_frame_at if self._last_frame_at else None
        return web.json_response(
            {
                "status": "ok",
                "queue_size": self.frame_queue.qsize(),
                "last_frame_age_sec": round(age, 1) if age else None,
            }
        )

    async def get_latest_frame(self, timeout: float = 10.0) -> Frame | None:
        """Block until a frame is available or timeout expires."""
        try:
            return await asyncio.wait_for(self.frame_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
