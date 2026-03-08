"""
Cooldown manager.

After a curiosity is ignored or rejected a 10-minute silence period begins.
During this window new curiosities must NOT be triggered.
"""

from __future__ import annotations

import time

from loguru import logger

from rpi.storage.database import Database


class CooldownManager:
    def __init__(self, db: Database, cooldown_minutes: float = 10.0) -> None:
        self._db = db
        self._cooldown_sec = cooldown_minutes * 60
        self._cooldown_ends_at: float = 0.0

    @property
    def is_active(self) -> bool:
        return time.time() < self._cooldown_ends_at

    @property
    def remaining_sec(self) -> float:
        remaining = self._cooldown_ends_at - time.time()
        return max(0.0, remaining)

    async def start(self, reason: str) -> None:
        now = time.time()
        ends_at = now + self._cooldown_sec
        self._cooldown_ends_at = ends_at
        await self._db.insert_cooldown(now, ends_at, reason)
        await self._db.log_event("cooldown_started", metadata={"reason": reason, "ends_at": ends_at})
        logger.info(f"Cooldown started ({reason}). Resumes in {self._cooldown_sec / 60:.0f} min.")

    async def cancel(self) -> None:
        """Cancel an active cooldown (e.g. user explicitly resumes)."""
        if self.is_active:
            self._cooldown_ends_at = 0.0
            await self._db.log_event("cooldown_ended")
            logger.info("Cooldown cancelled by user action.")
