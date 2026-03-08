"""
Curiosity session — represents one complete "train of thought".

States:
  active    – currently in dialogue
  completed – user ended naturally
  saved     – user said "save this" / "continue later"
  ignored   – no response within timeout
  rejected  – user said "no" / "not now"
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from loguru import logger

from rpi.storage.database import Database


class SessionStatus(str, Enum):
    ACTIVE    = "active"
    COMPLETED = "completed"
    SAVED     = "saved"
    IGNORED   = "ignored"
    REJECTED  = "rejected"


@dataclass
class Turn:
    role: Literal["assistant", "user"]
    content: str
    timestamp: float = field(default_factory=time.time)


class CuriositySession:
    def __init__(
        self,
        db: Database,
        trigger_question: str,
        scene_context: dict | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self._db = db
        self.trigger_question = trigger_question
        self.scene_context = scene_context
        self.turns: list[Turn] = []
        self.status = SessionStatus.ACTIVE
        self.started_at = time.time()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def begin(self) -> None:
        await self._db.insert_curiosity(self.id, self.trigger_question, self.scene_context)
        await self._db.log_event("curiosity_started", curiosity_id=self.id)
        logger.info(f"Curiosity {self.id[:8]} started.")

    async def add_turn(self, role: Literal["assistant", "user"], content: str) -> None:
        turn = Turn(role=role, content=content)
        self.turns.append(turn)
        idx = len(self.turns) - 1
        await self._db.insert_turn(self.id, idx, role, content)
        await self._db.increment_turn_count(self.id)
        if role == "user":
            await self._db.log_event("turn_answered", curiosity_id=self.id)

    async def end(self, status: SessionStatus, summary: str | None = None) -> None:
        self.status = status
        await self._db.update_curiosity_status(self.id, status.value, summary)
        await self._db.log_event(f"curiosity_{status.value}", curiosity_id=self.id)
        logger.info(f"Curiosity {self.id[:8]} -> {status.value}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def message_history(self) -> list[dict]:
        """Return turns in Anthropic messages format."""
        return [{"role": t.role, "content": t.content} for t in self.turns]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def depth(self) -> int:
        """Number of consecutive user turns without topic change (proxy for depth)."""
        return sum(1 for t in self.turns if t.role == "user")


class SessionStore:
    """Loads saved curiosities for resumption."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def list_saved(self) -> list[dict]:
        return await self._db.get_saved_curiosities()

    async def load_turns(self, curiosity_id: str) -> list[dict]:
        return await self._db.get_turns(curiosity_id)

    async def resume(self, curiosity_id: str) -> list[dict]:
        """Return the full message history for a saved curiosity."""
        turns = await self.load_turns(curiosity_id)
        return [{"role": t["role"], "content": t["content"]} for t in turns]
