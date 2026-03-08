"""
Analytics tracker.

Records engagement patterns and updates the hidden user profile
after each curiosity session closes.
"""

from __future__ import annotations

import time
from datetime import datetime

from loguru import logger

from rpi.agent.session import CuriositySession, SessionStatus
from rpi.storage.database import Database


class AnalyticsTracker:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_turn(self, session: CuriositySession) -> None:
        """Called after each completed user turn."""
        await self._db.log_event("turn_answered", curiosity_id=session.id)

    async def record_session_end(self, session: CuriositySession) -> None:
        """Called when a session closes (any status)."""
        status = session.status
        now = datetime.now()

        # depth = user turn count, breadth = number of distinct topics (proxy = turn ratio)
        user_turns = sum(1 for t in session.turns if t.role == "user")
        assistant_turns = sum(1 for t in session.turns if t.role == "assistant")

        await self._db.log_event(
            f"curiosity_{status.value}",
            curiosity_id=session.id,
            metadata={
                "turn_count": session.turn_count,
                "user_turns": user_turns,
                "duration_sec": round(time.time() - session.started_at, 1),
                "hour_of_day": now.hour,
            },
        )

        if status in (SessionStatus.COMPLETED, SessionStatus.SAVED):
            await self._update_engagement_profile(session, now)

    async def _update_engagement_profile(
        self, session: CuriositySession, dt: datetime
    ) -> None:
        # engagement by hour
        hour_key = f"engagement_hour_{dt.hour}"
        current = await self._db.get_profile_key(hour_key, 0)
        await self._db.set_profile_key(hour_key, current + 1)

        # depth score (running mean of user turn counts)
        user_turns = sum(1 for t in session.turns if t.role == "user")
        current_depth = await self._db.get_profile_key("avg_session_depth", 0.0)
        total_sessions = await self._db.get_profile_key("total_sessions_counted", 0)
        new_avg = (current_depth * total_sessions + user_turns) / (total_sessions + 1)
        await self._db.set_profile_key("avg_session_depth", round(new_avg, 2))
        await self._db.set_profile_key("total_sessions_counted", total_sessions + 1)

        # save rate
        saved = await self._db.get_profile_key("save_count", 0)
        if session.status == SessionStatus.SAVED:
            await self._db.set_profile_key("save_count", saved + 1)
