"""Async SQLite wrapper for the Curiosity Agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA_PATH.read_text())
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # ------------------------------------------------------------------
    # curiosities
    # ------------------------------------------------------------------

    async def insert_curiosity(
        self,
        curiosity_id: str,
        trigger_question: str,
        scene_context: dict | None = None,
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO curiosities (id, started_at, status, scene_context, trigger_question)
            VALUES (?, ?, 'active', ?, ?)
            """,
            (
                curiosity_id,
                time.time(),
                json.dumps(scene_context) if scene_context else None,
                trigger_question,
            ),
        )
        await self._conn.commit()

    async def update_curiosity_status(
        self,
        curiosity_id: str,
        status: str,
        summary: str | None = None,
    ) -> None:
        await self._conn.execute(
            """
            UPDATE curiosities
            SET status=?, ended_at=?, summary=?
            WHERE id=?
            """,
            (status, time.time(), summary, curiosity_id),
        )
        await self._conn.commit()

    async def increment_turn_count(self, curiosity_id: str) -> None:
        await self._conn.execute(
            "UPDATE curiosities SET turn_count = turn_count + 1 WHERE id=?",
            (curiosity_id,),
        )
        await self._conn.commit()

    async def get_curiosity(self, curiosity_id: str) -> dict | None:
        async with self._conn.execute(
            "SELECT * FROM curiosities WHERE id=?", (curiosity_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_saved_curiosities(self) -> list[dict]:
        async with self._conn.execute(
            "SELECT * FROM curiosities WHERE status='saved' ORDER BY started_at DESC"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_recent_questions(self, limit: int = 6) -> list[dict]:
        """Return the most recent curiosities that were completed or saved, with their questions and summaries."""
        async with self._conn.execute(
            """
            SELECT trigger_question, summary, status, started_at
            FROM curiosities
            WHERE status IN ('completed', 'saved')
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------

    async def insert_turn(
        self, curiosity_id: str, turn_index: int, role: str, content: str
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO turns (curiosity_id, turn_index, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (curiosity_id, turn_index, role, content, time.time()),
        )
        await self._conn.commit()

    async def get_turns(self, curiosity_id: str) -> list[dict]:
        async with self._conn.execute(
            "SELECT * FROM turns WHERE curiosity_id=? ORDER BY turn_index",
            (curiosity_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ------------------------------------------------------------------
    # analytics events
    # ------------------------------------------------------------------

    async def log_event(
        self,
        event_type: str,
        curiosity_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        now = time.time()
        import datetime
        dt = datetime.datetime.fromtimestamp(now)
        await self._conn.execute(
            """
            INSERT INTO analytics_events
              (event_type, curiosity_id, timestamp, hour_of_day, day_of_week, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                curiosity_id,
                now,
                dt.hour,
                dt.weekday(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # interest scores
    # ------------------------------------------------------------------

    async def upsert_interest_scores(self, scores: dict[str, float]) -> None:
        now = time.time()
        for category, delta in scores.items():
            await self._conn.execute(
                """
                INSERT INTO interest_scores (category, score, mention_count, last_seen_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(category) DO UPDATE SET
                    score = score + excluded.score,
                    mention_count = mention_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (category, delta, now),
            )
        await self._conn.commit()

    async def get_interest_scores(self) -> dict[str, float]:
        async with self._conn.execute(
            "SELECT category, score FROM interest_scores ORDER BY score DESC"
        ) as cur:
            return {r["category"]: r["score"] for r in await cur.fetchall()}

    async def get_top_topics(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the top-n interest categories by cumulative score."""
        async with self._conn.execute(
            "SELECT category, score FROM interest_scores ORDER BY score DESC LIMIT ?",
            (n,),
        ) as cur:
            return [(r["category"], r["score"]) for r in await cur.fetchall()]

    async def get_today_curiosity_nodes(self) -> list[dict]:
        """
        Return today's curiosities as node data for the breadth × depth map,
        in chronological order.

        breadth = number of distinct interest categories classified (domain spread).
        depth   = average depth_signal from the classifier (0 = broad, 1 = focused/deep).
        order   = 1-indexed chronological position within today.
        """
        import datetime
        today_start = datetime.datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        async with self._conn.execute(
            """
            SELECT e.curiosity_id, e.metadata, c.trigger_question, e.timestamp
            FROM analytics_events e
            JOIN curiosities c ON e.curiosity_id = c.id
            WHERE e.event_type = 'interest_classified'
              AND e.timestamp >= ?
            ORDER BY e.timestamp ASC
            """,
            (today_start,),
        ) as cur:
            rows = await cur.fetchall()

        # dict is insertion-ordered (Python 3.7+); first encounter = chronological order
        by_id: dict[str, dict] = {}
        for row in rows:
            cid = row["curiosity_id"]
            meta = json.loads(row["metadata"] or "{}")
            depth = float(meta.get("depth_signal", 0.5))
            categories: dict = meta.get("categories", {})

            if cid not in by_id:
                by_id[cid] = {
                    "question": row["trigger_question"],
                    "depth_vals": [],
                    "categories": set(),
                }
            by_id[cid]["depth_vals"].append(depth)
            by_id[cid]["categories"].update(categories.keys())

        return [
            {
                "question": v["question"],
                "depth": sum(v["depth_vals"]) / len(v["depth_vals"]),
                "breadth": len(v["categories"]),
                "order": idx + 1,
            }
            for idx, (_, v) in enumerate(by_id.items())
        ]

    async def get_hourly_type_data(self, since_hours: int = 24) -> dict[tuple[int, str], int]:
        """
        Return count of each ideation type per hour of day, looking back
        `since_hours` hours from now.

        Returns dict mapping (hour_of_day, ideation_type) → count.
        """
        since = time.time() - since_hours * 3600
        async with self._conn.execute(
            """
            SELECT hour_of_day, metadata
            FROM analytics_events
            WHERE event_type = 'interest_classified'
              AND timestamp >= ?
            """,
            (since,),
        ) as cur:
            rows = await cur.fetchall()

        result: dict[tuple[int, str], int] = {}
        for row in rows:
            hour = int(row["hour_of_day"])
            meta = json.loads(row["metadata"] or "{}")
            itype = meta.get("ideation_type", "speculative")
            key = (hour, itype)
            result[key] = result.get(key, 0) + 1
        return result

    # ------------------------------------------------------------------
    # user profile
    # ------------------------------------------------------------------

    async def set_profile_key(self, key: str, value: Any) -> None:
        await self._conn.execute(
            """
            INSERT INTO user_profile (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value), time.time()),
        )
        await self._conn.commit()

    async def get_profile_key(self, key: str, default: Any = None) -> Any:
        async with self._conn.execute(
            "SELECT value FROM user_profile WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
            return json.loads(row["value"]) if row else default

    async def get_full_profile(self) -> dict[str, Any]:
        async with self._conn.execute("SELECT key, value FROM user_profile") as cur:
            return {r["key"]: json.loads(r["value"]) for r in await cur.fetchall()}

    # ------------------------------------------------------------------
    # cooldown
    # ------------------------------------------------------------------

    async def insert_cooldown(self, starts_at: float, ends_at: float, reason: str) -> None:
        await self._conn.execute(
            "INSERT INTO cooldown_log (started_at, ends_at, reason) VALUES (?, ?, ?)",
            (starts_at, ends_at, reason),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # metrics for display
    # ------------------------------------------------------------------

    async def get_display_metrics(self) -> dict:
        """Aggregate stats suitable for the e-ink display."""
        metrics: dict[str, Any] = {}

        # total curiosities
        async with self._conn.execute(
            "SELECT COUNT(*) as n FROM curiosities"
        ) as cur:
            metrics["total_curiosities"] = (await cur.fetchone())["n"]

        # completed vs saved vs ignored
        async with self._conn.execute(
            "SELECT status, COUNT(*) as n FROM curiosities GROUP BY status"
        ) as cur:
            for row in await cur.fetchall():
                metrics[f"curiosities_{row['status']}"] = row["n"]

        # top 3 interest categories
        async with self._conn.execute(
            "SELECT category, score FROM interest_scores ORDER BY score DESC LIMIT 3"
        ) as cur:
            metrics["top_interests"] = [
                {"category": r["category"], "score": round(r["score"], 1)}
                for r in await cur.fetchall()
            ]

        # engagement by hour (last 7 days)
        week_ago = time.time() - 7 * 86400
        async with self._conn.execute(
            """
            SELECT hour_of_day, COUNT(*) as n
            FROM analytics_events
            WHERE event_type='curiosity_started' AND timestamp > ?
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """,
            (week_ago,),
        ) as cur:
            metrics["engagement_by_hour"] = {
                r["hour_of_day"]: r["n"] for r in await cur.fetchall()
            }

        # answered vs ignored ratio
        async with self._conn.execute(
            """
            SELECT
              SUM(CASE WHEN event_type='turn_answered' THEN 1 ELSE 0 END) as answered,
              SUM(CASE WHEN event_type='curiosity_ignored' THEN 1 ELSE 0 END) as ignored
            FROM analytics_events
            WHERE timestamp > ?
            """,
            (week_ago,),
        ) as cur:
            row = await cur.fetchone()
            metrics["answered_7d"] = row["answered"] or 0
            metrics["ignored_7d"] = row["ignored"] or 0

        return metrics
