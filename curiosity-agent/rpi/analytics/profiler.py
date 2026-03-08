"""
User profiler.

Aggregates the raw analytics data into a structured hidden profile
and exposes displayable metrics for the e-ink screen.
"""

from __future__ import annotations

from loguru import logger

from rpi.storage.database import Database


class UserProfiler:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_hidden_profile(self) -> dict:
        """Full hidden profile used to personalise future questions."""
        profile = await self._db.get_full_profile()
        scores = await self._db.get_interest_scores()

        top_interests = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "top_interests": [cat for cat, _ in top_interests],
            "interest_scores": scores,
            "dominant_ideation_type": profile.get("dominant_ideation_type", "exploratory"),
            "ideation_tally": profile.get("ideation_tally", {}),
            "avg_depth_signal": profile.get("avg_depth_signal", 0.5),
            "avg_session_depth": profile.get("avg_session_depth", 0.0),
            "save_count": profile.get("save_count", 0),
            "total_sessions_counted": profile.get("total_sessions_counted", 0),
        }

    async def get_display_metrics(self) -> dict:
        """Displayable (non-sensitive) metrics for the e-ink screen."""
        db_metrics = await self._db.get_display_metrics()
        scores = await self._db.get_interest_scores()
        profile = await self._db.get_full_profile()

        top_interests = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # engagement heatmap: hours with highest activity this week
        by_hour = db_metrics.get("engagement_by_hour", {})
        peak_hours = sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_labels = [f"{h:02d}:00" for h, _ in peak_hours]

        # depth vs breadth label
        avg_depth = profile.get("avg_depth_signal", 0.5)
        if avg_depth > 0.65:
            focus_style = "Deep diver"
        elif avg_depth < 0.35:
            focus_style = "Broad explorer"
        else:
            focus_style = "Balanced"

        return {
            "top_interests": [{"name": c, "score": round(s, 1)} for c, s in top_interests],
            "focus_style": focus_style,
            "dominant_ideation": profile.get("dominant_ideation_type", "—").capitalize(),
            "total_curiosities": db_metrics.get("total_curiosities", 0),
            "completed": db_metrics.get("curiosities_completed", 0),
            "saved": db_metrics.get("curiosities_saved", 0),
            "ignored": db_metrics.get("curiosities_ignored", 0),
            "peak_hours": peak_labels,
            "answered_7d": db_metrics.get("answered_7d", 0),
            "ignored_7d": db_metrics.get("ignored_7d", 0),
        }

    async def build_question_context(self) -> str:
        """
        Returns a prompt context string injected into the opening-question
        system prompt to steer Claude toward the user's interests.
        Not shown to the user.
        """
        profile = await self.get_hidden_profile()
        top = profile["top_interests"][:3]
        ideation = profile["dominant_ideation_type"]
        depth = profile["avg_depth_signal"]

        depth_desc = (
            "prefers deep, focused exploration"
            if depth > 0.65
            else "enjoys broad, wide-ranging exploration"
            if depth < 0.35
            else "enjoys both focused and wide-ranging exploration"
        )

        if top:
            interests_str = ", ".join(top)
            return (
                f"The user tends to be interested in {interests_str}. "
                f"Their questioning style is {ideation}. "
                f"They {depth_desc}. "
                "Use this to craft a question that will particularly resonate with them."
            )
        return ""
