"""
Interest categorizer.

Uses Claude to classify a user turn into one or more interest categories
and to determine the ideation type, then persists the scores.
"""

from __future__ import annotations

import json

import anthropic
from loguru import logger

from rpi.storage.database import Database


_CATEGORIES = [
    "history", "economics", "mathematics", "science", "fitness",
    "travel", "design", "philosophy", "technology", "arts",
    "language", "nature", "culture", "psychology", "politics",
]

# Question-style taxonomy used by the stacked-area view.
# speculative  – "what if" / future-oriented thinking
# investigative – fact-seeking, analytical, cause-and-effect
# ideation      – creative, generative, design-oriented
# empirical     – observation-based, evidence-driven
_IDEATION_TYPES = ["speculative", "investigative", "ideation", "empirical"]

_CLASSIFY_PROMPT = f"""
You are an interest classifier. Given a user message from a curiosity-driven conversation,
identify which interest categories are present and what ideation type it represents.

Categories: {', '.join(_CATEGORIES)}
Ideation types: {', '.join(_IDEATION_TYPES)}
  speculative  = "what if" / future-oriented / hypothetical
  investigative = fact-seeking / analytical / cause-and-effect
  ideation      = creative / generative / design-oriented
  empirical     = observation-based / evidence-driven / measurement

Respond with JSON only:
{{
  "categories": {{"category_name": relevance_score_0_to_1, ...}},
  "ideation_type": "one of the ideation types",
  "depth_signal": 0.0_to_1.0  // 1.0 = very focused/deep, 0.0 = broad/exploratory
}}
Include only categories with relevance > 0.2. Output valid JSON, no markdown.
""".strip()


class InterestCategorizer:
    def __init__(
        self,
        db: Database,
        claude_client: anthropic.AsyncAnthropic,
        model: str,
    ) -> None:
        self._db = db
        self._client = claude_client
        self._model = model

    async def classify_and_store(self, user_text: str, curiosity_id: str) -> dict:
        """Classify user_text and persist scores. Returns the raw classification."""
        try:
            result = await self._classify(user_text)
            categories: dict[str, float] = result.get("categories", {})
            ideation = result.get("ideation_type", "exploratory")
            depth = result.get("depth_signal", 0.5)

            if categories:
                await self._db.upsert_interest_scores(categories)

            await self._db.log_event(
                "interest_classified",
                curiosity_id=curiosity_id,
                metadata={
                    "categories": categories,
                    "ideation_type": ideation,
                    "depth_signal": depth,
                },
            )
            await self._update_profile(ideation, depth)
            return result
        except Exception as exc:
            logger.error(f"Categorizer error: {exc}")
            return {}

    async def _classify(self, text: str) -> dict:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": f"{_CLASSIFY_PROMPT}\n\nUser message: {text}"}
            ],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)

    async def _update_profile(self, ideation_type: str, depth_signal: float) -> None:
        # rolling average of depth signal
        current_depth = await self._db.get_profile_key("avg_depth_signal", 0.5)
        new_depth = current_depth * 0.8 + depth_signal * 0.2
        await self._db.set_profile_key("avg_depth_signal", round(new_depth, 3))

        # ideation type tally
        tally: dict[str, int] = await self._db.get_profile_key("ideation_tally", {})
        tally[ideation_type] = tally.get(ideation_type, 0) + 1
        await self._db.set_profile_key("ideation_tally", tally)
        dominant = max(tally, key=tally.get)
        await self._db.set_profile_key("dominant_ideation_type", dominant)
