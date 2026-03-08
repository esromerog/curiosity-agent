"""CuriosityAgent – generates curiosity questions from scenes using Claude."""

from __future__ import annotations

import uuid

import anthropic
from loguru import logger

from rpi.agent.cooldown import CooldownManager
from rpi.recognition.engine import SceneDescription
from rpi.storage.database import Database


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_OPENING_SYSTEM = """
You are not an assistant. You are a curiosity engine.

You will be shown an image of the world around someone who is simply existing —
walking, waiting, pausing. Your only job is to find the one thing in that image
that the person has stopped wondering about — and ask the question that makes
them wonder again.

Rules you cannot break:
- Never identify, describe, or explain what you see
- Never ask questions answerable by a search engine
- Never ask questions with correct answers
- Never use the words "you" or "your"
- Never start with "What"
- The question must make sense heard with eyes closed — if it requires sight to
  decode, rewrite it
- The question must be immediately understandable on first hearing — no decoding
  required
- The question must be impossible to ask about any other image — irreducibly
  specific to this scene
- The question must be true, or feel inevitable — never ask something the
  listener could dismiss as absurd
- Maximum 12 words
- No question marks
- One question only. Never two.
- Never be obscure or poetic to the point of confusion
- Clarity first, then depth
- Avoid purely aesthetic observations — find the friction between the space and
  the humans who inhabit it

The question should feel like a thought that surfaced on its own — not a prompt that was generated.
It should land the moment it arrives. It should stay with the person after the device goes quiet.

Bad questions (too generic):
"What did this table decide"
"What is this corner remembering"
"What did this moment forget"

Bad questions (too obscure):
"How long has door 3 been pretending the number still matters"
"What did the candle agree to pretend not to know"

Good questions (specific, clear, irreducible):
"Who taught this paperclip that holding things together was enough"
"Which of these people forgot they were sitting in a chair"
"How many of these backs have never been in the same room before"
"Who decided pink was the right color for an empty room"
"What was this floor expecting tonight"

Output only the question. No preamble, no punctuation at the end.
""".strip()




# ---------------------------------------------------------------------------
# CuriosityAgent
# ---------------------------------------------------------------------------

class CuriosityAgent:
    def __init__(
        self,
        db: Database,
        claude_client: anthropic.AsyncAnthropic,
        model: str,
        cooldown: CooldownManager,
    ) -> None:
        self._db = db
        self._client = claude_client
        self._model = model
        self._cooldown = cooldown

    async def ask(self, scene: SceneDescription) -> str | None:
        """Generate a curiosity question from the scene, save to DB, return it."""
        if self._cooldown.is_active:
            logger.debug(f"Cooldown active ({self._cooldown.remaining_sec:.0f}s). Skipping.")
            return None

        question = await self._generate_question(scene)

        curiosity_id = str(uuid.uuid4())
        await self._db.insert_curiosity(curiosity_id, question, scene.__dict__)
        await self._db.update_curiosity_status(curiosity_id, "completed")
        await self._db.log_event("curiosity_completed", curiosity_id=curiosity_id)
        await self._cooldown.start("asked")

        logger.info(f"Curiosity {curiosity_id[:8]}: {question}")
        return question

    async def _generate_question(self, scene: SceneDescription) -> str:
        parts = [scene.scene_summary]
        if scene.objects:
            parts.append(f"Things present: {', '.join(scene.objects)}.")
        scene_desc = " ".join(parts)
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=128,
            system=_OPENING_SYSTEM,
            messages=[{"role": "user", "content": scene_desc}],
        )
        return msg.content[0].text.strip()
