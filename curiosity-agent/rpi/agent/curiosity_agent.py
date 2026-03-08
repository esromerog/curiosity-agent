"""
CuriosityAgent – the core Claude-powered conversational agent.

Responsibilities:
  1. Generate an insightful opening question from a scene description.
  2. Conduct the multi-turn dialogue.
  3. Detect save / end / ignore / reject signals from the user.
  4. Summarise the curiosity on close.
  5. Trigger analytics classification after each user turn.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable

import anthropic
from loguru import logger

from rpi.agent.session import CuriositySession, SessionStatus
from rpi.agent.cooldown import CooldownManager
from rpi.analytics.categorizer import InterestCategorizer
from rpi.analytics.tracker import AnalyticsTracker
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

_CONVERSATION_SYSTEM = """
You are not an assistant. You are a curiosity engine in dialogue.

The conversation was opened by a question that surfaced from this scene:
{scene_summary}

Your role is to follow the thread wherever it leads — not to explain, not to
conclude, but to keep the wondering alive.

Guidelines:
- Respond to what was said, then push one level deeper — never sideways just
  to be interesting
- Answers should be 2-4 sentences: enough to move the thought forward, not
  enough to close it
- End every response with a follow-up that couldn't have been asked before this
  exact exchange happened
- Never summarise what was just said back to the listener
- Never use filler phrases ("great question", "absolutely", "certainly")
- Never reveal that interests or patterns are being tracked
- If the thread reaches a genuinely open question neither side can resolve, say
  so — and sit with it
- Speak as if thinking aloud, not presenting
""".strip()

_SUMMARY_PROMPT = """
In one sentence, name the question this conversation was actually about — not
the topic on the surface, but the deeper unresolved tension underneath it.
Do not explain. Do not use the word "conversation". Output only the sentence.
""".strip()


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_SAVE_PHRASES = frozenset([
    "that's all for now", "thats all for now",
    "continue this later", "save this", "pause", "save",
])

_END_PHRASES = frozenset(["goodbye", "done", "stop", "exit", "end", "finish"])

_REJECT_PHRASES = frozenset(["no", "not now", "not interested", "skip", "pass", "nope"])

_IGNORE_TIMEOUT_SEC = 30  # seconds to wait for first user response


def _detect_intent(text: str) -> str | None:
    """Return 'save' | 'end' | 'reject' | None."""
    lowered = text.strip().lower().rstrip(".")
    if any(lowered == p or lowered.startswith(p) for p in _SAVE_PHRASES):
        return "save"
    if any(lowered == p for p in _END_PHRASES):
        return "end"
    if any(lowered == p for p in _REJECT_PHRASES):
        return "reject"
    return None


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
        categorizer: InterestCategorizer,
        tracker: AnalyticsTracker,
        max_turns: int = 20,
    ) -> None:
        self._db = db
        self._client = claude_client
        self._model = model
        self._cooldown = cooldown
        self._categorizer = categorizer
        self._tracker = tracker
        self._max_turns = max_turns
        self._active_session: CuriositySession | None = None

    @property
    def is_active(self) -> bool:
        return self._active_session is not None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def trigger(
        self,
        scene: SceneDescription,
        speak: Callable[[str], Awaitable[None]],
        listen: Callable[[float], Awaitable[str | None]],
    ) -> None:
        """
        Full lifecycle of one curiosity:
          1. Generate opening question.
          2. Speak it.
          3. Wait for user response (with ignore timeout).
          4. Loop through dialogue turns.
          5. Handle save / end / max-turns.
        """
        if self._cooldown.is_active:
            logger.debug(f"Cooldown active ({self._cooldown.remaining_sec:.0f}s remaining). Skipping.")
            return
        if self.is_active:
            logger.debug("A curiosity is already active.")
            return

        # --- step 1: generate opening question ---
        question = await self._generate_opening_question(scene)
        session = CuriositySession(self._db, question, scene.__dict__)
        await session.begin()
        self._active_session = session

        # --- step 2: speak the question ---
        await session.add_turn("assistant", question)
        await speak(question)

        # --- step 3: wait for first response (ignore detection) ---
        user_text = await listen(_IGNORE_TIMEOUT_SEC)
        if not user_text:
            logger.info("No response to opening question → ignore → cooldown.")
            await session.end(SessionStatus.IGNORED)
            await self._cooldown.start("ignored")
            self._active_session = None
            return

        # --- step 4: dialogue loop ---
        await self._dialogue_loop(session, user_text, speak, listen)
        self._active_session = None

    async def resume(
        self,
        curiosity_id: str,
        speak: Callable[[str], Awaitable[None]],
        listen: Callable[[float], Awaitable[str | None]],
    ) -> None:
        """Resume a previously saved curiosity."""
        from rpi.agent.session import SessionStore
        store = SessionStore(self._db)
        history = await store.resume(curiosity_id)
        row = await self._db.get_curiosity(curiosity_id)
        if not row:
            logger.error(f"Curiosity {curiosity_id} not found.")
            return

        scene = row.get("scene_context") or {}
        session = CuriositySession(self._db, row["trigger_question"], scene)
        session.id = curiosity_id
        # replay history into in-memory turns list (no DB re-insert)
        from rpi.agent.session import Turn
        for h in history:
            session.turns.append(Turn(role=h["role"], content=h["content"]))

        self._active_session = session
        await self._db.update_curiosity_status(curiosity_id, SessionStatus.ACTIVE.value)

        resume_msg = "The thread is still open."
        await speak(resume_msg)
        await session.add_turn("assistant", resume_msg)

        user_text = await listen(60.0)
        if user_text:
            await self._dialogue_loop(session, user_text, speak, listen)
        self._active_session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate_opening_question(self, scene: SceneDescription) -> str:
        # Frame the scene around what is happening and who is present,
        # not as a catalogue of objects — keeps Claude focused on friction
        # between space and human rather than object identification.
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

    async def _dialogue_loop(
        self,
        session: CuriositySession,
        first_user_text: str,
        speak: Callable[[str], Awaitable[None]],
        listen: Callable[[float], Awaitable[str | None]],
    ) -> None:
        user_text = first_user_text

        while session.turn_count < self._max_turns * 2:
            intent = _detect_intent(user_text)

            if intent == "save":
                summary = await self._summarise(session)
                await session.end(SessionStatus.SAVED, summary)
                await speak("Saved. The thread stays open.")
                return

            if intent == "end":
                summary = await self._summarise(session)
                await session.end(SessionStatus.COMPLETED, summary)
                await speak("The thread closes here.")
                return

            if intent == "reject":
                await session.end(SessionStatus.REJECTED)
                await self._cooldown.start("rejected")
                await speak("Understood.")
                return

            # normal turn
            await session.add_turn("user", user_text)

            # classify interests in background
            asyncio.create_task(
                self._categorizer.classify_and_store(user_text, session.id)
            )

            # generate assistant reply
            reply = await self._generate_reply(session)
            await session.add_turn("assistant", reply)
            await speak(reply)

            # update tracker
            await self._tracker.record_turn(session)

            # listen for next input
            user_text = await listen(60.0)
            if not user_text:
                # user went quiet mid-conversation → save automatically
                summary = await self._summarise(session)
                await session.end(SessionStatus.SAVED, summary)
                await speak("Thread saved. It can resume.")
                return

        # max turns reached
        summary = await self._summarise(session)
        await session.end(SessionStatus.COMPLETED, summary)
        await speak("The thread has run its course.")

    async def _generate_reply(self, session: CuriositySession) -> str:
        system = _CONVERSATION_SYSTEM.format(
            scene_summary=session.scene_context.get("scene_summary", "an interesting scene")
            if session.scene_context else "an interesting scene"
        )
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=system,
            messages=session.message_history,
        )
        return msg.content[0].text.strip()

    async def _summarise(self, session: CuriositySession) -> str:
        if not session.turns:
            return ""
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[
                *session.message_history,
                {"role": "user", "content": _SUMMARY_PROMPT},
            ],
        )
        return msg.content[0].text.strip()
