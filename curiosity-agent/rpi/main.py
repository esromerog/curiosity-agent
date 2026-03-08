"""
Curiosity Agent — main entry point for the Raspberry Pi.

Startup sequence:
  1. Load config
  2. Connect to SQLite
  3. Start camera receiver HTTP server
  4. Start e-ink display refresh loop
  5. Enter main loop:
       a. Wait for a frame from ESP32
       b. Run object recognition
       c. If no cooldown and no active session → trigger a curiosity
       d. Audio I/O handles the conversation
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv
from loguru import logger

# local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rpi.agent.cooldown import CooldownManager
from rpi.agent.curiosity_agent import CuriosityAgent
from rpi.analytics.categorizer import InterestCategorizer
from rpi.analytics.profiler import UserProfiler
from rpi.analytics.tracker import AnalyticsTracker
from rpi.audio.stt import STT
from rpi.audio.tts import TTS
from rpi.camera.receiver import CameraReceiver
from rpi.camera.puller import CameraPuller
from rpi.display.eink import EinkDisplay
from rpi.recognition.engine import RecognitionEngine
from rpi.storage.database import Database


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(Path(__file__).parent.parent / path) as f:
        raw = yaml.safe_load(f)
    # substitute env vars in api_key
    api_key = raw["anthropic"]["api_key"]
    if api_key.startswith("${") and api_key.endswith("}"):
        var = api_key[2:-1]
        raw["anthropic"]["api_key"] = os.environ.get(var, "")
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    load_dotenv()
    cfg = load_config()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("data/curiosity.log", rotation="10 MB", level="DEBUG")

    # ---- database ---
    db = Database(cfg["storage"]["db_path"])
    await db.connect()
    logger.info("Database ready.")

    # ---- Claude client ---
    api_key = cfg["anthropic"]["api_key"]
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Exiting.")
        sys.exit(1)
    claude = anthropic.AsyncAnthropic(api_key=api_key)
    model  = cfg["anthropic"]["model"]

    # ---- services ---
    receiver = CameraReceiver(
        host=cfg["esp32"]["host"],
        port=cfg["esp32"]["port"],
    )
    recognition = RecognitionEngine(
        backend=cfg["recognition"]["backend"],
        claude_client=claude,
        claude_model=cfg["anthropic"]["vision_model"],
        yolo_model_path=cfg["recognition"]["yolo_model"],
        confidence_threshold=cfg["recognition"]["confidence_threshold"],
    )
    tts = TTS(
        engine=cfg["audio"]["tts_engine"],
        voice=cfg["audio"]["tts_voice"],
        output_device=cfg["audio"]["output_device_index"],
    )
    stt = STT(
        model_name=cfg["audio"]["stt_model"],
        input_device=cfg["audio"]["input_device_index"],
        silence_threshold_sec=cfg["audio"]["silence_threshold_sec"],
    )
    cooldown   = CooldownManager(db, cooldown_minutes=cfg["agent"]["cooldown_minutes"])
    categorizer = InterestCategorizer(db, claude, model)
    tracker    = AnalyticsTracker(db)
    profiler   = UserProfiler(db)
    agent      = CuriosityAgent(
        db=db,
        claude_client=claude,
        model=model,
        cooldown=cooldown,
        categorizer=categorizer,
        tracker=tracker,
        max_turns=cfg["agent"]["max_conversation_turns"],
    )
    display = EinkDisplay(refresh_minutes=cfg["display"]["metrics_refresh_minutes"])

    # ---- start background services ---
    await receiver.start()

    # pull mode: fetch frames from ESP32 CameraWebServer instead of waiting for POSTs
    puller: CameraPuller | None = None
    if cfg["esp32"].get("pull_mode"):
        puller = CameraPuller(
            capture_url=cfg["esp32"]["pull_url"],
            frame_queue=receiver.frame_queue,
            interval_sec=cfg["esp32"]["capture_interval_sec"],
        )
        await puller.start()

    await display.start(metrics_fn=profiler.get_display_metrics)

    logger.info("Curiosity Agent is running. Waiting for frames from ESP32…")

    # ---- main loop ---
    try:
        await _run_loop(
            cfg=cfg,
            receiver=receiver,
            recognition=recognition,
            agent=agent,
            profiler=profiler,
            tts=tts,
            stt=stt,
            tracker=tracker,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        if puller:
            await puller.stop()
        await receiver.stop()
        await display.stop()
        await db.close()


async def _run_loop(
    cfg: dict,
    receiver: CameraReceiver,
    recognition: RecognitionEngine,
    agent: CuriosityAgent,
    profiler: UserProfiler,
    tts: TTS,
    stt: STT,
    tracker: AnalyticsTracker,
) -> None:
    capture_interval = cfg["esp32"]["capture_interval_sec"]

    async def speak(text: str) -> None:
        logger.info(f"[AGENT] {text}")
        await tts.speak(text)

    async def listen(timeout: float) -> str | None:
        logger.debug(f"Listening (timeout={timeout}s)…")
        text = await stt.listen(timeout_sec=timeout)
        if text:
            logger.info(f"[USER] {text}")
        return text

    while True:
        # wait for a frame
        frame = await receiver.get_latest_frame(timeout=capture_interval * 2)

        if frame is None:
            logger.debug("No frame received, waiting…")
            continue

        # skip if agent is mid-conversation or cooldown is active
        if agent.is_active or cooldown_active(agent):
            await asyncio.sleep(1)
            continue

        # run recognition
        logger.debug("Analysing scene…")
        scene = await recognition.describe(frame.data)
        logger.info(f"Scene: {scene.scene_summary} | Objects: {scene.objects}")

        if not scene.objects and not scene.scene_summary:
            await asyncio.sleep(capture_interval)
            continue

        # personalise the question with user profile context
        profile_context = await profiler.build_question_context()
        if profile_context:
            # inject into the agent's opening prompt dynamically
            scene.scene_summary = f"{scene.scene_summary}. [{profile_context}]"

        # trigger curiosity
        await agent.trigger(scene=scene, speak=speak, listen=listen)

        # brief pause before next trigger
        await asyncio.sleep(capture_interval)


def cooldown_active(agent: CuriosityAgent) -> bool:
    # access via the agent's cooldown reference
    return agent._cooldown.is_active


if __name__ == "__main__":
    asyncio.run(main())
