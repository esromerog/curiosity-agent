"""
E-ink display driver (Waveshare 4.2" V2, 400x300).

Displays a scrollable backlog of curiosity questions.
Scrolling is controlled by a rotary encoder on GPIO 14 & 15.
Falls back gracefully on non-RPi machines (saves PNG preview to /tmp/).
"""

from __future__ import annotations

import asyncio
import textwrap
import threading
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from waveshare_epd import epd4in2_V2 as epd_driver
    EPD_AVAILABLE = True
except ImportError:
    EPD_AVAILABLE = False

try:
    from gpiozero import RotaryEncoder
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False


_FONT_DIR = Path(__file__).parent / "fonts"
_W, _H = 400, 300
_BG = 255  # white
_FG = 0    # black
_GRAY = 128
_VISIBLE = 3  # questions visible at once


def _load_font(size: int) -> Any:
    try:
        return ImageFont.truetype(str(_FONT_DIR / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


class EinkDisplay:
    def __init__(self, encoder_pin_a: int = 14, encoder_pin_b: int = 15) -> None:
        self._epd = None
        self._encoder: RotaryEncoder | None = None
        self._pin_a = encoder_pin_a
        self._pin_b = encoder_pin_b
        self._questions: list[dict] = []
        self._offset = 0
        self._redraw_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        if EPD_AVAILABLE:
            self._epd = epd_driver.EPD()
            self._epd.init()
            self._epd.Clear()
            logger.info("E-ink display initialised.")
        else:
            logger.warning("Waveshare EPD not available — display will save PNG to /tmp/.")

        if ENCODER_AVAILABLE:
            # swap pin_a / pin_b in config if rotation direction is inverted
            self._encoder = RotaryEncoder(self._pin_a, self._pin_b, max_steps=0)
            self._encoder.when_rotated_clockwise = self._on_scroll_down
            self._encoder.when_rotated_counter_clockwise = self._on_scroll_up
            logger.info(f"Rotary encoder ready on GPIO {self._pin_a}/{self._pin_b}.")
        else:
            logger.warning("gpiozero not available — rotary encoder disabled.")

        self._task = asyncio.create_task(self._redraw_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._encoder:
            self._encoder.close()
        if self._epd:
            self._epd.sleep()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_questions(self, questions: list[dict]) -> None:
        """Set the full questions list and redraw (scrolled to top)."""
        with self._lock:
            self._questions = list(questions)
            self._offset = 0
        self._redraw_event.set()

    # ------------------------------------------------------------------
    # Rotary encoder callbacks (run in gpiozero background thread)
    # ------------------------------------------------------------------

    def _on_scroll_down(self) -> None:
        with self._lock:
            max_offset = max(0, len(self._questions) - _VISIBLE)
            if self._offset < max_offset:
                self._offset += 1
        if self._loop:
            self._loop.call_soon_threadsafe(self._redraw_event.set)

    def _on_scroll_up(self) -> None:
        with self._lock:
            if self._offset > 0:
                self._offset -= 1
        if self._loop:
            self._loop.call_soon_threadsafe(self._redraw_event.set)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    async def _redraw_loop(self) -> None:
        while True:
            try:
                await self._redraw_event.wait()
                self._redraw_event.clear()
                # debounce: wait for fast scrolling to settle
                await asyncio.sleep(0.3)
                self._redraw_event.clear()
                await self._render()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Display error: {exc}")

    async def _render(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._render_sync)

    def _render_sync(self) -> None:
        with self._lock:
            questions = list(self._questions)
            offset = self._offset

        if not PIL_AVAILABLE:
            self._print_fallback(questions, offset)
            return

        w = self._epd.width if (EPD_AVAILABLE and self._epd) else _W
        h = self._epd.height if (EPD_AVAILABLE and self._epd) else _H
        img = Image.new("1", (w, h), _BG)
        draw = ImageDraw.Draw(img)
        self._draw_questions(draw, questions, offset, w, h)

        if EPD_AVAILABLE and self._epd:
            self._epd.display(self._epd.getbuffer(img))
            logger.debug("E-ink display updated.")
        else:
            img.save("/tmp/curiosity_display_preview.png")
            logger.debug("Display preview saved to /tmp/curiosity_display_preview.png")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_questions(
        self, draw: Any, questions: list[dict], offset: int, w: int = _W, h: int = _H
    ) -> None:
        f_large = _load_font(18)
        f_med   = _load_font(14)
        f_small = _load_font(12)
        f_tiny  = _load_font(10)

        # --- header ---
        draw.text((10, 8), "What you've been wondering about", font=f_large, fill=_FG)
        draw.line([(10, 32), (w - 20, 32)], fill=_FG, width=2)

        # --- questions ---
        visible = questions[offset : offset + _VISIBLE]
        y = 40
        row_h = 72
        max_chars = (w - 50) // 7

        for i, q in enumerate(visible):
            text = q.get("trigger_question", "").strip()
            lines = textwrap.wrap(text, width=max_chars) or [text]

            draw.text((10, y + 2), "–", font=f_med, fill=_FG)
            draw.text((24, y), lines[0], font=f_med, fill=_FG)
            if len(lines) > 1:
                draw.text((24, y + 18), lines[1], font=f_small, fill=_FG)

            summary = (q.get("summary") or "").strip()
            if summary:
                short = textwrap.shorten(summary, width=max_chars + 5, placeholder="...")
                draw.text((24, y + 36), short, font=f_tiny, fill=_GRAY)

            if i < len(visible) - 1:
                draw.line([(10, y + row_h - 4), (w - 20, y + row_h - 4)], fill=_GRAY, width=1)
            y += row_h

        if not questions:
            draw.text((10, 60), "No curiosities yet.", font=f_med, fill=_GRAY)

        # --- scroll indicator ---
        total = len(questions)
        if total > _VISIBLE:
            bar_x = w - 10
            bar_top = 40
            bar_bottom = h - 30
            bar_h = bar_bottom - bar_top
            thumb_h = max(10, int(bar_h * _VISIBLE / total))
            max_offset = total - _VISIBLE
            thumb_y = bar_top + int((bar_h - thumb_h) * offset / max_offset) if max_offset else bar_top
            draw.line([(bar_x, bar_top), (bar_x, bar_bottom)], fill=_GRAY, width=1)
            draw.rectangle([bar_x - 2, thumb_y, bar_x + 2, thumb_y + thumb_h], fill=_FG)

        # --- footer ---
        draw.line([(10, h - 18), (w - 20, h - 18)], fill=_GRAY, width=1)
        page_start = offset + 1
        page_end = min(offset + _VISIBLE, total)
        footer = f"{page_start}–{page_end} of {total}" if total else "0 questions"
        draw.text((10, h - 14), footer, font=f_tiny, fill=_GRAY)

    def _print_fallback(self, questions: list[dict], offset: int) -> None:
        print("\n" + "=" * 60)
        print("  WHAT YOU'VE BEEN WONDERING ABOUT")
        print("=" * 60)
        visible = questions[offset : offset + _VISIBLE]
        for q in visible:
            print(f"  – {q.get('trigger_question', '')}")
            if q.get("summary"):
                print(f"      {q['summary']}")
        if not questions:
            print("  No curiosities yet.")
        total = len(questions)
        print("-" * 60)
        print(f"  Showing {offset + 1}–{min(offset + _VISIBLE, total)} of {total}")
        print("=" * 60 + "\n")
