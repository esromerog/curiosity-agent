"""
E-ink display driver (Waveshare 4.2" V2, 400x300).

Displays a backlog of past curiosity questions and top interest themes.
Falls back gracefully on non-RPi machines (saves PNG preview to /tmp/).

Install Waveshare driver:
  git clone https://github.com/waveshare/e-Paper
  pip install ./e-Paper/RaspberryPi_JetsonNano/python/

Requires SPI enabled on the RPi:
  sudo raspi-config -> Interface Options -> SPI -> Yes
"""

from __future__ import annotations

import asyncio
import textwrap
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


_FONT_DIR = Path(__file__).parent / "fonts"
_W, _H = 400, 300
_BG = 255   # white
_FG = 0     # black
_GRAY = 128


def _load_font(size: int) -> Any:
    try:
        return ImageFont.truetype(str(_FONT_DIR / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


class EinkDisplay:
    def __init__(self, refresh_minutes: float = 5.0) -> None:
        self._refresh_interval = refresh_minutes * 60
        self._epd = None
        self._task: asyncio.Task | None = None

    async def start(self, metrics_fn) -> None:
        """
        Start the background refresh loop.
        metrics_fn is an async callable returning a metrics dict.
        """
        if EPD_AVAILABLE:
            self._epd = epd_driver.EPD()
            self._epd.init()
            self._epd.Clear()
            logger.info("E-ink display initialised.")
        else:
            logger.warning("Waveshare EPD not available — display will print to stdout.")

        self._task = asyncio.create_task(self._loop(metrics_fn))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._epd:
            self._epd.sleep()

    async def _loop(self, metrics_fn) -> None:
        while True:
            try:
                metrics = await metrics_fn()
                await self._render(metrics)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Display error: {exc}")
            await asyncio.sleep(self._refresh_interval)

    async def render_now(self, metrics: dict) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._render_sync, metrics)

    async def _render(self, metrics: dict) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._render_sync, metrics)

    def _render_sync(self, metrics: dict) -> None:
        if not PIL_AVAILABLE:
            self._print_fallback(metrics)
            return

        w = self._epd.width if (EPD_AVAILABLE and self._epd) else _W
        h = self._epd.height if (EPD_AVAILABLE and self._epd) else _H
        img = Image.new("1", (w, h), _BG)
        draw = ImageDraw.Draw(img)
        self._draw_dashboard(draw, metrics, w, h)

        if EPD_AVAILABLE and self._epd:
            self._epd.display(self._epd.getbuffer(img))
            logger.debug("E-ink display updated.")
        else:
            img.save("/tmp/curiosity_display_preview.png")
            logger.debug("Display preview saved to /tmp/curiosity_display_preview.png")

    def _draw_dashboard(self, draw: Any, m: dict, w: int = _W, h: int = _H) -> None:
        # Font sizes tuned for 400x300 display
        f_large = _load_font(18)
        f_med   = _load_font(14)
        f_small = _load_font(12)
        f_tiny  = _load_font(10)

        # --- header ---
        # Layout: header 0-38px, body 44-252px (3 rows x 69px), themes 252-278px, footer 278-300px
        draw.text((10, 8), "What you've been wondering about", font=f_large, fill=_FG)
        draw.line([(10, 32), (w - 10, 32)], fill=_FG, width=2)

        # --- questions backlog (main body) ---
        questions = m.get("recent_questions", [])
        y = 40
        row_h = 69  # 3 rows fit in 207px body
        max_chars = (w - 40) // 7  # chars per line at ~7px/char for size 14
        for i, q in enumerate(questions[:3]):
            x = 10
            text = q.get("trigger_question", "").strip()
            lines = textwrap.wrap(text, width=max_chars) or [text]
            draw.text((x, y + 2), "-", font=f_med, fill=_FG)
            x += 14
            draw.text((x, y), lines[0], font=f_med, fill=_FG)
            if len(lines) > 1:
                draw.text((x, y + 16), lines[1], font=f_small, fill=_FG)
            summary = (q.get("summary") or "").strip()
            if summary:
                summary_wrapped = textwrap.shorten(summary, width=max_chars + 5, placeholder="...")
                draw.text((x, y + 32), summary_wrapped, font=f_tiny, fill=_GRAY)
            if i < len(questions) - 1:
                draw.line([(10, y + row_h - 4), (w - 10, y + row_h - 4)], fill=_GRAY, width=1)
            y += row_h

        if not questions:
            draw.text((10, 60), "No completed curiosities yet.", font=f_med, fill=_GRAY)

        # --- themes strip ---
        themes_y = h - 46
        draw.line([(10, themes_y - 4), (w - 10, themes_y - 4)], fill=_FG, width=1)
        draw.text((10, themes_y), "Themes:", font=f_tiny, fill=_FG)
        interests = m.get("top_interests", [])
        tag_x = 68
        for item in interests[:5]:
            label = item["name"].capitalize()
            tag_w = len(label) * 7 + 8
            if tag_x + tag_w > w - 10:
                break
            draw.rectangle([tag_x, themes_y, tag_x + tag_w, themes_y + 14], outline=_FG)
            draw.text((tag_x + 4, themes_y + 1), label, font=f_tiny, fill=_FG)
            tag_x += tag_w + 6

        # --- footer ---
        draw.line([(10, h - 18), (w - 10, h - 18)], fill=_GRAY, width=1)
        total = m.get("total_curiosities", 0)
        style = m.get("focus_style", "")
        footer = f"{total} curiosities  |  {style}" if style else f"{total} curiosities"
        draw.text((10, h - 14), footer, font=f_tiny, fill=_GRAY)

    def _print_fallback(self, metrics: dict) -> None:
        print("\n" + "=" * 60)
        print("  WHAT YOU'VE BEEN WONDERING ABOUT")
        print("=" * 60)
        for q in metrics.get("recent_questions", []):
            print(f"  – {q.get('trigger_question', '')}")
            if q.get("summary"):
                print(f"      {q['summary']}")
        if not metrics.get("recent_questions"):
            print("  No completed curiosities yet.")
        print("-" * 60)
        themes = "  ".join(i["name"].capitalize() for i in metrics.get("top_interests", [])[:6])
        print(f"  Themes: {themes or '—'}")
        print(f"  {metrics.get('total_curiosities', 0)} curiosities  ·  {metrics.get('focus_style', '')}")
        print("=" * 60 + "\n")
