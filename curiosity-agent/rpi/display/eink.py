"""
E-ink display driver (Waveshare 7.5" V2, 800×480).

Renders a digestible metrics dashboard using Pillow.
Falls back gracefully on non-RPi machines (prints to stdout).

Install Waveshare driver:
  git clone https://github.com/waveshare/e-Paper
  pip install ./e-Paper/RaspberryPi_JetsonNano/python/
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
    from waveshare_epd import epd7in5_V2 as epd_driver
    EPD_AVAILABLE = True
except ImportError:
    EPD_AVAILABLE = False


_FONT_DIR = Path(__file__).parent / "fonts"
_W, _H = 800, 480
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
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._render_sync, metrics)

    async def _render(self, metrics: dict) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._render_sync, metrics)

    def _render_sync(self, metrics: dict) -> None:
        if not PIL_AVAILABLE:
            self._print_fallback(metrics)
            return

        img = Image.new("1", (_W, _H), _BG)
        draw = ImageDraw.Draw(img)
        self._draw_dashboard(draw, metrics)

        if EPD_AVAILABLE and self._epd:
            self._epd.display(self._epd.getbuffer(img))
            logger.debug("E-ink display updated.")
        else:
            img.save("/tmp/curiosity_display_preview.png")
            logger.debug("Display preview saved to /tmp/curiosity_display_preview.png")

    def _draw_dashboard(self, draw: Any, m: dict) -> None:
        f_large = _load_font(28)
        f_med   = _load_font(20)
        f_small = _load_font(16)

        # --- header ---
        draw.text((20, 10), "Curiosity Dashboard", font=f_large, fill=_FG)
        draw.line([(20, 48), (_W - 20, 48)], fill=_FG, width=2)

        # --- column 1: interests ---
        x1 = 20
        y = 60
        draw.text((x1, y), "Top Interests", font=f_med, fill=_FG)
        y += 28
        interests = m.get("top_interests", [])
        for item in interests[:5]:
            name = item["name"].capitalize()
            score = item["score"]
            bar_w = min(int(score * 4), 140)
            draw.text((x1, y), name, font=f_small, fill=_FG)
            draw.rectangle([x1 + 100, y + 4, x1 + 100 + bar_w, y + 16], fill=_FG)
            y += 24

        # --- column 2: style ---
        x2 = 320
        y = 60
        draw.text((x2, y), "Thinking Style", font=f_med, fill=_FG)
        y += 28
        draw.text((x2, y), m.get("focus_style", "—"), font=f_small, fill=_FG)
        y += 24
        draw.text((x2, y), m.get("dominant_ideation", "—"), font=f_small, fill=_FG)

        # --- column 3: session counts ---
        x3 = 560
        y = 60
        draw.text((x3, y), "Sessions", font=f_med, fill=_FG)
        y += 28
        stats = [
            ("Total", m.get("total_curiosities", 0)),
            ("Completed", m.get("completed", 0)),
            ("Saved", m.get("saved", 0)),
            ("Ignored", m.get("ignored", 0)),
        ]
        for label, val in stats:
            draw.text((x3, y), f"{label}: {val}", font=f_small, fill=_FG)
            y += 22

        # --- peak hours ---
        y = 310
        draw.line([(20, y - 10), (_W - 20, y - 10)], fill=_FG, width=1)
        draw.text((20, y), "Peak Hours (7-day)", font=f_med, fill=_FG)
        peak = m.get("peak_hours", [])
        draw.text((20, y + 28), "  ".join(peak) if peak else "—", font=f_small, fill=_FG)

        # --- engagement ratio ---
        answered = m.get("answered_7d", 0)
        ignored  = m.get("ignored_7d", 0)
        total    = answered + ignored
        ratio_str = f"{answered}/{total} engaged (7d)" if total else "No data yet"
        draw.text((320, y), "Engagement", font=f_med, fill=_FG)
        draw.text((320, y + 28), ratio_str, font=f_small, fill=_FG)

        # --- footer ---
        draw.line([(20, _H - 30), (_W - 20, _H - 30)], fill=_FG, width=1)
        draw.text((20, _H - 22), "curiosity-agent  |  press to refresh", font=f_small, fill=_GRAY)

    def _print_fallback(self, metrics: dict) -> None:
        print("\n" + "=" * 50)
        print("  CURIOSITY DASHBOARD")
        print("=" * 50)
        print(f"  Focus: {metrics.get('focus_style', '—')}  |  Ideation: {metrics.get('dominant_ideation', '—')}")
        print(f"  Sessions: {metrics.get('total_curiosities', 0)} total, "
              f"{metrics.get('completed', 0)} completed, "
              f"{metrics.get('saved', 0)} saved")
        print("  Top interests:")
        for item in metrics.get("top_interests", [])[:5]:
            print(f"    {item['name'].capitalize()}: {item['score']:.1f}")
        print("=" * 50 + "\n")
