"""
E-ink display driver (Waveshare 4.2" V2, 400x300).

Displays a scrollable backlog of curiosity questions.
Scrolling is controlled by a rotary encoder on GPIO 14 & 15.
Falls back gracefully on non-RPi machines (saves PNG preview to /tmp/).
"""

from __future__ import annotations

import asyncio
import io
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
    from gpiozero import Button, RotaryEncoder
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


_FONT_DIR = Path(__file__).parent / "fonts"
_W, _H = 400, 300
_BG = 255  # white
_FG = 0    # black
_GRAY = 128
_VISIBLE = 3  # questions visible at once
_NUM_VIEWS = 4
_VIEW_TITLES = [
    "What you've been wondering about",
    "Today's Curiosity",
    "Types Throughout My Day",
    "Topics that make me curious",
]
# Curiosity-type taxonomy for the stacked-area view (must match categorizer.py)
_TYPE_ORDER = ["speculative", "investigative", "ideation", "empirical"]
# E-ink hatch patterns per type: first is solid black, rest are hatched on white
_TYPE_HATCHES = ["", "/", "x", "."]


def _load_font(size: int) -> Any:
    try:
        return ImageFont.truetype(str(_FONT_DIR / "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


class EinkDisplay:
    def __init__(
        self,
        encoder_pin_a: int = 14,
        encoder_pin_b: int = 15,
        encoder_pin_button: int = 18,
        db: Any = None,
        type_chart_hours: int = 12,
    ) -> None:
        self._epd = None
        self._encoder: RotaryEncoder | None = None
        self._button: "Button | None" = None
        self._pin_a = encoder_pin_a
        self._pin_b = encoder_pin_b
        self._pin_button = encoder_pin_button
        self._db = db                          # Database instance for graph queries
        self._type_chart_hours = type_chart_hours
        self._questions: list[dict] = []
        self._offset = 0
        self._view = 0
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
            self._button = Button(self._pin_button)
            self._button.when_pressed = self._on_button_pressed
            logger.info(
                f"Rotary encoder ready on GPIO {self._pin_a}/{self._pin_b} "
                f"(button GPIO {self._pin_button})."
            )
        else:
            logger.warning("gpiozero not available — rotary encoder disabled.")

        self._task = asyncio.create_task(self._redraw_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._button:
            self._button.close()
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

    async def _fetch_graph_data(self, view: int) -> dict:
        """Async: query the database for whatever the current view needs."""
        if self._db is None:
            return {}
        try:
            if view == 1:
                return {"nodes": await self._db.get_today_curiosity_nodes()}
            elif view == 2:
                return {"hourly_types": await self._db.get_hourly_type_data(self._type_chart_hours)}
            elif view == 3:
                return {"top_topics": await self._db.get_top_topics(5)}
        except Exception as exc:
            logger.error(f"Failed to fetch graph data for view {view}: {exc}")
        return {}

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

    def _on_button_pressed(self) -> None:
        with self._lock:
            self._view = (self._view + 1) % _NUM_VIEWS
            self._offset = 0
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
        with self._lock:
            view = self._view
        graph_data = await self._fetch_graph_data(view)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._render_sync, graph_data)

    def _render_sync(self, graph_data: dict | None = None) -> None:
        with self._lock:
            questions = list(self._questions)
            offset = self._offset
            view = self._view
        graph_data = graph_data or {}

        if not PIL_AVAILABLE:
            self._print_fallback(questions, offset)
            return

        w = self._epd.width if (EPD_AVAILABLE and self._epd) else _W
        h = self._epd.height if (EPD_AVAILABLE and self._epd) else _H

        if view == 0:
            img = Image.new("1", (w, h), _BG)
            draw = ImageDraw.Draw(img)
            self._draw_questions(draw, questions, offset, w, h)
        elif view == 1:
            img = self._render_view_today(graph_data, w, h)
        elif view == 2:
            img = self._render_view_types(graph_data, w, h)
        elif view == 3:
            img = self._render_view_topics(graph_data, w, h)
        else:
            img = Image.new("1", (w, h), _BG)

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
        draw.text((10, 8), "What you've been wondering about", font=f_med, fill=_FG)
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
        # view indicator dots
        dots = " ".join("●" if i == 0 else "○" for i in range(_NUM_VIEWS))
        draw.text((w - 50, h - 14), dots, font=f_tiny, fill=_GRAY)

    # ------------------------------------------------------------------
    # Graph views (matplotlib-based)
    # ------------------------------------------------------------------

    def _mpl_to_pil(self, fig: Any, w: int, h: int) -> "Image.Image":
        """Render a matplotlib figure to an exact w×h 1-bit PIL Image."""
        buf = io.BytesIO()
        # No bbox_inches so the output is exactly figsize×dpi pixels.
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("L")
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        return img.point(lambda p: 0 if p < 128 else 255).convert("1")

    def _mpl_base_fig(self, title: str, view_idx: int, w: int, h: int, **layout_kw):
        """Return (fig, ax) sized exactly w×h px at 100 dpi, with title and dot indicator."""
        fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        margins = dict(top=0.86, bottom=0.20, left=0.18, right=0.97)
        margins.update(layout_kw)
        fig.subplots_adjust(**margins)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
        dots = "   ".join("\u25cf" if i == view_idx else "\u25cb" for i in range(_NUM_VIEWS))
        fig.text(0.5, 0.02, dots, ha="center", fontsize=7, color="#555555")
        return fig, ax

    def _render_view_today(self, graph_data: dict, w: int, h: int) -> "Image.Image":
        """
        View 1 — Today's Curiosity: breadth × depth scatter map.

        Each node is a curiosity from today.
          x (breadth) = number of distinct interest domains touched.
          y (depth)   = average depth_signal from the classifier (0=broad → 1=deep).
        Labels show the first four words of the trigger question.
        """
        if not MPL_AVAILABLE:
            img = Image.new("1", (w, h), _BG)
            self._draw_fallback_title(ImageDraw.Draw(img), "Today's Curiosity", 1, w, h)
            return img

        nodes = graph_data.get("nodes", [])
        fig, ax = self._mpl_base_fig("Today's Curiosity", 1, w, h, left=0.15, bottom=0.18)

        if not nodes:
            ax.text(0.5, 0.5, "No curiosities logged today yet",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_xlim(0, 5)
            ax.set_ylim(-0.05, 1.05)
        else:
            xs = [n["breadth"] for n in nodes]
            ys = [n["depth"]   for n in nodes]
            ax.scatter(xs, ys, s=28, c="black", zorder=3, linewidths=0)

            for i, (x, y, node) in enumerate(zip(xs, ys, nodes)):
                label = " ".join(node["question"].split()[:4])
                dy = 5 if i % 2 == 0 else -10
                ax.annotate(
                    label, (x, y),
                    xytext=(3, dy), textcoords="offset points",
                    fontsize=5, va="bottom" if dy > 0 else "top",
                    clip_on=True,
                )

            ax.set_xlim(-0.3, max(5, max(xs)) + 0.8)
            ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel("Breadth  (no. of domains)", fontsize=7)
        ax.set_ylabel("Depth", fontsize=7)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["broad", "mid", "deep"], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(True, linestyle=":", linewidth=0.4, color="gray")
        return self._mpl_to_pil(fig, w, h)

    def _render_view_types(self, graph_data: dict, w: int, h: int) -> "Image.Image":
        """
        View 2 — Types Throughout My Day: stacked-area chart.

        Four curiosity types stacked on the y-axis against hour-of-day on x.
        E-ink friendly: solid-black base + distinct hatch patterns per band.
        The x-axis span is controlled by display.curiosity_type_hours in config.
        """
        if not MPL_AVAILABLE:
            img = Image.new("1", (w, h), _BG)
            self._draw_fallback_title(ImageDraw.Draw(img), "Types Throughout My Day", 2, w, h)
            return img

        hourly = graph_data.get("hourly_types", {})
        x_limit = self._type_chart_hours
        hours = list(range(x_limit + 1))

        # Per-type count arrays aligned to hour indices
        counts = {
            t: [hourly.get((h, t), 0) for h in hours]
            for t in _TYPE_ORDER
        }

        # Cumulative stacks (numpy arrays for fill_between)
        cum = [np.zeros(len(hours))]
        for t in _TYPE_ORDER:
            cum.append(cum[-1] + np.array(counts[t], dtype=float))

        fig, ax = self._mpl_base_fig(
            "Types Throughout My Day", 2, w, h,
            bottom=0.22, left=0.14, right=0.97,
        )

        # Solid black base + 3 hatched bands — each clearly distinct on 1-bit display
        fill_styles = [
            dict(facecolor="black", hatch="",  edgecolor="black", linewidth=0.5),
            dict(facecolor="white", hatch="/",  edgecolor="black", linewidth=0.5),
            dict(facecolor="white", hatch="x",  edgecolor="black", linewidth=0.5),
            dict(facecolor="white", hatch=".",  edgecolor="black", linewidth=0.5),
        ]
        for i, (t, style) in enumerate(zip(_TYPE_ORDER, fill_styles)):
            ax.fill_between(hours, cum[i], cum[i + 1],
                            label=t.capitalize(), **style)
            # Explicit boundary line for crisp e-ink rendering
            ax.plot(hours, cum[i + 1], "k-", linewidth=0.4)

        ax.set_xlim(0, x_limit)
        step = max(1, x_limit // 6)
        ax.set_xticks(range(0, x_limit + 1, step))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Hour of day", fontsize=7)
        ax.set_ylabel("# curiosities", fontsize=7)
        ax.legend(
            loc="upper left", fontsize=5, ncol=2,
            frameon=True, edgecolor="black",
            handlelength=1.2, handleheight=0.8,
            borderpad=0.3, labelspacing=0.2,
        )
        return self._mpl_to_pil(fig, w, h)

    def _render_view_topics(self, graph_data: dict, w: int, h: int) -> "Image.Image":
        """
        View 3 — Topics that make me curious: horizontal bar chart (top 5).

        Bars are normalised so the leading topic = 100 %, making the scale
        meaningful even when absolute scores differ by orders of magnitude.
        """
        if not MPL_AVAILABLE:
            img = Image.new("1", (w, h), _BG)
            self._draw_fallback_title(ImageDraw.Draw(img), "Topics that make me curious", 3, w, h)
            return img

        top_topics = graph_data.get("top_topics", [])
        fig, ax = self._mpl_base_fig(
            "Topics that make me curious", 3, w, h,
            left=0.22, bottom=0.18, right=0.95,
        )

        if not top_topics:
            ax.text(0.5, 0.5, "No topics recorded yet",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
        else:
            names  = [c.capitalize() for c, _ in top_topics]
            scores = [s for _, s in top_topics]
            max_s  = max(scores) if scores else 1
            pct    = [s / max_s * 100 for s in scores]

            y_pos = list(range(len(names)))
            bars  = ax.barh(y_pos, pct, color="black", height=0.55, zorder=2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Relative interest (%)", fontsize=7)
            ax.set_xlim(0, 115)
            ax.tick_params(axis="x", labelsize=6)
            # Percentage label at the right end of each bar
            for bar, val in zip(bars, pct):
                ax.text(
                    val + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    va="center", fontsize=6,
                )
        return self._mpl_to_pil(fig, w, h)

    def _draw_fallback_title(
        self, draw: Any, title: str, view_idx: int, w: int, h: int
    ) -> None:
        """Non-matplotlib fallback for graph views when PIL is available but MPL is not."""
        f_med  = _load_font(14)
        f_tiny = _load_font(10)
        draw.text((10, 8), title, font=f_med, fill=_FG)
        draw.line([(10, 32), (w - 20, 32)], fill=_FG, width=2)
        draw.text((10, 60), "Install matplotlib to enable graphs.", font=f_tiny, fill=_GRAY)
        dots = " ".join("●" if i == view_idx else "○" for i in range(_NUM_VIEWS))
        draw.text((w - 50, h - 14), dots, font=f_tiny, fill=_GRAY)

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
