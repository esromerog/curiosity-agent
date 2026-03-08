"""
Object / scene recognition engine.

Three backends are supported:
  - "claude"  : Claude vision API (highest quality, requires API key + internet)
  - "yolo"    : YOLOv8 local model (fast, offline)
  - "clip"    : OpenCLIP zero-shot classification (offline, broad vocabulary)

The result is always a SceneDescription: a list of detected objects / concepts
plus a short natural-language scene summary.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from loguru import logger


@dataclass
class SceneDescription:
    objects: list[str]          # e.g. ["coffee mug", "laptop", "notebook"]
    scene_summary: str          # e.g. "a person working at a desk in a home office"
    raw_labels: list[str] = field(default_factory=list)  # backend-specific raw labels
    backend: str = "unknown"


# ---------------------------------------------------------------------------
# Claude Vision backend
# ---------------------------------------------------------------------------

_CLAUDE_SCENE_PROMPT = """
You are a scene-description assistant. Analyze the image and respond with JSON only.
Format:
{
  "objects": ["list", "of", "notable", "objects"],
  "scene_summary": "one-sentence natural description of the scene"
}
Be concise. Focus on objects that could inspire intellectual curiosity.
""".strip()


async def _describe_with_claude(
    client: anthropic.AsyncAnthropic, model: str, jpeg_bytes: bytes
) -> SceneDescription:
    b64 = base64.standard_b64encode(jpeg_bytes).decode()
    msg = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _CLAUDE_SCENE_PROMPT},
                ],
            }
        ],
    )
    text = msg.content[0].text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    return SceneDescription(
        objects=data.get("objects", []),
        scene_summary=data.get("scene_summary", ""),
        backend="claude",
    )


# ---------------------------------------------------------------------------
# YOLOv8 backend
# ---------------------------------------------------------------------------

def _describe_with_yolo(model_path: str, jpeg_bytes: bytes, threshold: float) -> SceneDescription:
    import io
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(model_path)
    img = Image.open(io.BytesIO(jpeg_bytes))
    results = model(img, conf=threshold, verbose=False)

    labels: list[str] = []
    for result in results:
        for cls_idx in result.boxes.cls.tolist():
            label = result.names[int(cls_idx)]
            if label not in labels:
                labels.append(label)

    summary = "A scene containing: " + ", ".join(labels) if labels else "An unidentified scene."
    return SceneDescription(
        objects=labels,
        scene_summary=summary,
        raw_labels=labels,
        backend="yolo",
    )


# ---------------------------------------------------------------------------
# CLIP backend (zero-shot)
# ---------------------------------------------------------------------------

_CLIP_CANDIDATE_LABELS = [
    "kitchen", "office", "bedroom", "living room", "outdoor park", "street",
    "library", "laboratory", "workshop", "gym", "classroom", "café",
    "laptop", "book", "coffee mug", "plant", "clock", "artwork", "musical instrument",
    "sports equipment", "food", "cooking", "technology", "nature",
]


def _describe_with_clip(jpeg_bytes: bytes, threshold: float) -> SceneDescription:
    import io
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    img = preprocess(Image.open(io.BytesIO(jpeg_bytes))).unsqueeze(0)
    text_tokens = tokenizer(_CLIP_CANDIDATE_LABELS)

    with torch.no_grad():
        img_feat = model.encode_image(img)
        txt_feat = model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0]

    matched = [
        _CLIP_CANDIDATE_LABELS[i]
        for i, p in enumerate(probs.tolist())
        if p > threshold
    ]
    summary = "A scene featuring: " + ", ".join(matched) if matched else "An unidentified scene."
    return SceneDescription(
        objects=matched,
        scene_summary=summary,
        raw_labels=matched,
        backend="clip",
    )


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------

class RecognitionEngine:
    def __init__(
        self,
        backend: Literal["claude", "yolo", "clip"],
        claude_client: anthropic.AsyncAnthropic | None = None,
        claude_model: str = "claude-opus-4-6",
        yolo_model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
    ) -> None:
        self._backend = backend
        self._claude_client = claude_client
        self._claude_model = claude_model
        self._yolo_model_path = yolo_model_path
        self._confidence_threshold = confidence_threshold

    async def describe(self, jpeg_bytes: bytes) -> SceneDescription:
        try:
            if self._backend == "claude":
                if not self._claude_client:
                    raise RuntimeError("Claude client not provided")
                return await _describe_with_claude(
                    self._claude_client, self._claude_model, jpeg_bytes
                )
            elif self._backend == "yolo":
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    _describe_with_yolo,
                    self._yolo_model_path,
                    jpeg_bytes,
                    self._confidence_threshold,
                )
            elif self._backend == "clip":
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, _describe_with_clip, jpeg_bytes, self._confidence_threshold
                )
            else:
                raise ValueError(f"Unknown backend: {self._backend}")
        except Exception as exc:
            logger.error(f"Recognition failed ({self._backend}): {exc}")
            return SceneDescription(
                objects=[],
                scene_summary="Unable to analyze the scene.",
                backend=self._backend,
            )
