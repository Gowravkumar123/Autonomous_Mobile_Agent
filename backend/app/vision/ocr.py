"""
OCR fallback via RapidOCR (ONNX, on-device/on-server, no API). Reads the visible text
and its position off a screenshot, for screens where the accessibility tree is empty.

Positions are returned as normalized (0..1) screen fractions, matching the `tap_xy`
action — so the agent can both READ the text and TAP it. Degrades gracefully to an
empty list if RapidOCR isn't installed, so the rest of the system still runs.
"""
from __future__ import annotations

import base64
import io
import logging
from functools import lru_cache

log = logging.getLogger("agent.ocr")


@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
        log.info("RapidOCR engine loaded")
        return RapidOCR()
    except Exception as e:  # not installed, or model load failed
        log.warning("RapidOCR unavailable (%s) — OCR disabled", e)
        return None


def available() -> bool:
    return _engine() is not None


def ocr_items(image_b64: str, max_items: int = 40, min_conf: float = 0.4) -> list[dict]:
    """Return [{text, x, y, conf}] with x/y normalized 0..1 (box centre), top-to-bottom."""
    engine = _engine()
    if engine is None or not image_b64:
        return []
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        w, h = img.size
        if w == 0 or h == 0:
            return []
        result, _ = engine(np.asarray(img))
        items: list[dict] = []
        for box, text, conf in (result or []):
            c = float(conf)
            if c < min_conf or not str(text).strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append({
                "text": str(text).strip()[:60],
                "x": round(sum(xs) / len(xs) / w, 3),
                "y": round(sum(ys) / len(ys) / h, 3),
                "conf": round(c, 2),
            })
        items.sort(key=lambda it: (it["y"], it["x"]))
        return items[:max_items]
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return []
