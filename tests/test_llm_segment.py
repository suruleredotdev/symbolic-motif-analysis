"""Tests for Claude Suggest (panel_art.llm_segment) and the draft queue it feeds.

No network: a stub client records the request and replays a canned reply.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from panel_art import llm_segment as llm
from panel_art import motif_segment as ms
from panel_art.pipeline_state import PipelineState


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class StubClient:
    def __init__(self, reply: dict | None = None, stop_reason: str = "end_turn",
                 category: str | None = None):
        self.requests: list[dict] = []
        text = json.dumps(reply) if reply is not None else ""
        self._message = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=text)],
            stop_reason=stop_reason,
            stop_details=SimpleNamespace(category=category) if category else None,
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
            model="claude-opus-5-5",
        )
        self.messages = self

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        return _Stream(self._message)


CANDS = [
    {"x": 10, "y": 10, "w": 60, "h": 120},     # 0 — a figure
    {"x": 100, "y": 20, "w": 150, "h": 150},   # 1 — a clump
    {"x": 300, "y": 40, "w": 50, "h": 90},     # 2 — a figure, loose box
]

REPLY = {
    "reviews": [
        {"candidate": 0, "verdict": "accept", "bbox": {"x": 11, "y": 9, "w": 61, "h": 119},
         "label": "standing figure", "reason": "single figure"},
        {"candidate": 1, "verdict": "reject", "bbox": CANDS[1],
         "label": "", "reason": "two figures in one box"},
        {"candidate": 2, "verdict": "adjust", "bbox": {"x": 305, "y": 45, "w": 40, "h": 80},
         "label": "bird", "reason": "tightened"},
        {"candidate": 7, "verdict": "accept", "bbox": CANDS[0],       # no such candidate
         "label": "x", "reason": "x"},
    ],
    "additions": [
        {"bbox": {"x": 120, "y": 30, "w": 60, "h": 120}, "label": "left figure of clump",
         "confidence": "medium", "reason": "split clump"},
        {"bbox": {"x": 180, "y": 30, "w": 60, "h": 120}, "label": "right figure of clump",
         "confidence": "high", "reason": "split clump"},
        {"bbox": {"x": 390, "y": 890, "w": 40, "h": 40}, "label": "off the edge",
         "confidence": "low", "reason": "clamped to nothing"},
    ],
}


def _panel(w: int = 400, h: int = 300) -> np.ndarray:
    return np.full((h, w, 3), 180, dtype=np.uint8)


# ── suggest_boxes ────────────────────────────────────────────────────────────

def test_request_shape():
    client = StubClient(REPLY)
    llm.suggest_boxes(_panel(), [{"x": 0, "y": 200, "w": 50, "h": 50}], CANDS,
                      model="claude-opus-5-5", effort="high", client=client)
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"
    images = [b for b in req["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 2                      # grid view + boxes view
    text = req["messages"][0]["content"][-1]["text"]
    assert '"candidate": 2' in text and "E0" in text


def test_queue_order_and_verdicts():
    result = llm.suggest_boxes(_panel(), [], CANDS, client=StubClient(REPLY))
    s = result.suggestions
    # Reviewed SAM boxes first in SAM order, then additions by confidence.
    assert [x.origin for x in s] == ["sam_accept", "sam_adjust", "llm_add", "llm_add"]
    assert s[0].bbox == CANDS[0]                 # accept keeps the exact SAM box
    assert s[1].bbox == {"x": 305, "y": 45, "w": 40, "h": 80}
    assert [x.confidence for x in s[2:]] == ["high", "medium"]
    assert result.rejected == [{"candidate": 1, "reason": "two figures in one box"}]
    assert s[0].source == "sam_llm" and s[2].source == "llm"
    assert result.usage == {"input_tokens": 1200, "output_tokens": 300}


def test_large_panels_are_downscaled_and_mapped_back():
    w, h = 3136, 1000                            # 2× MAX_IMAGE_EDGE on the long side
    reply = {"reviews": [], "additions": [
        {"bbox": {"x": 100, "y": 50, "w": 30, "h": 40}, "label": "f",
         "confidence": "high", "reason": "r"}]}
    client = StubClient(reply)
    result = llm.suggest_boxes(_panel(w, h), [], [], client=client)
    img_block = next(b for b in client.requests[0]["messages"][0]["content"] if b["type"] == "image")
    import base64, io
    sent = Image.open(io.BytesIO(base64.b64decode(img_block["source"]["data"])))
    assert max(sent.size) == llm.MAX_IMAGE_EDGE
    assert result.suggestions[0].bbox == {"x": 200, "y": 100, "w": 60, "h": 80}


def test_refusal_raises():
    with pytest.raises(llm.LLMSegmentError, match="declined"):
        llm.suggest_boxes(_panel(), [], CANDS,
                          client=StubClient(None, stop_reason="refusal", category="bio"))


def test_truncation_raises():
    with pytest.raises(llm.LLMSegmentError, match="max_tokens"):
        llm.suggest_boxes(_panel(), [], CANDS, client=StubClient(None, stop_reason="max_tokens"))


def test_grid_step_scales_with_panel():
    assert llm.grid_step(400, 300) == 25
    assert llm.grid_step(959, 618) == 50
    assert llm.grid_step(3000, 2000) == 200


# ── PipelineState queue + decision log ───────────────────────────────────────

@pytest.fixture
def state(analysis_dir: Path) -> PipelineState:
    ps = PipelineState()
    ps.load_from_disk(
        annotated_dir=analysis_dir / "annotated",
        panels_dir=analysis_dir / "panels",
        labels_path=analysis_dir / "motif_labels.json",
    )
    return ps


def test_llm_suggestions_flow_through_the_draft_queue(state: PipelineState, analysis_dir: Path):
    result = llm.suggest_boxes(_panel(), [], CANDS, client=StubClient(REPLY))
    n = state.cache_llm_suggestions("panel_a", result.suggestions)
    assert n == 4
    before = len(state.motifs_for_panel("panel_a"))

    rec = state.accept_draft("panel_a")                      # SAM #0, accepted as-is
    assert rec.source == "sam_llm" and rec.notes == "llm: standing figure"
    state.skip_draft("panel_a")                              # SAM #2 adjusted
    tightened = {"x": 122, "y": 32, "w": 55, "h": 110}
    rec = state.accept_draft("panel_a", adjusted_bbox=tightened)
    assert rec.source == "llm" and rec.bbox == tightened
    assert len(state.motifs_for_panel("panel_a")) == before + 2

    state.save_approved("panel_a")
    log = [json.loads(line) for line in
           (analysis_dir / "annotated" / "draft_log.jsonl").read_text().splitlines()]
    assert [(e["origin"], e["action"]) for e in log] == [
        ("sam_accept", "accept"), ("sam_adjust", "skip"), ("llm_add", "accept")]
    assert log[0]["iou"] == 1.0
    assert log[2]["iou"] < 1.0 and log[2]["label"] == "right figure of clump"
    # Flushed entries are not written twice.
    state.save_approved("panel_a")
    assert len((analysis_dir / "annotated" / "draft_log.jsonl").read_text().splitlines()) == 3


def test_pending_drafts_tracks_the_cursor(state: PipelineState):
    result = llm.suggest_boxes(_panel(), [], CANDS, client=StubClient(REPLY))
    state.cache_llm_suggestions("panel_b", result.suggestions)
    state.skip_draft("panel_b")
    assert len(state.pending_drafts("panel_b")) == 3


# ── SAM prompting helpers (no model needed) ──────────────────────────────────

def test_carving_peaks_skip_flat_wood_and_existing_boxes():
    img = np.full((200, 300), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    img[40:100, 30:90] = rng.integers(0, 255, (60, 60))      # carved patch A
    img[40:100, 200:260] = rng.integers(0, 255, (60, 60))    # carved patch B
    peaks = ms.carving_peaks(img, spacing=40, min_density=0.12)
    xs = sorted(x for x, _, _ in peaks)
    assert xs and all(20 <= x <= 100 or 190 <= x <= 270 for x in xs)

    occupied = np.zeros_like(img, dtype=bool)
    occupied[:, 150:] = True
    assert all(x < 150 for x, _, _ in ms.carving_peaks(img, 40, 0.12, occupied))


def test_rank_prefers_carved_motif_sized_boxes():
    median = 90 * 160
    motif = ms.rank_score(0.30, 90 * 160, median, 1.0)
    blank = ms.rank_score(0.04, 90 * 160, median, 1.0)
    clump = ms.rank_score(0.30, 3 * 90 * 160, median, 1.0)
    assert motif > clump > blank


def test_trim_outsized_drops_the_odd_large_form():
    boxes = [{"w": 75, "h": 190}, {"w": 134, "h": 111}, {"w": 74, "h": 168},
             {"w": 138, "h": 331}]
    assert {"w": 138, "h": 331} not in ms._trim_outsized(boxes)
