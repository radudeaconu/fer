"""Smoke tests for the Gradio app helpers.

We don't exercise model inference here — that depends on heavy checkpoints.
Goal: confirm the new face-detect + ckpt-filtering helpers behave on edge
cases (no face, all checkpoints missing, partial checkpoints present).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

mediapipe = pytest.importorskip("mediapipe")  # skip cleanly if dep missing
gradio = pytest.importorskip("gradio")

from app import gradio_app


def test_detect_no_face_returns_none_with_preview():
    blank = Image.new("RGB", (320, 240), color=(0, 0, 0))
    crop, preview = gradio_app.detect_and_crop_face(blank)
    assert crop is None
    assert preview is not None and preview.size == blank.size


def test_detect_handles_none_input():
    crop, preview = gradio_app.detect_and_crop_face(None)
    assert crop is None and preview is None


def test_available_checkpoints_excludes_missing(tmp_path, monkeypatch):
    real = tmp_path / "real.pth"
    real.write_bytes(b"\x00")
    fake = {
        "real": ("dan", real),
        "missing": ("convnext_tiny", tmp_path / "nope.pth"),
    }
    monkeypatch.setattr(gradio_app, "CHECKPOINTS", fake)
    out = gradio_app._available_checkpoints()
    assert "real" in out and "missing" not in out


def test_build_ui_raises_when_all_checkpoints_missing(tmp_path, monkeypatch):
    fake = {"missing": ("dan", tmp_path / "nope.pth")}
    monkeypatch.setattr(gradio_app, "CHECKPOINTS", fake)
    with pytest.raises(SystemExit):
        gradio_app.build_ui()
