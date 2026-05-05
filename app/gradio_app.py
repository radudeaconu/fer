"""Gradio demo for the FER project.

Run: python -m app.gradio_app

Pipeline per snapshot:
    PIL image -> MediaPipe Face Detection -> largest-bbox crop (+20% pad)
              -> Predictor(crop) -> 7-class probabilities

Models load lazily on first prediction. Checkpoints that don't exist on
disk are filtered out of the dropdowns at startup so a demo doesn't
silently fall back to ImageNet weights and show garbage.

UI tabs:
    1. Image upload    — model picker, top-3 probabilities, detected-face preview.
    2. Webcam snapshot — same, browser-side webcam.
    3. Compare         — runs DAN AND ConvNeXt on the same crop, side-by-side.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from src.data import CLASSES
from src.predict import Predictor

# MediaPipe BlazeFace short-range (selfie distance, ~2m). Float16 tflite.
_MP_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_MP_MODEL_PATH = Path.home() / ".cache" / "fer-app" / "blaze_face_short_range.tflite"

CHECKPOINTS = {
    "DAN (baseline, ResNet-18+attention)":  ("dan",           Path("runs/dan_fer2013/best.pth")),
    "ConvNeXt-Tiny (modern CNN)":           ("convnext_tiny", Path("runs/convnext_fer2013/best.pth")),
    "POSTER++ (RAF-DB only, optional)":     ("poster_pp",     Path("runs/poster_rafdb/best.pth")),
}

DAN_LABEL = "DAN (baseline, ResNet-18+attention)"
CNX_LABEL = "ConvNeXt-Tiny (modern CNN)"

# Lazy caches.
_PREDICTORS: dict[str, Predictor] = {}
_FACE_DETECTOR = None


def _available_checkpoints() -> dict[str, tuple[str, Path]]:
    """Filter CHECKPOINTS to entries whose ckpt path exists on disk."""
    return {label: spec for label, spec in CHECKPOINTS.items() if spec[1].exists()}


def _ensure_mp_model() -> Path:
    if _MP_MODEL_PATH.exists():
        return _MP_MODEL_PATH
    _MP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MP_MODEL_URL, _MP_MODEL_PATH)
    return _MP_MODEL_PATH


def _detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_ensure_mp_model())),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(opts)
    return _FACE_DETECTOR


def _annotate(image: Image.Image, text: str, color: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    pad = 6
    bbox = draw.textbbox((pad, pad), text)
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=color)
    draw.text((pad, pad), text, fill="white")
    return out


def detect_and_crop_face(
    image: Image.Image | None, pad: float = 0.2
) -> tuple[Image.Image | None, Image.Image | None]:
    """Detect the largest face and return (crop, annotated_preview).

    crop is None if no face was detected. preview is always returned when an
    input image is provided — either with a green bbox over the detected face
    or with a 'No face detected' banner.
    """
    if image is None:
        return None, None
    import mediapipe as mp
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = _detector().detect(mp_image)
    if not res.detections:
        return None, _annotate(image, "No face detected", "red")

    best = max(
        res.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height,
    )
    bb = best.bounding_box  # absolute pixel coords
    px = int(round(pad * bb.width))
    py = int(round(pad * bb.height))
    x0 = max(0, bb.origin_x - px)
    y0 = max(0, bb.origin_y - py)
    x1 = min(w, bb.origin_x + bb.width + px)
    y1 = min(h, bb.origin_y + bb.height + py)
    if x1 <= x0 or y1 <= y0:
        return None, _annotate(image, "No face detected", "red")

    crop = image.crop((x0, y0, x1, y1))
    preview = image.copy()
    ImageDraw.Draw(preview).rectangle([x0, y0, x1, y1], outline="lime", width=4)
    return crop, preview


def _get_predictor(choice: str) -> Predictor:
    if choice not in _PREDICTORS:
        model_name, ckpt = CHECKPOINTS[choice]
        _PREDICTORS[choice] = Predictor(model_name=model_name, ckpt=ckpt)
    return _PREDICTORS[choice]


def predict_single(
    image: Image.Image | None, model_choice: str
) -> tuple[dict[str, float], Image.Image | None]:
    crop, preview = detect_and_crop_face(image)
    if crop is None:
        return {c: 0.0 for c in CLASSES}, preview
    probs = _get_predictor(model_choice).predict(crop).probabilities
    return probs, preview


def predict_compare(
    image: Image.Image | None,
) -> tuple[dict[str, float], dict[str, float], Image.Image | None]:
    crop, preview = detect_and_crop_face(image)
    empty = {c: 0.0 for c in CLASSES}
    if crop is None:
        return empty, empty, preview
    dan = _get_predictor(DAN_LABEL).predict(crop).probabilities if DAN_LABEL in CHECKPOINTS else empty
    cnx = _get_predictor(CNX_LABEL).predict(crop).probabilities if CNX_LABEL in CHECKPOINTS else empty
    return dan, cnx, preview


def build_ui() -> gr.Blocks:
    available = _available_checkpoints()
    if not available:
        missing = "\n  ".join(f"{lbl}: {spec[1]}" for lbl, spec in CHECKPOINTS.items())
        raise SystemExit(
            "No checkpoints found on disk. Expected at least one of:\n  " + missing
        )

    choices = list(available)
    can_compare = DAN_LABEL in available and CNX_LABEL in available

    with gr.Blocks(title="Facial Emotion Recognition") as demo:
        gr.Markdown(
            "# Facial Emotion Recognition\n"
            "Models trained on FER-2013 PrivateTest (3,589 images). "
            "Each input goes through MediaPipe face detection first; the "
            "largest detected face (with 20% padding) is what the classifier sees.\n\n"
            "Reference report: `research/facial-emotion-recognition/report.md`."
        )

        with gr.Tab("Image upload"):
            model_choice = gr.Radio(choices=choices, value=choices[0], label="Model")
            with gr.Row():
                img_in = gr.Image(type="pil", label="Input image")
                preview = gr.Image(type="pil", label="Detected face", interactive=False)
                label_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            img_in.change(predict_single, [img_in, model_choice], [label_out, preview])
            model_choice.change(predict_single, [img_in, model_choice], [label_out, preview])

        with gr.Tab("Webcam"):
            webcam_choice = gr.Radio(choices=choices, value=choices[0], label="Model")
            with gr.Row():
                webcam_in = gr.Image(sources=["webcam"], type="pil", label="Webcam snapshot")
                webcam_preview = gr.Image(type="pil", label="Detected face", interactive=False)
                webcam_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            webcam_in.change(
                predict_single, [webcam_in, webcam_choice], [webcam_out, webcam_preview]
            )

        if can_compare:
            with gr.Tab("Compare DAN vs ConvNeXt"):
                gr.Markdown(
                    "Both models score the same detected-face crop. "
                    "Disagreements are exactly where ensembling can recover errors."
                )
                with gr.Row():
                    cmp_in = gr.Image(type="pil", label="Input image")
                    cmp_preview = gr.Image(type="pil", label="Detected face", interactive=False)
                with gr.Row():
                    dan_out = gr.Label(num_top_classes=3, label="DAN")
                    cnx_out = gr.Label(num_top_classes=3, label="ConvNeXt-Tiny")
                cmp_in.change(predict_compare, cmp_in, [dan_out, cnx_out, cmp_preview])

        gr.Markdown(
            f"Class taxonomy (matches FER-2013): {', '.join(CLASSES)}.\n\n"
            f"Loaded checkpoints: {', '.join(choices)}."
        )
    return demo


def main() -> None:
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
