"""Gradio demo for the FER project.

Run: python -m app.gradio_app

Loads DAN and/or POSTER++ checkpoints lazily — the first prediction with each
model triggers the load. Falls back to ImageNet-init if no checkpoint is found,
with a warning printed to stdout.

The UI has two tabs:
    1. Image upload — drag/drop a face image; outputs top-3 probabilities.
    2. Webcam snapshot — take a single frame from the webcam (browser-side).

A model selector switches between DAN and POSTER++.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from src.data import CLASSES
from src.predict import Predictor

CHECKPOINTS = {
    "DAN (baseline, 89.7% RAF-DB)": ("dan", Path("runs/dan_rafdb/best.pth")),
    "POSTER++ (stretch, 92.2% RAF-DB)": ("poster_pp", Path("runs/poster_rafdb/best.pth")),
}

# Lazy cache so each model is loaded at most once per process.
_PREDICTORS: dict[str, Predictor] = {}


def _get_predictor(choice: str) -> Predictor:
    if choice not in _PREDICTORS:
        model_name, ckpt = CHECKPOINTS[choice]
        _PREDICTORS[choice] = Predictor(model_name=model_name, ckpt=ckpt)
    return _PREDICTORS[choice]


def predict_fn(image: Image.Image | None, model_choice: str) -> dict[str, float]:
    if image is None:
        return {c: 0.0 for c in CLASSES}
    predictor = _get_predictor(model_choice)
    result = predictor.predict(image)
    return result.probabilities


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Facial Emotion Recognition") as demo:
        gr.Markdown(
            "# Facial Emotion Recognition\n"
            "Upload a face image (or take a webcam snapshot) to get a 7-class probability distribution.\n"
            "Models trained on RAF-DB (29,672 in-the-wild images, basic-7 taxonomy)."
        )
        with gr.Row():
            model_choice = gr.Radio(
                choices=list(CHECKPOINTS),
                value=list(CHECKPOINTS)[0],
                label="Model",
            )

        with gr.Tab("Image upload"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Input face")
                label_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            img_in.change(fn=predict_fn, inputs=[img_in, model_choice], outputs=label_out)

        with gr.Tab("Webcam"):
            with gr.Row():
                webcam_in = gr.Image(sources=["webcam"], type="pil", label="Webcam snapshot")
                webcam_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            webcam_in.change(fn=predict_fn, inputs=[webcam_in, model_choice], outputs=webcam_out)

        gr.Markdown(
            "Class taxonomy (matches FER-2013 / unified RAF-DB): "
            f"{', '.join(CLASSES)}.\n\n"
            "Project source: see `CLAUDE.md`. Reference report: "
            "`research/facial-emotion-recognition/report.md`."
        )
    return demo


def main() -> None:
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
