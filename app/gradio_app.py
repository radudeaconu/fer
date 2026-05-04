"""Gradio demo for the FER project.

Run: python -m app.gradio_app

Loads DAN, ConvNeXt-Tiny, and (if RAF-DB ckpts are available) POSTER++ lazily.
The first prediction with each model triggers the load; falls back to
ImageNet/random init if no checkpoint is found, with a warning printed.

UI tabs:
    1. Image upload — single model picker, top-3 probabilities.
    2. Webcam snapshot — same, browser-side webcam.
    3. Compare — runs DAN AND ConvNeXt on the same image, side-by-side bars.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from src.data import CLASSES
from src.predict import Predictor

CHECKPOINTS = {
    "DAN (baseline, ResNet-18+attention)":  ("dan",           Path("runs/dan_fer2013/best.pth")),
    "ConvNeXt-Tiny (modern CNN)":           ("convnext_tiny", Path("runs/convnext_fer2013/best.pth")),
    "POSTER++ (RAF-DB only, optional)":     ("poster_pp",     Path("runs/poster_rafdb/best.pth")),
}

# Lazy cache so each model is loaded at most once per process.
_PREDICTORS: dict[str, Predictor] = {}


def _get_predictor(choice: str) -> Predictor:
    if choice not in _PREDICTORS:
        model_name, ckpt = CHECKPOINTS[choice]
        _PREDICTORS[choice] = Predictor(model_name=model_name, ckpt=ckpt)
    return _PREDICTORS[choice]


def predict_single(image: Image.Image | None, model_choice: str) -> dict[str, float]:
    if image is None:
        return {c: 0.0 for c in CLASSES}
    return _get_predictor(model_choice).predict(image).probabilities


def predict_compare(
    image: Image.Image | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run DAN and ConvNeXt on the same image; return both probability dicts."""
    if image is None:
        empty = {c: 0.0 for c in CLASSES}
        return empty, empty
    dan_choice = "DAN (baseline, ResNet-18+attention)"
    cnx_choice = "ConvNeXt-Tiny (modern CNN)"
    dan_probs = _get_predictor(dan_choice).predict(image).probabilities
    cnx_probs = _get_predictor(cnx_choice).predict(image).probabilities
    return dan_probs, cnx_probs


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Facial Emotion Recognition") as demo:
        gr.Markdown(
            "# Facial Emotion Recognition\n"
            "Upload a face image (or take a webcam snapshot) to get a 7-class probability distribution.\n"
            "Models trained on FER-2013 PrivateTest (3,589 images). "
            "Reference report: `research/facial-emotion-recognition/report.md`."
        )

        with gr.Tab("Image upload"):
            with gr.Row():
                model_choice = gr.Radio(
                    choices=list(CHECKPOINTS),
                    value=list(CHECKPOINTS)[0],
                    label="Model",
                )
            with gr.Row():
                img_in = gr.Image(type="pil", label="Input face")
                label_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            img_in.change(fn=predict_single, inputs=[img_in, model_choice], outputs=label_out)
            model_choice.change(fn=predict_single, inputs=[img_in, model_choice], outputs=label_out)

        with gr.Tab("Webcam"):
            with gr.Row():
                webcam_choice = gr.Radio(
                    choices=list(CHECKPOINTS),
                    value=list(CHECKPOINTS)[0],
                    label="Model",
                )
            with gr.Row():
                webcam_in = gr.Image(sources=["webcam"], type="pil", label="Webcam snapshot")
                webcam_out = gr.Label(num_top_classes=3, label="Top-3 emotion")
            webcam_in.change(fn=predict_single, inputs=[webcam_in, webcam_choice], outputs=webcam_out)

        with gr.Tab("Compare DAN vs ConvNeXt"):
            gr.Markdown(
                "Runs both models on the same image. Use this to spot disagreements — "
                "those are the cases where ensembling could recover errors."
            )
            with gr.Row():
                cmp_in = gr.Image(type="pil", label="Input face")
            with gr.Row():
                dan_out = gr.Label(num_top_classes=3, label="DAN")
                cnx_out = gr.Label(num_top_classes=3, label="ConvNeXt-Tiny")
            cmp_in.change(fn=predict_compare, inputs=cmp_in, outputs=[dan_out, cnx_out])

        gr.Markdown(
            "Class taxonomy (matches FER-2013): "
            f"{', '.join(CLASSES)}."
        )
    return demo


def main() -> None:
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
