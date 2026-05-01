"""Single-image inference helper used by the Gradio app and CLI demos.

Run: python -m src.predict --model dan --ckpt runs/dan_rafdb/best.pth --image path/to/face.jpg
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.data import CLASSES, build_transforms
from src.models import build_model


@dataclass
class Prediction:
    label: str
    label_idx: int
    probabilities: dict[str, float]  # ordered by descending prob

    def topk(self, k: int = 3) -> list[tuple[str, float]]:
        return list(self.probabilities.items())[:k]


class Predictor:
    """Lazy-loading wrapper around a trained FER model.

    First call to `predict()` moves the model to GPU if available.
    """

    def __init__(self, model_name: str, ckpt: str | Path, image_size: int = 224) -> None:
        self.model_name = model_name
        self.ckpt = Path(ckpt)
        self.image_size = image_size
        self.transform = build_transforms(train=False, image_size=image_size)
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    def _load(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(self.model_name)
        if self.ckpt.exists():
            state = torch.load(self.ckpt, map_location=device, weights_only=False)
            sd = state.get("model", state.get("model_state_dict", state))
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        else:
            print(f"[Predictor] Warning: checkpoint {self.ckpt} not found — using ImageNet init.")
        model.to(device).eval()
        self._model = model
        self._device = device

    def predict(self, image: Image.Image | str | Path) -> Prediction:
        if self._model is None:
            self._load()
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        ranked = sorted(zip(CLASSES, probs), key=lambda kv: -kv[1])
        return Prediction(
            label=ranked[0][0],
            label_idx=CLASSES.index(ranked[0][0]),
            probabilities={k: round(v, 4) for k, v in ranked},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dan", "poster_pp"])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    predictor = Predictor(args.model, args.ckpt)
    result = predictor.predict(args.image)
    print(f"top-1: {result.label}")
    for cls, p in result.topk(3):
        print(f"  {cls:<10s} {p:.4f}")


if __name__ == "__main__":
    main()
