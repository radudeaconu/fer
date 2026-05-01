"""Model wrappers for DAN and POSTER++.

Both wrappers expect the corresponding reference repository to be cloned into
`third_party/`:

    third_party/DAN/        from https://github.com/yaoing/DAN     (report.md:272)
    third_party/POSTER_V2/  from https://github.com/Talented-Q/POSTER_V2 (report.md:926)

Released checkpoints must be downloaded separately from each repo's README
(usually a Google Drive link) and placed under `checkpoints/`.

These wrappers do NOT re-implement the architectures — they import the
reference code and adapt it to the project's 7-class label space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.data import NUM_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"


def _ensure_third_party_on_path(name: str) -> Path:
    """Add `third_party/<name>` to sys.path so its modules import as top-level."""
    repo_dir = THIRD_PARTY / name
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Reference repository missing: {repo_dir}. "
            f"Clone it via the Colab notebook or run "
            f"`git clone <url> {repo_dir}` (see CLAUDE.md for URLs)."
        )
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return repo_dir


def build_dan(num_classes: int = NUM_CLASSES, pretrained_ckpt: str | Path | None = None) -> nn.Module:
    """Construct DAN (ResNet-18 backbone with multi-head attention) — report.md:252.

    Args:
        num_classes: target classes for the final FC. RAF-DB and FER-2013 both use 7.
        pretrained_ckpt: path to the released RAF-DB checkpoint (DAN.pth). If None,
            backbone weights are still ImageNet-init via the reference repo's loader.

    The reference impl's class is at `networks.DAN` in yaoing/DAN.
    """
    _ensure_third_party_on_path("DAN")
    from networks.DAN import DAN  # type: ignore

    model = DAN(num_class=num_classes, num_head=4, pretrained=True)

    if pretrained_ckpt is not None:
        ckpt_path = Path(pretrained_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"DAN checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Reference repo saves under 'model_state_dict' for the released ckpt.
        sd = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[build_dan] unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        if missing:
            print(f"[build_dan] missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    return model


def build_poster_pp(
    num_classes: int = NUM_CLASSES,
    pretrained_ckpt: str | Path | None = None,
) -> nn.Module:
    """Construct POSTER++ (IR-50 + MobileFaceNet + Pyramid Cross-Fusion Transformer) — report.md:905.

    Args:
        num_classes: 7 for RAF-DB.
        pretrained_ckpt: path to the released RAF-DB checkpoint (rafdb-best.pth or similar).

    POSTER++ requires the IR-50 and MobileFaceNet backbones to be pretrained on
    face recognition (report.md:933) — never train from random init. The reference
    repo's loader pulls these from a `pretrain/` directory inside third_party/POSTER_V2.
    """
    _ensure_third_party_on_path("POSTER_V2")
    from models.PosterV2_7cls import pyramid_trans_expr2  # type: ignore

    model = pyramid_trans_expr2(img_size=224, num_classes=num_classes)

    if pretrained_ckpt is not None:
        ckpt_path = Path(pretrained_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"POSTER++ checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state.get("model_state_dict", state))
        # Strip 'module.' DDP prefix if present.
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[build_poster_pp] unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        if missing:
            print(f"[build_poster_pp] missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    return model


MODEL_REGISTRY = {
    "dan": build_dan,
    "poster_pp": build_poster_pp,
}


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained_ckpt: str | Path | None = None) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained_ckpt=pretrained_ckpt)
