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


def build_dan(
    num_classes: int = NUM_CLASSES,
    pretrained_ckpt: str | Path | None = None,
    use_imagenet_init: bool = True,
) -> nn.Module:
    """Construct DAN (ResNet-18 backbone with multi-head attention) — report.md:252.

    Args:
        num_classes: target classes for the final FC. RAF-DB and FER-2013 both use 7.
        pretrained_ckpt: path to a trained DAN checkpoint (e.g., released RAF-DB ckpt).
            Takes precedence over `use_imagenet_init`.
        use_imagenet_init: when True (default), load torchvision ImageNet ResNet-18 weights
            into the backbone. Set False for the random-init ablation that quantifies
            the contribution of transfer learning.

    Note: DAN's __init__ with pretrained=True hard-codes a load of
    `./models/resnet18_msceleb.pth` (MS-Celeb-1M backbone). We don't ship that file;
    instead we construct DAN with pretrained=False and post-hoc load torchvision's
    ImageNet ResNet-18 into `model.features`. ImageNet init is empirically only a
    few points worse than MS-Celeb-1M init for FER fine-tuning.
    """
    _ensure_third_party_on_path("DAN")
    from networks.dan import DAN  # type: ignore

    model = DAN(num_class=num_classes, num_head=4, pretrained=False)

    if pretrained_ckpt is None:
        if not use_imagenet_init:
            return model  # leave the random init from DAN's __init__
        from torchvision import models as tv_models
        from torchvision.models import ResNet18_Weights
        imagenet_rn18 = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # DAN trims to children()[:-2] (drops avgpool + fc); mirror that for the load.
        imagenet_features = nn.Sequential(*list(imagenet_rn18.children())[:-2])
        missing, unexpected = model.features.load_state_dict(imagenet_features.state_dict(), strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        return model

    ckpt_path = Path(pretrained_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DAN checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
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


def build_convnext_tiny(
    num_classes: int = NUM_CLASSES,
    pretrained_ckpt: str | Path | None = None,
) -> nn.Module:
    """Construct ConvNeXt-Tiny with ImageNet init. Backbone of EmoNeXt — report.md:488.

    Modern CNN baseline (Liu et al., 2022) intended as the "stretch" comparison vs DAN's
    ResNet-18 + multi-head attention. ImageNet weights are pulled from torchvision; the
    final 1000-way classifier is replaced with a 7-way Linear over the same 768-D feature.

    Args:
        num_classes: target classes (7 for FER-2013 / RAF-DB).
        pretrained_ckpt: optional path to a fine-tuned checkpoint to resume from.
    """
    from torchvision import models as tv_models
    from torchvision.models import ConvNeXt_Tiny_Weights

    model = tv_models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # ConvNeXt's classifier is Sequential(LayerNorm2d, Flatten, Linear(768, 1000)).
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    if pretrained_ckpt is not None:
        ckpt_path = Path(pretrained_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ConvNeXt checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("model", state.get("model_state_dict", state.get("state_dict", state)))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[build_convnext_tiny] unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        if missing:
            print(f"[build_convnext_tiny] missing keys: {missing[:3]}{'...' if len(missing) > 3 else ''}")

    return model


MODEL_REGISTRY = {
    "dan": build_dan,
    "poster_pp": build_poster_pp,
    "convnext_tiny": build_convnext_tiny,
}


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained_ckpt: str | Path | None = None,
    **kwargs,
) -> nn.Module:
    """Dispatch to a model constructor.

    Extra kwargs are forwarded to the constructor — e.g. `use_imagenet_init=False`
    for the DAN no-imagenet ablation. Constructors silently ignore unknown kwargs
    via the per-builder signatures (callers are expected to pass only the
    constructor's documented kwargs).
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained_ckpt=pretrained_ckpt, **kwargs)
