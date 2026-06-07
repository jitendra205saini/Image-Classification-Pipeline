#!/usr/bin/env python3
"""
Advanced command-line image classifier.

Usage
-----
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --top_k 10
    python predict.py path/to/image.jpg --model efficientnet_b0

Prints a clean, formatted ASCII table of the top-k predictions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def load_model(model_name: str) -> Tuple[torch.nn.Module, transforms.Compose, List[str]]:
    """Load a pretrained model with its canonical preprocessing and labels."""
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: mobilenet_v3_large, efficientnet_b0"
        )

    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


def classify(
    image_path: Path, model_name: str, top_k: int
) -> Tuple[List[Tuple[str, float]], float]:
    """Return (list of (label, confidence), inference_ms)."""
    model, preprocess, categories = load_model(model_name)

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    top_k = max(1, min(top_k, probs.numel()))
    confidences, indices = torch.topk(probs, top_k)
    predictions = [
        (categories[idx], float(conf))
        for conf, idx in zip(confidences.tolist(), indices.tolist())
    ]
    return predictions, inference_ms


def render_table(predictions: List[Tuple[str, float]]) -> str:
    """Render predictions as a bordered ASCII table with confidence bars."""
    rank_w, label_w, conf_w, bar_w = 4, 28, 10, 22

    def row(rank: str, label: str, conf: str, bar: str) -> str:
        return (
            f"| {rank:<{rank_w}} | {label:<{label_w}} | "
            f"{conf:<{conf_w}} | {bar:<{bar_w}} |"
        )

    border = (
        "+" + "-" * (rank_w + 2) + "+" + "-" * (label_w + 2)
        + "+" + "-" * (conf_w + 2) + "+" + "-" * (bar_w + 2) + "+"
    )

    lines = [border, row("#", "Class", "Conf.", "Confidence"), border]
    for i, (label, conf) in enumerate(predictions, start=1):
        pct = conf * 100
        filled = int(round((bar_w) * conf))
        bar = "#" * filled + "." * (bar_w - filled)
        lines.append(row(str(i), label[:label_w], f"{pct:6.2f}%", bar))
    lines.append(border)
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify an image from the command line using a pretrained CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top predictions to display."
    )
    parser.add_argument(
        "--model",
        default="mobilenet_v3_large",
        choices=["mobilenet_v3_large", "efficientnet_b0"],
        help="Model architecture to use.",
    )
    args = parser.parse_args(argv)

    if not args.image.exists():
        print(f"Error: file not found -> {args.image}", file=sys.stderr)
        return 1

    print(f"\n  Model : {args.model}")
    print(f"  Image : {args.image}\n")

    try:
        predictions, inference_ms = classify(args.image, args.model, args.top_k)
    except Exception as exc:  # noqa: BLE001
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 1

    print(render_table(predictions))
    print(f"\n  Top prediction : {predictions[0][0]} ({predictions[0][1] * 100:.2f}%)")
    print(f"  Inference time : {inference_ms:.1f} ms\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
