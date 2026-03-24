"""CLIP model wrapper for generating image and text embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image as PILImage

_processor = None
_model = None


def _load_model() -> tuple:
    """Lazily load the CLIP model and processor."""
    global _processor, _model  # noqa: PLW0603

    if _model is None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        _processor = CLIPProcessor.from_pretrained(model_name)
        _model = CLIPModel.from_pretrained(model_name)
        _model.eval()

    return _processor, _model


def embed_image(image: "PILImage.Image") -> list[float]:
    """Generate a normalised CLIP embedding for a PIL image.

    Args:
        image: A PIL ``Image`` object (will be converted to RGB if needed).

    Returns:
        A list of floats representing the embedding vector.
    """
    import torch

    processor, model = _load_model()

    rgb_image = image.convert("RGB")
    inputs = processor(images=rgb_image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features[0].numpy()
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()


def embed_text(text: str) -> list[float]:
    """Generate a normalised CLIP embedding for a text string.

    Args:
        text: The search query string.

    Returns:
        A list of floats representing the embedding vector.
    """
    import torch

    processor, model = _load_model()

    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    vector = features[0].numpy()
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()


def vector_size() -> int:
    """Return the dimensionality of the CLIP embedding vectors."""
    return 512
