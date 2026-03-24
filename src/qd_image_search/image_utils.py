"""Utilities for discovering and loading image files."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterator

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def iter_image_paths(path: Path) -> Iterator[Path]:
    """Yield image file paths from a file or directory.

    Args:
        path: A ``Path`` to a single image file or a directory.  When a
              directory is provided every file with a supported extension is
              yielded (non-recursive).

    Raises:
        ValueError: If ``path`` does not exist or is not a file/directory.
    """
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        else:
            raise ValueError(f"Unsupported image format: {path.suffix!r}")
    elif path.is_dir():
        found = False
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                found = True
                yield child
        if not found:
            raise ValueError(f"No supported image files found in directory: {path}")
    else:
        raise ValueError(f"Path is neither a file nor a directory: {path}")


def load_image(path: Path) -> Image.Image:
    """Open an image file and return a PIL Image object.

    Args:
        path: Path to the image file.

    Returns:
        A ``PIL.Image.Image`` instance.
    """
    return Image.open(path)


def image_metadata(path: Path, image: Image.Image) -> dict:
    """Extract metadata from an image file.

    Args:
        path: The filesystem path of the image.
        image: The opened PIL image.

    Returns:
        A dict with keys ``filename``, ``file_type``, ``width``, ``height``.
    """
    width, height = image.size
    return {
        "filename": path.name,
        "file_type": (image.format or path.suffix.lstrip(".")).upper(),
        "width": width,
        "height": height,
    }


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 string.

    Args:
        image: The PIL image to encode.
        fmt: PIL format string (default ``"JPEG"``).

    Returns:
        A base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
