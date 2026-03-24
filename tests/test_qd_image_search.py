"""Tests for qd-image-search."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_DIM = 512


def _make_fake_vector() -> list[float]:
    import numpy as np

    v = np.random.rand(VECTOR_DIM).astype(float)
    v /= float(np.linalg.norm(v))
    return v.tolist()


def _make_test_image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _save_test_image(path: Path, width: int = 64, height: int = 64) -> None:
    img = _make_test_image(width, height)
    img.save(path)


# ---------------------------------------------------------------------------
# image_utils tests
# ---------------------------------------------------------------------------


class TestImageUtils:
    def test_iter_image_paths_single_file(self, tmp_path: Path):
        from qd_image_search.image_utils import iter_image_paths

        img_file = tmp_path / "test.jpg"
        _save_test_image(img_file)
        paths = list(iter_image_paths(img_file))
        assert paths == [img_file]

    def test_iter_image_paths_directory(self, tmp_path: Path):
        from qd_image_search.image_utils import iter_image_paths

        for name in ("a.jpg", "b.png", "c.txt"):
            (tmp_path / name).write_bytes(b"")
        # Overwrite the images with real image data
        _save_test_image(tmp_path / "a.jpg")
        _save_test_image(tmp_path / "b.png")

        paths = list(iter_image_paths(tmp_path))
        names = {p.name for p in paths}
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_iter_image_paths_missing(self, tmp_path: Path):
        from qd_image_search.image_utils import iter_image_paths

        with pytest.raises(ValueError, match="does not exist"):
            list(iter_image_paths(tmp_path / "nonexistent.jpg"))

    def test_iter_image_paths_unsupported_extension(self, tmp_path: Path):
        from qd_image_search.image_utils import iter_image_paths

        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF")
        with pytest.raises(ValueError, match="Unsupported image format"):
            list(iter_image_paths(p))

    def test_iter_image_paths_empty_directory(self, tmp_path: Path):
        from qd_image_search.image_utils import iter_image_paths

        with pytest.raises(ValueError, match="No supported image files found"):
            list(iter_image_paths(tmp_path))

    def test_image_metadata(self, tmp_path: Path):
        from qd_image_search.image_utils import image_metadata, load_image

        img_file = tmp_path / "photo.png"
        _save_test_image(img_file, width=100, height=200)
        image = load_image(img_file)
        meta = image_metadata(img_file, image)

        assert meta["filename"] == "photo.png"
        assert meta["file_type"] == "PNG"
        assert meta["width"] == 100
        assert meta["height"] == 200

    def test_image_to_base64_is_string(self):
        from qd_image_search.image_utils import image_to_base64

        img = _make_test_image()
        b64 = image_to_base64(img)
        assert isinstance(b64, str)
        # Must be valid base64
        import base64

        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# qdrant module tests
# ---------------------------------------------------------------------------


class TestQdrant:
    def _mock_client(self):
        client = MagicMock()
        # Simulate empty collection list
        client.get_collections.return_value = MagicMock(collections=[])
        return client

    def test_ensure_collection_creates_when_absent(self):
        from qd_image_search.qdrant import ensure_collection

        client = self._mock_client()
        ensure_collection(client, "MyCollection", 512)
        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "MyCollection"

    def test_ensure_collection_skips_existing(self):
        from qd_image_search.qdrant import ensure_collection

        client = self._mock_client()
        existing = MagicMock()
        existing.name = "MyCollection"
        client.get_collections.return_value = MagicMock(collections=[existing])

        ensure_collection(client, "MyCollection", 512)
        client.create_collection.assert_not_called()

    def test_upsert_image_point_returns_uuid(self):
        from qd_image_search.qdrant import upsert_image_point

        client = self._mock_client()
        vector = _make_fake_vector()
        meta = {"filename": "dog.jpg", "file_type": "JPEG", "width": 64, "height": 64}
        point_id = upsert_image_point(client, "Animals", vector, meta, "base64data")

        assert uuid.UUID(point_id)  # valid UUID
        client.upsert.assert_called_once()

    def test_search_collection_returns_records(self):
        from qd_image_search.qdrant import search_collection

        client = self._mock_client()
        hit = MagicMock()
        hit.score = 0.95
        hit.payload = {
            "id": str(uuid.uuid4()),
            "filename": "car.jpg",
            "file_type": "JPEG",
            "width": 640,
            "height": 480,
            "image": "base64encodeddata",
        }
        client.search.return_value = [hit]

        results = search_collection(client, "Cars", _make_fake_vector(), limit=5)
        assert len(results) == 1
        assert results[0]["filename"] == "car.jpg"
        assert results[0]["score"] == 0.95
        # "image" payload should be stripped from results
        assert "image" not in results[0]

    def test_search_collection_empty(self):
        from qd_image_search.qdrant import search_collection

        client = self._mock_client()
        client.search.return_value = []
        results = search_collection(client, "Cars", _make_fake_vector())
        assert results == []


# ---------------------------------------------------------------------------
# CLI integration tests (mocked QDrant + CLIP)
# ---------------------------------------------------------------------------


class TestCLI:
    def _fake_embed_image(self, image):
        return _make_fake_vector()

    def _fake_embed_text(self, text):
        return _make_fake_vector()

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_qdrant(self):
        """Patch QDrant client at the CLI level."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        with patch("qd_image_search.qdrant.QdrantClient", return_value=mock_client):
            yield mock_client

    @pytest.fixture
    def mock_embeddings(self):
        with (
            patch(
                "qd_image_search.cli.embed_image",
                side_effect=self._fake_embed_image,
            ),
            patch(
                "qd_image_search.cli.embed_text",
                side_effect=self._fake_embed_text,
            ),
            patch("qd_image_search.cli.vector_size", return_value=VECTOR_DIM),
        ):
            yield

    def test_add_single_file(self, runner, tmp_path, mock_qdrant, mock_embeddings):
        from qd_image_search.cli import main

        img_file = tmp_path / "cat.jpg"
        _save_test_image(img_file)

        result = runner.invoke(main, ["add", "Pets", str(img_file)])
        assert result.exit_code == 0, result.output
        assert "cat.jpg" in result.output
        assert "Pets" in result.output
        mock_qdrant.upsert.assert_called_once()

    def test_add_directory(self, runner, tmp_path, mock_qdrant, mock_embeddings):
        from qd_image_search.cli import main

        for name in ("cat.jpg", "dog.png"):
            _save_test_image(tmp_path / name)

        result = runner.invoke(main, ["add", "Pets", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert mock_qdrant.upsert.call_count == 2

    def test_add_nonexistent_path(self, runner, mock_qdrant, mock_embeddings):
        from qd_image_search.cli import main

        result = runner.invoke(main, ["add", "Pets", "/nonexistent/path"])
        assert result.exit_code != 0 or "Error" in result.output

    def test_search_returns_results(self, runner, mock_qdrant, mock_embeddings):
        from qd_image_search.cli import main

        hit = MagicMock()
        hit.score = 0.87
        hit.payload = {
            "id": str(uuid.uuid4()),
            "filename": "sports_car.jpg",
            "file_type": "JPEG",
            "width": 1920,
            "height": 1080,
            "image": "base64data",
        }
        mock_qdrant.search.return_value = [hit]

        result = runner.invoke(main, ["search", "Cars", "red sports car"])
        assert result.exit_code == 0, result.output
        assert "sports_car.jpg" in result.output
        assert "score=" in result.output

    def test_search_no_results(self, runner, mock_qdrant, mock_embeddings):
        from qd_image_search.cli import main

        mock_qdrant.search.return_value = []
        result = runner.invoke(main, ["search", "Cars", "flying unicorn"])
        assert result.exit_code == 0, result.output
        assert "No results found" in result.output

    def test_help(self, runner):
        from qd_image_search.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "add" in result.output
        assert "search" in result.output

    def test_add_help(self, runner):
        from qd_image_search.cli import main

        result = runner.invoke(main, ["add", "--help"])
        assert result.exit_code == 0

    def test_search_help(self, runner):
        from qd_image_search.cli import main

        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
