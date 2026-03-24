"""CLI entry point for qd-image-search."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .embeddings import embed_image, embed_text, vector_size
from .image_utils import image_metadata, image_to_base64, iter_image_paths, load_image
from .qdrant import ensure_collection, get_client, search_collection, upsert_image_point


@click.group()
def main() -> None:
    """qd-image-search – Add images to and search QDrant collections."""


@main.command("add")
@click.argument("collection")
@click.argument("path", type=click.Path(exists=True))
@click.option("--host", default="localhost", show_default=True, help="QDrant host")
@click.option("--port", default=6333, show_default=True, help="QDrant port")
def add_command(collection: str, path: str, host: str, port: int) -> None:
    """Add image(s) at PATH to COLLECTION.

    PATH may be a single image file or a directory of image files.
    """
    image_path = Path(path)

    try:
        image_paths = list(iter_image_paths(image_path))
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    client = get_client(host=host, port=port)
    vec_size = vector_size()

    try:
        ensure_collection(client, collection, vec_size)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error ensuring collection exists in QDrant at {host}:{port}: {exc}", err=True)
        sys.exit(1)

    added = 0
    for img_path in image_paths:
        try:
            image = load_image(img_path)
            meta = image_metadata(img_path, image)
            vector = embed_image(image)
            b64 = image_to_base64(image)
            point_id = upsert_image_point(client, collection, vector, meta, b64)
            click.echo(
                f"Added {img_path.name!r} → {collection!r} (id={point_id})"
            )
            added += 1
        except Exception as exc:  # noqa: BLE001
            click.echo(f"Warning: skipping {img_path}: {exc}", err=True)

    click.echo(f"\n{added} image(s) added to collection {collection!r}.")


@main.command("search")
@click.argument("collection")
@click.argument("query")
@click.option("--limit", default=10, show_default=True, help="Number of results to return")
@click.option("--host", default="localhost", show_default=True, help="QDrant host")
@click.option("--port", default=6333, show_default=True, help="QDrant port")
def search_command(collection: str, query: str, limit: int, host: str, port: int) -> None:
    """Search COLLECTION with a text QUERY string."""
    client = get_client(host=host, port=port)

    try:
        query_vector = embed_text(query)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error generating query embedding: {exc}", err=True)
        sys.exit(1)

    try:
        results = search_collection(client, collection, query_vector, limit=limit)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error searching QDrant at {host}:{port}: {exc}", err=True)
        sys.exit(1)

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"Top {len(results)} result(s) for {query!r} in {collection!r}:\n")
    for i, record in enumerate(results, start=1):
        score = record.pop("score", None)
        parts = "  ".join(f"{k}={v!r}" for k, v in record.items())
        score_str = f"  score={score:.4f}" if score is not None else ""
        click.echo(f"{i:>3}. {parts}{score_str}")
