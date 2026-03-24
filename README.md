# qd-image-search

A local image search CLI tool that uses a [QDrant](https://qdrant.tech/) vector database (port 6333)
and a local [CLIP](https://openai.com/research/clip) model to add and search images.

## Installation

```bash
pip install -e .
```

## Usage

### Add images to a collection

```bash
# Add a single image
qd-image-search add Cars-Collection /data/images/car.jpg

# Add all images from a directory
qd-image-search add Cars-Collection /data/images/cars
```

Each image is processed into a CLIP vector embedding and uploaded to QDrant together with its
metadata (UUID, filename, file type, dimensions) and the image payload.

### Search a collection

```bash
qd-image-search search Cars-Collection "red sports car"
```

Returns the top matching metadata records from QDrant for the given text query.

### Options

```
qd-image-search add --help
qd-image-search search --help
```

Both commands accept `--host` (default `localhost`) and `--port` (default `6333`) options to
configure the QDrant server address.  The `search` command also accepts `--limit` (default `10`)
to control how many results are returned.

## Development

```bash
pip install -e ".[dev]"
pytest
```

