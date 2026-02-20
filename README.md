# Face Finder

CLI tool to find images containing a specific person using facial recognition.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run face-finder find \
  --reference-dir ./known_person/ \
  --search-dir ./photos_to_search/ \
  --output-dir ./matches/
```

## Options

- `-r, --reference-dir`: Directory with reference images of the person (required)
- `-s, --search-dir`: Directory to search for the person (required)
- `-o, --output-dir`: Directory to copy matching images (required)
- `-t, --tolerance`: Match tolerance (default 0.6, lower = stricter)
- `-R, --recursive`: Search subdirectories recursively
- `-n, --limit`: Maximum number of matches to find
