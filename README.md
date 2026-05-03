# Face Finder

CLI tool to find images containing a specific person using facial recognition given a set of candidate images.

## Installation

```bash
uv sync
```

## Commands

### analyze

Analyze reference images to identify the common face and optionally save the encoding for reuse.

```bash
uv run face-finder analyze \
  --reference-dir ./known_person/ \
  --save-encoding ./person.npy \
  --save-thumbnail ./person_thumb.jpg
```

**Options:**
- `-r, --reference-dir`: Directory containing reference images (required)
- `-t, --tolerance`: Match tolerance (default 0.40, lower = stricter)
- `-m, --model`: Face detection model: `hog` (fast), `cnn` (handles profiles), `auto` (hog first, cnn fallback)
- `-e, --save-encoding`: Save face encoding to a .npy file for later use
- `-T, --save-thumbnail`: Save a thumbnail of the detected face for verification

### find

Find images containing a specific person and copy matches to output directory.

```bash
uv run face-finder find \
  --reference-dir ./known_person/ \
  --search-dir ./photos_to_search/ \
  --output-dir ./matches/
```

Or using a pre-computed encoding:

```bash
uv run face-finder find \
  --encoding ./person.npy \
  --search-dir ./photos_to_search/ \
  --output-dir ./matches/
```

Images can also be piped via stdin instead of using `--search-dir`:

```bash
find ./photos -name "*.jpg" | uv run face-finder find \
  --encoding ./person.npy \
  --output-dir ./matches/
```

**Options:**
- `-r, --reference-dir`: Directory with reference images (required unless `--encoding` is used)
- `-e, --encoding`: Pre-computed face encoding file (.npy) from `analyze`
- `-s, --search-dir`: Directory to search (if omitted, reads image paths from stdin)
- `-o, --output-dir`: Directory to copy matching images (required)
- `-t, --tolerance`: Match tolerance (default 0.40, lower = stricter)
- `-m, --model`: Face detection model: `hog`, `cnn`, or `auto`
- `-R, --recursive`: Search subdirectories recursively
- `-n, --limit`: Maximum number of matches to find
- `-l, --log`: Write detailed processing log to file
- `-v, --verbose`: Show distance values for each image
- `--no-mtcnn`: Disable MTCNN prefilter (use skin-color heuristics instead)
- `--reject-dir`: Copy rejected images to a directory (for debugging)
- `-D, --skip-duplicates`: Skip duplicate images by content hash

**Output modes:**
- Non-verbose (default): prints every processed image path to stdout, enabling resumability
- Verbose (`-v`): prints detailed match status for each image

### prefilter

Run only the fast prefilter to quickly narrow down a large photo collection to images likely containing people.

```bash
uv run face-finder prefilter \
  --search-dir ./large_collection/ \
  --output-dir ./candidates/
```

**Options:**
- `-s, --search-dir`: Directory to search (if omitted, reads from stdin)
- `-o, --output-dir`: Directory to copy passing images (required)
- `-e, --encoding`: Optional encoding file for targeted skin color filtering
- `-R, --recursive`: Search subdirectories recursively
- `-n, --limit`: Maximum number of passing images to copy
- `-l, --log`: Write detailed processing log to file
- `-v, --verbose`: Show prefilter scores for each image
- `--no-mtcnn`: Disable MTCNN prefilter (use skin-color heuristics instead)
- `--reject-dir`: Copy rejected images to a directory
- `-D, --skip-duplicates`: Skip duplicate images by content hash

### extract

Extract all faces from images, saving cropped thumbnails and face encodings.

```bash
uv run face-finder extract \
  --search-dir ./photos/ \
  --output-dir ./faces/
```

For each face found, saves `{source_stem}_face_{N}.jpg` and `{source_stem}_face_{N}.npy`.

**Options:**
- `-s, --search-dir`: Directory to search (if omitted, reads from stdin)
- `-o, --output-dir`: Directory to save extracted faces (required)
- `-m, --model`: Face detection model: `hog`, `cnn`, or `auto`
- `-p, --padding`: Padding around face crop as fraction (default 0.3 = 30%)
- `-R, --recursive`: Search subdirectories recursively
- `-v, --verbose`: Show per-image details
- `-D, --skip-duplicates`: Skip duplicate images by content hash
