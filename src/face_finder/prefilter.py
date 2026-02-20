"""Fast prefilter to skip CNN on images without human presence."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Lazy-loaded MTCNN detector (expensive to initialize)
_mtcnn_detector = None


def get_mtcnn_detector():
    """Lazy-load MTCNN detector (expensive to initialize, ~1-2s first call)."""
    global _mtcnn_detector
    if _mtcnn_detector is None:
        from mtcnn import MTCNN
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector


def mtcnn_prefilter(image: NDArray[np.uint8]) -> "PrefilterResult":
    """
    Use MTCNN to detect if image contains faces.

    MTCNN is a fast (~50-100ms) face detector using 3 cascaded CNNs.
    Much more accurate than pixel-based heuristics for face detection.

    Args:
        image: RGB image array (H, W, 3)

    Returns:
        PrefilterResult with should_skip=True if no faces found.
    """
    h, w = image.shape[:2]
    image_size = (h, w)

    detector = get_mtcnn_detector()
    faces = detector.detect_faces(image)

    if not faces:
        return PrefilterResult(
            should_skip=True,
            reason="MTCNN found no faces",
            entropy=None,
            skin_percentage=None,
            image_size=image_size,
            reference_skin_percentage=None,
            used_targeted_filter=False,
            mtcnn_faces=0,
        )

    return PrefilterResult(
        should_skip=False,
        reason="",
        entropy=None,
        skin_percentage=None,
        image_size=image_size,
        reference_skin_percentage=None,
        used_targeted_filter=False,
        mtcnn_faces=len(faces),
    )


@dataclass
class PrefilterResult:
    """Detailed prefilter results including computed scores."""

    should_skip: bool
    reason: str
    entropy: float | None = None  # Shannon entropy (bits)
    skin_percentage: float | None = None  # % of skin-toned pixels (generic heuristic)
    image_size: tuple[int, int] | None = None  # (height, width)
    reference_skin_percentage: float | None = None  # % of pixels matching reference skin
    used_targeted_filter: bool = False  # True if reference skin filter was used
    blob_aspect_ratio: float | None = None  # height/width of largest skin blob
    blob_percentage: float | None = None  # % of image covered by largest blob
    mtcnn_faces: int | None = None  # Number of faces detected by MTCNN (None if not run)


def calculate_entropy(image: NDArray[np.uint8]) -> float:
    """Calculate Shannon entropy of image histogram.

    Low entropy indicates a blank/solid/uniform image that cannot contain faces.
    Typical photos have entropy > 5 bits, while blank images have < 3.5 bits.
    """
    gray = np.mean(image, axis=2).astype(np.uint8) if len(image.shape) == 3 else image
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))


def calculate_skin_percentage(image: NDArray[np.uint8]) -> tuple[float, NDArray[np.bool_]]:
    """Estimate percentage of skin-toned pixels using RGB heuristics.

    Uses a color-based heuristic that works across various skin tones.
    Returns tuple of (percentage 0-100, boolean mask of skin pixels).
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    # Skin tone heuristic (works across many skin tones)
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b)) > 15) &
        (np.abs(r.astype(np.int16) - g.astype(np.int16)) > 15) &
        (r > g) & (r > b)
    )
    return 100 * np.sum(skin_mask) / skin_mask.size, skin_mask


def analyze_skin_blob(skin_mask: NDArray[np.bool_]) -> tuple[float, float]:
    """
    Analyze the largest skin-colored blob for face-like characteristics.

    Returns:
        Tuple of (aspect_ratio, blob_percentage).
        - aspect_ratio: height/width of bounding box (faces ~0.5-2.0)
        - blob_percentage: % of image covered by largest blob
    """
    from scipy import ndimage

    labeled, num_features = ndimage.label(skin_mask)
    if num_features == 0:
        return 0.0, 0.0

    component_sizes = ndimage.sum(skin_mask, labeled, range(1, num_features + 1))
    largest_idx = np.argmax(component_sizes) + 1
    largest_area = int(component_sizes[largest_idx - 1])

    slice_y, slice_x = ndimage.find_objects(labeled == largest_idx)[0]
    h = slice_y.stop - slice_y.start
    w = slice_x.stop - slice_x.start
    aspect_ratio = h / w if w > 0 else 0.0
    blob_pct = 100.0 * largest_area / skin_mask.size

    return aspect_ratio, blob_pct


def should_skip_cnn(image: NDArray[np.uint8]) -> PrefilterResult:
    """
    Determine if image can safely skip CNN detection.

    Uses multi-stage filtering to identify images that definitely have no people:
    1. Size check: Images too small to contain a face
    2. Entropy check: Blank/solid images
    3. Skin tone check: Images with no skin-colored pixels

    Returns:
        PrefilterResult with skip decision, reason, and computed scores.
        Only skips when confident there are NO people.
    """
    h, w = image.shape[:2]
    image_size = (h, w)

    # Stage 1: Size check (free)
    if min(h, w) < 60:
        return PrefilterResult(
            should_skip=True,
            reason="image too small",
            entropy=None,
            skin_percentage=None,
            image_size=image_size,
        )

    # Stage 2: Entropy check (~13ms)
    entropy = calculate_entropy(image)
    if entropy < 3.5:
        return PrefilterResult(
            should_skip=True,
            reason="blank/solid image",
            entropy=entropy,
            skin_percentage=None,
            image_size=image_size,
        )

    # Stage 3: Skin tone check (~7ms)
    skin_pct, _ = calculate_skin_percentage(image)
    if skin_pct < 1.0:
        return PrefilterResult(
            should_skip=True,
            reason="no skin tones detected",
            entropy=entropy,
            skin_percentage=skin_pct,
            image_size=image_size,
        )

    return PrefilterResult(
        should_skip=False,
        reason="",
        entropy=entropy,
        skin_percentage=skin_pct,
        image_size=image_size,
    )


def calculate_reference_skin_percentage(
    image: NDArray[np.uint8],
    skin_info: dict,
    threshold: float = 1.5,
) -> float:
    """
    Calculate percentage of pixels matching reference skin color.

    Args:
        image: RGB image array (H, W, 3)
        skin_info: Dict with "median_rgb" and "std_rgb" tuples
        threshold: Number of standard deviations for tolerance (default: 1.5)

    Returns:
        Percentage (0-100) of pixels within tolerance of reference skin color.
    """
    median = np.array(skin_info["median_rgb"], dtype=np.float32)
    # Cap std between 10 and 25 to prevent overly wide/narrow tolerance ranges
    std = np.clip(np.array(skin_info["std_rgb"], dtype=np.float32), 10.0, 25.0)

    diff = np.abs(image.astype(np.float32) - median)
    within_tolerance = np.all(diff <= threshold * std, axis=2)
    return 100.0 * np.sum(within_tolerance) / within_tolerance.size


def should_skip_cnn_targeted(
    image: NDArray[np.uint8],
    skin_info: dict | None = None,
    use_mtcnn: bool = True,
) -> PrefilterResult:
    """
    Enhanced prefilter using MTCNN face detection and/or skin color heuristics.

    Filtering stages (in order):
    1. Size check (~0ms) - Skip images too small to contain faces
    2. Entropy check (~13ms) - Skip blank/solid images
    3. MTCNN check (~50-100ms) - Use neural network face detection (if enabled)
    4. Skin heuristics (~20ms) - Fall back to pixel-based skin color detection (if MTCNN disabled)

    Args:
        image: RGB image array (H, W, 3)
        skin_info: Optional dict with "median_rgb" and "std_rgb" from reference analysis
        use_mtcnn: If True, use MTCNN face detection instead of skin heuristics.
            MTCNN is more accurate but requires TensorFlow. Default: True.

    Returns:
        PrefilterResult with skip decision and computed scores.
    """
    h, w = image.shape[:2]
    image_size = (h, w)

    # Stage 1: Size check (free)
    if min(h, w) < 60:
        return PrefilterResult(
            should_skip=True,
            reason="image too small",
            entropy=None,
            skin_percentage=None,
            image_size=image_size,
            reference_skin_percentage=None,
            used_targeted_filter=skin_info is not None,
        )

    # Stage 2: Entropy check (~13ms)
    entropy = calculate_entropy(image)
    if entropy < 3.5:
        return PrefilterResult(
            should_skip=True,
            reason="blank/solid image",
            entropy=entropy,
            skin_percentage=None,
            image_size=image_size,
            reference_skin_percentage=None,
            used_targeted_filter=skin_info is not None,
        )

    # Stage 3: MTCNN face detection (~50-100ms)
    # MTCNN uses 3 cascaded neural networks for accurate face detection.
    # Much more reliable than pixel-based heuristics.
    if use_mtcnn:
        mtcnn_result = mtcnn_prefilter(image)
        if mtcnn_result.should_skip:
            return PrefilterResult(
                should_skip=True,
                reason=mtcnn_result.reason,
                entropy=entropy,
                skin_percentage=None,
                image_size=image_size,
                reference_skin_percentage=None,
                used_targeted_filter=skin_info is not None,
                mtcnn_faces=mtcnn_result.mtcnn_faces,
            )
        # MTCNN found faces - image passes prefilter
        return PrefilterResult(
            should_skip=False,
            reason="",
            entropy=entropy,
            skin_percentage=None,
            image_size=image_size,
            reference_skin_percentage=None,
            used_targeted_filter=skin_info is not None,
            mtcnn_faces=mtcnn_result.mtcnn_faces,
        )

    # Stage 4: Skin heuristics (only if MTCNN disabled)
    # Skin tone check
    # Use hybrid filter if reference skin info is available:
    # Require BOTH generic skin detection AND reference skin detection to pass.
    # This prevents false positives from wood, sand, etc. that may pass one but not both.
    if skin_info is not None:
        ref_skin_pct = calculate_reference_skin_percentage(image, skin_info)
        generic_skin_pct, skin_mask = calculate_skin_percentage(image)

        # Hybrid filter: require BOTH generic >= 6% AND ref_skin >= 1%
        # This filters images that pass one heuristic but not the other
        if generic_skin_pct < 6.0:
            return PrefilterResult(
                should_skip=True,
                reason="no skin tones detected",
                entropy=entropy,
                skin_percentage=generic_skin_pct,
                image_size=image_size,
                reference_skin_percentage=ref_skin_pct,
                used_targeted_filter=True,
            )

        if ref_skin_pct < 1.0:
            return PrefilterResult(
                should_skip=True,
                reason="no reference skin color",
                entropy=entropy,
                skin_percentage=generic_skin_pct,
                image_size=image_size,
                reference_skin_percentage=ref_skin_pct,
                used_targeted_filter=True,
            )

        # Stage 4: Blob shape check
        # Real faces form coherent, roughly-square blobs (aspect ratio 0.5-2.0)
        # Wood grain often forms extremely wide/flat patterns (aspect ratio ~0.08)
        aspect, blob_pct = analyze_skin_blob(skin_mask)
        if aspect < 0.3 or aspect > 2.0:
            return PrefilterResult(
                should_skip=True,
                reason="no face-shaped skin region",
                entropy=entropy,
                skin_percentage=generic_skin_pct,
                image_size=image_size,
                reference_skin_percentage=ref_skin_pct,
                used_targeted_filter=True,
                blob_aspect_ratio=aspect,
                blob_percentage=blob_pct,
            )

        return PrefilterResult(
            should_skip=False,
            reason="",
            entropy=entropy,
            skin_percentage=generic_skin_pct,
            image_size=image_size,
            reference_skin_percentage=ref_skin_pct,
            used_targeted_filter=True,
            blob_aspect_ratio=aspect,
            blob_percentage=blob_pct,
        )

    # Fall back to generic skin detection
    skin_pct, _ = calculate_skin_percentage(image)
    if skin_pct < 1.0:
        return PrefilterResult(
            should_skip=True,
            reason="no skin tones detected",
            entropy=entropy,
            skin_percentage=skin_pct,
            image_size=image_size,
            reference_skin_percentage=None,
            used_targeted_filter=False,
        )

    return PrefilterResult(
        should_skip=False,
        reason="",
        entropy=entropy,
        skin_percentage=skin_pct,
        image_size=image_size,
        reference_skin_percentage=None,
        used_targeted_filter=False,
    )
