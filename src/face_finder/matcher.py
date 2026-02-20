"""Face matching and file operations module."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import face_recognition
import numpy as np
from numpy.typing import NDArray

from .detector import IMAGE_EXTENSIONS, load_image_with_exif_rotation, resize_for_cnn
from .prefilter import PrefilterResult, should_skip_cnn, should_skip_cnn_targeted


@dataclass
class MatchInfo:
    """Detailed match information for verbose output."""

    matched: bool
    min_distance: float | None  # None if no faces detected
    num_faces: int
    prefilter_skipped: bool = False  # True if CNN was skipped by prefilter
    prefilter_reason: str = ""  # Reason for skipping (e.g., "no skin tones detected")
    prefilter_entropy: float | None = None  # Shannon entropy (bits) from prefilter
    prefilter_skin_pct: float | None = None  # % of skin-toned pixels from prefilter (generic)
    prefilter_ref_skin_pct: float | None = None  # % of pixels matching reference skin color
    used_targeted_filter: bool = False  # True if reference skin filter was used


def iter_images(directory: Path, recursive: bool = False) -> Iterator[Path]:
    """
    Generator yielding image paths one at a time.

    Uses lazy directory traversal to minimize memory usage.
    """
    if recursive:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
    else:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


class MatchError(Exception):
    """Error during face matching."""
    pass


def is_match(
    image_path: Path,
    reference_encoding: NDArray[np.float64],
    tolerance: float = 0.40,
    model: str = "hog",
    skin_info: dict | None = None,
) -> bool:
    """
    Check if an image contains a face matching the reference encoding.

    Args:
        image_path: Path to the image file
        reference_encoding: Face encoding to match against
        tolerance: Distance threshold for matching
        model: Face detection model:
               - "hog" (fast, frontal faces only)
               - "cnn" (slower, better for angles/profiles)
               - "auto" (try hog first, fall back to cnn if no faces found)
        skin_info: Optional dict with reference skin color for targeted prefiltering

    Returns True if any face in the image matches within the tolerance.
    Handles EXIF orientation to properly detect faces in rotated images.
    Raises MatchError if processing fails.
    """
    try:
        image = load_image_with_exif_rotation(image_path)

        # Validate image shape for face detection (must be 3-channel RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False  # Skip invalid images

        # Ensure correct dtype for dlib
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Resize large images for CNN to prevent hangs/crashes
        if model in ("cnn", "auto"):
            image = resize_for_cnn(image)

        # Prefilter for CNN: skip images that definitely have no people
        if model == "cnn":
            prefilter = should_skip_cnn_targeted(image, skin_info)
            if prefilter.should_skip:
                return False  # No people detected, skip expensive CNN

        if model == "auto":
            # Try HOG first (fast)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                # Prefilter before CNN fallback
                prefilter = should_skip_cnn_targeted(image, skin_info)
                if prefilter.should_skip:
                    return False  # No people detected, skip expensive CNN
                # Fall back to CNN if no faces found
                face_locations = face_recognition.face_locations(image, model="cnn")
        else:
            face_locations = face_recognition.face_locations(image, model=model)

        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            return False

        # Check each face in the image against the reference
        distances = face_recognition.face_distance(encodings, reference_encoding)
        return bool(np.any(distances <= tolerance))
    except Exception as e:
        raise MatchError(f"Error processing {image_path.name}: {e}") from e


def check_match(
    image_path: Path,
    reference_encoding: NDArray[np.float64],
    tolerance: float = 0.40,
    model: str = "hog",
    prefilter_result: PrefilterResult | None = None,
    skin_info: dict | None = None,
) -> MatchInfo:
    """
    Check if an image contains a face matching the reference encoding.

    Returns MatchInfo with detailed match information including distance values.
    This is useful for verbose output and tuning tolerance values.

    Args:
        image_path: Path to the image file
        reference_encoding: Face encoding to match against
        tolerance: Distance threshold for matching (0.40 = default, 0.6 = permissive)
        model: Face detection model ("hog", "cnn", or "auto")
        prefilter_result: Pre-computed prefilter result (avoids recomputing if provided)
        skin_info: Optional dict with reference skin color for targeted prefiltering

    Returns:
        MatchInfo with matched status, minimum distance, and number of faces detected.
    """
    try:
        image = load_image_with_exif_rotation(image_path)

        # Validate image shape for face detection (must be 3-channel RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            return MatchInfo(matched=False, min_distance=None, num_faces=0)

        # Ensure correct dtype for dlib
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Resize large images for CNN to prevent hangs/crashes
        if model in ("cnn", "auto"):
            image = resize_for_cnn(image)

        # Track prefilter results for verbose output
        # Use pre-computed result if provided, otherwise compute
        prefilter: PrefilterResult | None = prefilter_result

        # Prefilter for CNN: skip images that definitely have no people
        if model == "cnn":
            if prefilter is None:
                prefilter = should_skip_cnn_targeted(image, skin_info)
            if prefilter.should_skip:
                return MatchInfo(
                    matched=False,
                    min_distance=None,
                    num_faces=0,
                    prefilter_skipped=True,
                    prefilter_reason=prefilter.reason,
                    prefilter_entropy=prefilter.entropy,
                    prefilter_skin_pct=prefilter.skin_percentage,
                    prefilter_ref_skin_pct=prefilter.reference_skin_percentage,
                    used_targeted_filter=prefilter.used_targeted_filter,
                )

        if model == "auto":
            # Try HOG first (fast)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                # Prefilter before CNN fallback (use pre-computed if available)
                if prefilter is None:
                    prefilter = should_skip_cnn_targeted(image, skin_info)
                if prefilter.should_skip:
                    return MatchInfo(
                        matched=False,
                        min_distance=None,
                        num_faces=0,
                        prefilter_skipped=True,
                        prefilter_reason=prefilter.reason,
                        prefilter_entropy=prefilter.entropy,
                        prefilter_skin_pct=prefilter.skin_percentage,
                        prefilter_ref_skin_pct=prefilter.reference_skin_percentage,
                        used_targeted_filter=prefilter.used_targeted_filter,
                    )
                # Fall back to CNN if no faces found
                face_locations = face_recognition.face_locations(image, model="cnn")
        else:
            face_locations = face_recognition.face_locations(image, model=model)

        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            return MatchInfo(
                matched=False,
                min_distance=None,
                num_faces=0,
                prefilter_entropy=prefilter.entropy if prefilter else None,
                prefilter_skin_pct=prefilter.skin_percentage if prefilter else None,
                prefilter_ref_skin_pct=prefilter.reference_skin_percentage if prefilter else None,
                used_targeted_filter=prefilter.used_targeted_filter if prefilter else False,
            )

        # Check each face in the image against the reference
        distances = face_recognition.face_distance(encodings, reference_encoding)
        min_distance = float(np.min(distances))
        matched = min_distance <= tolerance

        return MatchInfo(
            matched=matched,
            min_distance=min_distance,
            num_faces=len(encodings),
            prefilter_entropy=prefilter.entropy if prefilter else None,
            prefilter_skin_pct=prefilter.skin_percentage if prefilter else None,
            prefilter_ref_skin_pct=prefilter.reference_skin_percentage if prefilter else None,
            used_targeted_filter=prefilter.used_targeted_filter if prefilter else False,
        )
    except Exception as e:
        raise MatchError(f"Error processing {image_path.name}: {e}") from e


def copy_image(source: Path, output_directory: Path) -> Path:
    """
    Copy a matched image to the output directory.

    Handles filename conflicts by appending a counter.
    Returns the path to the copied file.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    dest = output_directory / source.name

    # Handle filename conflicts
    if dest.exists():
        counter = 1
        stem = source.stem
        suffix = source.suffix
        while dest.exists():
            dest = output_directory / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(source, dest)
    return dest


class MatchResult:
    """Results from processing images."""

    def __init__(self) -> None:
        self.total_scanned = 0
        self.matches_found = 0
        self.errors = 0
        self.matched_files: list[Path] = []


def process_images_prefilter_only(
    search_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    limit: int | None = None,
    progress_callback: callable = None,
    skin_info: dict | None = None,
) -> MatchResult:
    """
    Process images using only prefilter, copying passing images.

    This is useful for quickly filtering a large photo collection to images
    likely containing people, without running expensive face detection.

    Args:
        search_dir: Directory to search for images
        output_dir: Directory to copy passing images to
        recursive: Whether to search subdirectories
        limit: Maximum number of passing images to copy
        progress_callback: Called after each image with:
            (image_path: Path, passed: bool, error: bool, error_msg: str | None,
             prefilter: PrefilterResult | None)
        skin_info: Optional dict with reference skin color for targeted prefiltering.
            When provided, uses reference-targeted prefilter instead of generic.

    Returns:
        MatchResult with total_scanned, matches_found (images that passed),
        and errors count.
    """
    result = MatchResult()

    for image_path in iter_images(search_dir, recursive):
        result.total_scanned += 1

        try:
            image = load_image_with_exif_rotation(image_path)

            # Validate image shape (must be 3-channel RGB)
            if len(image.shape) != 3 or image.shape[2] != 3:
                if progress_callback:
                    progress_callback(image_path, passed=False, error=False, error_msg=None, prefilter=None)
                continue

            # Ensure correct dtype
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Resize for consistent prefilter behavior
            image = resize_for_cnn(image)

            # Run prefilter
            prefilter = should_skip_cnn_targeted(image, skin_info)

            if not prefilter.should_skip:
                # Image passes prefilter - copy it
                copy_image(image_path, output_dir)
                result.matches_found += 1

                if progress_callback:
                    progress_callback(image_path, passed=True, error=False, error_msg=None, prefilter=prefilter)

                # Check limit
                if limit is not None and result.matches_found >= limit:
                    break
            else:
                if progress_callback:
                    progress_callback(image_path, passed=False, error=False, error_msg=None, prefilter=prefilter)

        except Exception as e:
            result.errors += 1
            if progress_callback:
                progress_callback(image_path, passed=False, error=True, error_msg=str(e), prefilter=None)

    return result


def process_images(
    search_dir: Path,
    output_dir: Path,
    reference_encoding: NDArray[np.float64],
    tolerance: float = 0.40,
    recursive: bool = False,
    limit: int | None = None,
    progress_callback: callable = None,
    pre_process_callback: callable = None,
    model: str = "hog",
    verbose: bool = False,
    prefilter_callback: callable = None,
    skin_info: dict | None = None,
) -> MatchResult:
    """
    Main processing loop: iterate images, check matches, copy immediately.

    Uses streaming/generator pattern for memory efficiency.

    Args:
        model: Face detection model - "hog" (fast) or "cnn" (better for profiles)
        pre_process_callback: Called BEFORE each image is processed. Useful for
            logging which image is being processed in case of crashes.
            Signature: pre_process_callback(image_path: Path, model: str)
        verbose: If True, progress_callback receives additional match_info parameter
            with MatchInfo containing distance values.
        prefilter_callback: Called immediately after prefilter runs (CNN/auto only).
            Fires BEFORE the slow CNN detection. Useful for showing prefilter results
            in real-time. Signature: prefilter_callback(image_path: Path, prefilter: PrefilterResult)
        skin_info: Optional dict with reference skin color for targeted prefiltering.
            When provided, uses reference-targeted prefilter instead of generic.
    """
    result = MatchResult()

    for image_path in iter_images(search_dir, recursive):
        result.total_scanned += 1

        # Log BEFORE processing (critical for crash diagnosis)
        if pre_process_callback:
            pre_process_callback(image_path, model)

        # Run prefilter immediately for CNN/auto models and fire callback
        prefilter_result: PrefilterResult | None = None
        if verbose and model in ("cnn", "auto") and prefilter_callback:
            try:
                image = load_image_with_exif_rotation(image_path)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    image = resize_for_cnn(image)
                    prefilter_result = should_skip_cnn_targeted(image, skin_info)
                    prefilter_callback(image_path, prefilter_result)
            except Exception:
                pass  # Prefilter errors will be handled by check_match

        try:
            if verbose:
                # Use check_match for detailed info (pass prefilter to avoid recompute)
                match_info = check_match(
                    image_path, reference_encoding, tolerance, model=model,
                    prefilter_result=prefilter_result, skin_info=skin_info
                )
                if match_info.matched:
                    copied_path = copy_image(image_path, output_dir)
                    result.matches_found += 1
                    result.matched_files.append(copied_path)

                    if progress_callback:
                        progress_callback(image_path, matched=True, match_info=match_info)

                    # Check if we've reached the limit
                    if limit is not None and result.matches_found >= limit:
                        break
                else:
                    if progress_callback:
                        progress_callback(image_path, matched=False, match_info=match_info)
            else:
                # Use is_match for simple bool result (faster, less memory)
                if is_match(image_path, reference_encoding, tolerance, model=model, skin_info=skin_info):
                    copied_path = copy_image(image_path, output_dir)
                    result.matches_found += 1
                    result.matched_files.append(copied_path)

                    if progress_callback:
                        progress_callback(image_path, matched=True)

                    # Check if we've reached the limit
                    if limit is not None and result.matches_found >= limit:
                        break
                else:
                    if progress_callback:
                        progress_callback(image_path, matched=False)
        except Exception as e:
            result.errors += 1
            if progress_callback:
                progress_callback(image_path, matched=False, error=True, error_msg=str(e))

    return result
