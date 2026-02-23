"""Face matching and file operations module."""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import face_recognition
import numpy as np
from numpy.typing import NDArray

from .detector import IMAGE_EXTENSIONS, load_image_with_exif_rotation, resize_for_cnn
from .prefilter import PrefilterResult, should_skip_cnn_targeted


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
    prefilter_mtcnn_faces: int | None = None  # Number of faces detected by MTCNN (None if not run)


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


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class MatchError(Exception):
    """Error during face matching."""
    pass


def is_match(
    image_path: Path,
    reference_encoding: NDArray[np.float64],
    tolerance: float = 0.40,
    model: str = "hog",
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

    Returns True if any face in the image matches within the tolerance.
    Handles EXIF orientation to properly detect faces in rotated images.
    Raises MatchError if processing fails.

    Note: Prefiltering should be done by the caller before calling this function.
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

        if model == "auto":
            # Try HOG first (fast)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
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

    Returns:
        MatchInfo with matched status, minimum distance, and number of faces detected.
        Prefilter fields are left as defaults (caller populates them if needed).

    Note: Prefiltering should be done by the caller before calling this function.
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

        if model == "auto":
            # Try HOG first (fast)
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                # Fall back to CNN if no faces found
                face_locations = face_recognition.face_locations(image, model="cnn")
        else:
            face_locations = face_recognition.face_locations(image, model=model)

        encodings = face_recognition.face_encodings(image, face_locations)

        if not encodings:
            return MatchInfo(matched=False, min_distance=None, num_faces=0)

        # Check each face in the image against the reference
        distances = face_recognition.face_distance(encodings, reference_encoding)
        min_distance = float(np.min(distances))
        matched = min_distance <= tolerance

        return MatchInfo(
            matched=matched,
            min_distance=min_distance,
            num_faces=len(encodings),
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
    image_source: Path | Iterator[Path],
    output_dir: Path,
    recursive: bool = False,
    limit: int | None = None,
    progress_callback: callable = None,
    skin_info: dict | None = None,
    use_mtcnn: bool = True,
    reject_dir: Path | None = None,
    skip_duplicates: bool = False,
) -> MatchResult:
    """
    Process images using only prefilter, copying passing images.

    This is useful for quickly filtering a large photo collection to images
    likely containing people, without running expensive face detection.

    Args:
        image_source: Directory to search for images, or an iterator of image paths.
            When a directory (Path), uses iter_images() with recursive flag.
            When an iterator, uses it directly (recursive flag is ignored).
        output_dir: Directory to copy passing images to
        recursive: Whether to search subdirectories (only used when image_source is a directory)
        limit: Maximum number of passing images to copy
        progress_callback: Called after each image with:
            (image_path: Path, passed: bool, error: bool, error_msg: str | None,
             prefilter: PrefilterResult | None, duplicate: bool)
        skin_info: Optional dict with reference skin color for targeted prefiltering.
            When provided, uses reference-targeted prefilter instead of generic.
        use_mtcnn: If True, use MTCNN face detection instead of skin heuristics.
            MTCNN is more accurate but requires TensorFlow. Default: True.
        reject_dir: Optional directory to copy rejected images to (for testing/debugging).
        skip_duplicates: If True, skip processing of duplicate images (by content hash).
            Duplicates are still reported via progress_callback with duplicate=True.

    Returns:
        MatchResult with total_scanned, matches_found (images that passed),
        and errors count.
    """
    result = MatchResult()
    seen_hashes: set[str] = set() if skip_duplicates else None

    # Determine image iterator based on source type
    if isinstance(image_source, Path):
        image_iter = iter_images(image_source, recursive)
    else:
        image_iter = image_source

    for image_path in image_iter:
        # Check for duplicates if enabled
        if seen_hashes is not None:
            file_hash = compute_file_hash(image_path)
            if file_hash in seen_hashes:
                result.total_scanned += 1
                if progress_callback:
                    progress_callback(image_path, passed=False, error=False, error_msg=None, prefilter=None, duplicate=True)
                continue
            seen_hashes.add(file_hash)
        result.total_scanned += 1

        try:
            image = load_image_with_exif_rotation(image_path)

            # Validate image shape (must be 3-channel RGB)
            if len(image.shape) != 3 or image.shape[2] != 3:
                if reject_dir is not None:
                    copy_image(image_path, reject_dir)
                if progress_callback:
                    progress_callback(image_path, passed=False, error=False, error_msg=None, prefilter=None, duplicate=False)
                continue

            # Ensure correct dtype
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Resize for consistent prefilter behavior
            image = resize_for_cnn(image)

            # Run prefilter
            prefilter = should_skip_cnn_targeted(image, skin_info, use_mtcnn=use_mtcnn)

            if not prefilter.should_skip:
                # Image passes prefilter - copy it
                copy_image(image_path, output_dir)
                result.matches_found += 1

                if progress_callback:
                    progress_callback(image_path, passed=True, error=False, error_msg=None, prefilter=prefilter, duplicate=False)

                # Check limit
                if limit is not None and result.matches_found >= limit:
                    break
            else:
                # Image rejected by prefilter
                if reject_dir is not None:
                    copy_image(image_path, reject_dir)
                if progress_callback:
                    progress_callback(image_path, passed=False, error=False, error_msg=None, prefilter=prefilter, duplicate=False)

        except Exception as e:
            result.errors += 1
            if progress_callback:
                progress_callback(image_path, passed=False, error=True, error_msg=str(e), prefilter=None, duplicate=False)

    return result


def process_images(
    image_source: Path | Iterator[Path],
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
    use_mtcnn: bool = True,
    reject_dir: Path | None = None,
    skip_duplicates: bool = False,
) -> MatchResult:
    """
    Main processing loop: iterate images, check matches, copy immediately.

    Uses streaming/generator pattern for memory efficiency.
    Prefiltering is done here, before calling match functions.

    Args:
        image_source: Directory to search for images, or an iterator of image paths.
            When a directory (Path), uses iter_images() with recursive flag.
            When an iterator, uses it directly (recursive flag is ignored).
        output_dir: Directory to copy matching images to.
        reference_encoding: Face encoding to match against.
        tolerance: Distance threshold for matching.
        recursive: Whether to search subdirectories (only used when image_source is a directory).
        limit: Maximum number of matches to find.
        progress_callback: Called after each image is processed.
        pre_process_callback: Called BEFORE each image is processed. Useful for
            logging which image is being processed in case of crashes.
            Signature: pre_process_callback(image_path: Path, model: str)
        model: Face detection model - "hog" (fast) or "cnn" (better for profiles)
        verbose: If True, progress_callback receives additional match_info parameter
            with MatchInfo containing distance values.
        prefilter_callback: Called immediately after prefilter runs (CNN/auto only).
            Fires BEFORE the slow CNN detection. Useful for showing prefilter results
            in real-time. Signature: prefilter_callback(image_path: Path, prefilter: PrefilterResult)
        skin_info: Optional dict with reference skin color for targeted prefiltering.
            When provided, uses reference-targeted prefilter instead of generic.
        use_mtcnn: If True, use MTCNN face detection in prefilter instead of skin heuristics.
            MTCNN is more accurate but requires TensorFlow. Default: True.
        reject_dir: Optional directory to copy rejected images to (for testing/debugging).
            Images are copied when skipped by prefilter or when they don't match.
        skip_duplicates: If True, skip processing of duplicate images (by content hash).
            Duplicates are still reported via progress_callback with duplicate=True.
    """
    result = MatchResult()
    seen_hashes: set[str] = set() if skip_duplicates else None

    # Determine image iterator based on source type
    if isinstance(image_source, Path):
        image_iter = iter_images(image_source, recursive)
    else:
        image_iter = image_source

    for image_path in image_iter:
        result.total_scanned += 1

        # Check for duplicates if enabled (before pre_process_callback)
        if seen_hashes is not None:
            file_hash = compute_file_hash(image_path)
            if file_hash in seen_hashes:
                if progress_callback:
                    if verbose:
                        progress_callback(image_path, matched=False, match_info=None, duplicate=True)
                    else:
                        progress_callback(image_path, matched=False, duplicate=True)
                continue
            seen_hashes.add(file_hash)

        # Log BEFORE processing (critical for crash diagnosis)
        if pre_process_callback:
            pre_process_callback(image_path, model)

        try:
            # Run prefilter for CNN/auto models BEFORE calling match functions
            prefilter_result: PrefilterResult | None = None
            if model in ("cnn", "auto"):
                image = load_image_with_exif_rotation(image_path)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image.dtype != np.uint8:
                        image = image.astype(np.uint8)
                    image = resize_for_cnn(image)
                    prefilter_result = should_skip_cnn_targeted(image, skin_info, use_mtcnn=use_mtcnn)

                    # Fire prefilter callback if provided
                    if prefilter_callback:
                        prefilter_callback(image_path, prefilter_result)

                    # If prefilter says skip, handle rejection and continue
                    if prefilter_result.should_skip:
                        if reject_dir is not None:
                            copy_image(image_path, reject_dir)
                        if progress_callback:
                            if verbose:
                                # Create MatchInfo with prefilter details
                                match_info = MatchInfo(
                                    matched=False,
                                    min_distance=None,
                                    num_faces=0,
                                    prefilter_skipped=True,
                                    prefilter_reason=prefilter_result.reason,
                                    prefilter_entropy=prefilter_result.entropy,
                                    prefilter_skin_pct=prefilter_result.skin_percentage,
                                    prefilter_ref_skin_pct=prefilter_result.reference_skin_percentage,
                                    used_targeted_filter=prefilter_result.used_targeted_filter,
                                    prefilter_mtcnn_faces=prefilter_result.mtcnn_faces,
                                )
                                progress_callback(image_path, matched=False, match_info=match_info, duplicate=False)
                            else:
                                progress_callback(image_path, matched=False, duplicate=False)
                        continue

            # Now call match function (prefilter already passed or not applicable)
            if verbose:
                match_info = check_match(
                    image_path, reference_encoding, tolerance, model=model
                )
                # Add prefilter info to match_info if we ran prefilter
                if prefilter_result is not None:
                    match_info.prefilter_entropy = prefilter_result.entropy
                    match_info.prefilter_skin_pct = prefilter_result.skin_percentage
                    match_info.prefilter_ref_skin_pct = prefilter_result.reference_skin_percentage
                    match_info.used_targeted_filter = prefilter_result.used_targeted_filter
                    match_info.prefilter_mtcnn_faces = prefilter_result.mtcnn_faces

                if match_info.matched:
                    copied_path = copy_image(image_path, output_dir)
                    result.matches_found += 1
                    result.matched_files.append(copied_path)

                    if progress_callback:
                        progress_callback(image_path, matched=True, match_info=match_info, duplicate=False)

                    # Check if we've reached the limit
                    if limit is not None and result.matches_found >= limit:
                        break
                else:
                    # Image rejected - copy to reject_dir if provided
                    if reject_dir is not None:
                        copy_image(image_path, reject_dir)
                    if progress_callback:
                        progress_callback(image_path, matched=False, match_info=match_info, duplicate=False)
            else:
                # Use is_match for simple bool result (faster, less memory)
                if is_match(image_path, reference_encoding, tolerance, model=model):
                    copied_path = copy_image(image_path, output_dir)
                    result.matches_found += 1
                    result.matched_files.append(copied_path)

                    if progress_callback:
                        progress_callback(image_path, matched=True, duplicate=False)

                    # Check if we've reached the limit
                    if limit is not None and result.matches_found >= limit:
                        break
                else:
                    # Image rejected - copy to reject_dir if provided
                    if reject_dir is not None:
                        copy_image(image_path, reject_dir)
                    if progress_callback:
                        progress_callback(image_path, matched=False, duplicate=False)
        except Exception as e:
            result.errors += 1
            if progress_callback:
                progress_callback(image_path, matched=False, error=True, error_msg=str(e), duplicate=False)

    return result
