"""Face detection and encoding module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import face_recognition
import numpy as np
from numpy.typing import NDArray
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Maximum image dimension for CNN model to prevent memory issues/crashes
# Images larger than this will be resized before CNN processing
CNN_MAX_DIMENSION = 1800


@dataclass
class FaceData:
    """A face encoding with its source image and location."""

    encoding: NDArray[np.float64]
    source_image: Path
    location: tuple[int, int, int, int] | None = None  # (top, right, bottom, left)
    image: NDArray[np.uint8] | None = None  # The image array for skin extraction


@dataclass
class PrefilterInfo:
    """Prefilter results for a single image."""

    entropy: float | None = None
    skin_percentage: float | None = None


@dataclass
class AnalysisResult:
    """Detailed results from reference image analysis."""

    encoding: NDArray[np.float64]
    images_with_face: list[Path]
    images_without_face: list[Path]
    total_images: int
    total_faces_detected: int
    faces_per_image: dict[Path, int] = field(default_factory=dict)
    distances_to_common: dict[Path, float] = field(default_factory=dict)  # Distance from each image's closest face to common encoding
    prefilter_info: dict[Path, PrefilterInfo] = field(default_factory=dict)  # Prefilter scores per image
    skin_color: dict | None = None  # Extracted skin color: {"median_rgb", "std_rgb", "sample_count", "image_count"}

    @property
    def match_count(self) -> int:
        """Number of images containing the common face."""
        return len(self.images_with_face)


def iter_reference_images(directory: Path) -> Iterator[Path]:
    """Iterate over image files in a directory."""
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image_with_exif_rotation(image_path: Path) -> NDArray[np.uint8]:
    """
    Load an image and apply EXIF orientation correction.

    Many phone cameras store images in landscape orientation with an EXIF tag
    indicating how to display them. This function reads that tag and rotates
    the image accordingly so face detection works correctly.
    """
    from PIL import ExifTags

    pil_image = Image.open(image_path)

    # Try to get EXIF orientation
    try:
        exif = pil_image._getexif()
        if exif is not None:
            # Find the orientation tag
            orientation_key = None
            for key, val in ExifTags.TAGS.items():
                if val == "Orientation":
                    orientation_key = key
                    break

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]

                # Apply rotation based on EXIF orientation value
                if orientation == 2:
                    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 4:
                    pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    pil_image = pil_image.rotate(-90, expand=True).transpose(
                        Image.FLIP_LEFT_RIGHT
                    )
                elif orientation == 6:
                    pil_image = pil_image.rotate(-90, expand=True)
                elif orientation == 7:
                    pil_image = pil_image.rotate(90, expand=True).transpose(
                        Image.FLIP_LEFT_RIGHT
                    )
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF data or orientation tag
        pass

    # Convert to RGB if necessary (face_recognition expects RGB)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Ensure C-contiguous array for dlib compatibility
    # PIL color mode conversion can produce non-contiguous arrays that crash dlib's CNN
    return np.ascontiguousarray(np.array(pil_image))


def resize_for_cnn(image: NDArray[np.uint8], max_dimension: int = CNN_MAX_DIMENSION) -> NDArray[np.uint8]:
    """
    Resize image if it exceeds the maximum dimension for CNN processing.

    Large images can cause dlib's CNN face detector to hang or crash.
    This function resizes the image proportionally if either dimension
    exceeds the maximum.

    Args:
        image: NumPy array of image (H, W, 3)
        max_dimension: Maximum allowed dimension (default: CNN_MAX_DIMENSION)

    Returns:
        Resized image as C-contiguous NumPy array, or original if no resize needed.
    """
    height, width = image.shape[:2]
    max_current = max(height, width)

    if max_current <= max_dimension:
        return image

    # Calculate new dimensions maintaining aspect ratio
    scale = max_dimension / max_current
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Use PIL for high-quality resize
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.ascontiguousarray(np.array(pil_image))


def extract_skin_color_from_face(
    image: NDArray[np.uint8],
    face_location: tuple[int, int, int, int],
) -> dict | None:
    """
    Extract skin color statistics from a face region.

    Uses the center 50% of the face bounding box to avoid hair and background,
    then applies a generic skin mask to isolate skin pixels.

    Args:
        image: RGB image array (H, W, 3)
        face_location: Face bounding box as (top, right, bottom, left)

    Returns:
        Dict with "median_rgb", "std_rgb", "sample_count" or None if < 100 skin pixels.
    """
    from .prefilter import calculate_skin_percentage

    top, right, bottom, left = face_location

    # Use center 50% of face to avoid hair/background
    height = bottom - top
    width = right - left
    margin_h = height // 4
    margin_w = width // 4

    center_top = top + margin_h
    center_bottom = bottom - margin_h
    center_left = left + margin_w
    center_right = right - margin_w

    # Clamp to image bounds
    center_top = max(0, center_top)
    center_bottom = min(image.shape[0], center_bottom)
    center_left = max(0, center_left)
    center_right = min(image.shape[1], center_right)

    if center_bottom <= center_top or center_right <= center_left:
        return None

    # Extract face center region
    face_region = image[center_top:center_bottom, center_left:center_right]

    # Apply generic skin mask to isolate skin pixels
    r, g, b = face_region[:, :, 0], face_region[:, :, 1], face_region[:, :, 2]
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b)) > 15) &
        (np.abs(r.astype(np.int16) - g.astype(np.int16)) > 15) &
        (r > g) & (r > b)
    )

    # Get skin pixels
    skin_pixels = face_region[skin_mask]

    if len(skin_pixels) < 100:
        return None

    # Calculate statistics
    median_rgb = tuple(int(x) for x in np.median(skin_pixels, axis=0))
    std_rgb = tuple(float(x) for x in np.std(skin_pixels, axis=0))

    return {
        "median_rgb": median_rgb,
        "std_rgb": std_rgb,
        "sample_count": len(skin_pixels),
    }


def extract_all_faces(
    image_path: Path,
    model: str = "hog",
) -> tuple[list[NDArray[np.float64]], list[tuple[int, int, int, int]], str, NDArray[np.uint8]]:
    """
    Extract all face encodings from an image.

    Args:
        image_path: Path to the image file
        model: Face detection model:
               - "hog" (fast, frontal faces only)
               - "cnn" (slower, better for angles/profiles)
               - "auto" (try hog first, fall back to cnn if no faces found)

    Returns:
        Tuple of (list of encodings, face_locations, model_used, image).
        Face locations are (top, right, bottom, left) tuples.
        Handles EXIF orientation to properly detect faces in rotated images.
    """
    image = load_image_with_exif_rotation(image_path)

    # Resize large images for CNN to prevent hangs/crashes
    if model in ("cnn", "auto"):
        image = resize_for_cnn(image)

    if model == "auto":
        # Try HOG first (fast)
        face_locations = face_recognition.face_locations(image, model="hog")
        if face_locations:
            return face_recognition.face_encodings(image, face_locations), face_locations, "hog", image
        # Fall back to CNN if no faces found
        face_locations = face_recognition.face_locations(image, model="cnn")
        return face_recognition.face_encodings(image, face_locations), face_locations, "cnn", image
    else:
        face_locations = face_recognition.face_locations(image, model=model)
        return face_recognition.face_encodings(image, face_locations), face_locations, model, image


def get_average_encoding(encodings: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """
    Compute the average of multiple face encodings.

    Averaging multiple reference encodings improves matching accuracy.
    """
    return np.mean(encodings, axis=0)


def find_common_face(
    all_faces: list[FaceData],
    tolerance: float = 0.6,
) -> tuple[NDArray[np.float64], list[Path], list[FaceData]] | None:
    """
    Find the face that appears across the most reference images.

    Args:
        all_faces: List of FaceData objects with encodings, source images, locations, and images
        tolerance: Distance threshold for considering faces as the same person

    Returns:
        Tuple of (averaged encoding, list of matching image paths, list of matching FaceData) for the most common face,
        or None if no faces found.
    """
    if not all_faces:
        return None

    # For each face, count how many unique images contain a matching face
    best_matching_faces: list[FaceData] = []
    best_matching_images: set[Path] = set()

    for candidate in all_faces:
        # Find all faces that match this candidate
        matching_faces: list[FaceData] = []
        matching_images: set[Path] = set()

        for face in all_faces:
            distance = face_recognition.face_distance(
                [candidate.encoding], face.encoding
            )[0]
            if distance <= tolerance:
                matching_faces.append(face)
                matching_images.add(face.source_image)

        # Track the face that appears in the most images
        if len(matching_images) > len(best_matching_images):
            best_matching_images = matching_images
            best_matching_faces = matching_faces

    if not best_matching_faces:
        return None

    # Average the encodings of all matching faces
    averaged_encoding = get_average_encoding(
        [f.encoding for f in best_matching_faces]
    )

    return averaged_encoding, list(best_matching_images), best_matching_faces


def analyze_reference_images(
    directory: Path,
    tolerance: float = 0.6,
    model: str = "hog",
    progress_callback: callable = None,
) -> AnalysisResult | None:
    """
    Analyze reference images and find the common face with detailed results.

    Args:
        directory: Directory containing reference images
        tolerance: Distance threshold for face matching
        model: Face detection model - "hog" (fast) or "cnn" (better for profiles)
        progress_callback: Optional callback(image_path, faces_found, model_used, prefilter_info)
                          called after each image. prefilter_info is a PrefilterInfo object
                          (only populated when using CNN/auto mode).

    Returns:
        AnalysisResult with detailed information, or None if no common face found.
    """
    from .prefilter import should_skip_cnn

    all_faces: list[FaceData] = []
    faces_by_image: dict[Path, list[NDArray[np.float64]]] = {}
    all_images: list[Path] = []
    total_faces = 0
    prefilter_info_map: dict[Path, PrefilterInfo] = {}

    for image_path in iter_reference_images(directory):
        all_images.append(image_path)

        # Run prefilter for CNN/auto mode to get entropy/skin scores
        pf_info: PrefilterInfo | None = None
        if model in ("cnn", "auto"):
            image = load_image_with_exif_rotation(image_path)
            if model in ("cnn", "auto"):
                image = resize_for_cnn(image)
            prefilter = should_skip_cnn(image)
            pf_info = PrefilterInfo(
                entropy=prefilter.entropy,
                skin_percentage=prefilter.skin_percentage,
            )
            prefilter_info_map[image_path] = pf_info

        encodings, locations, model_used, image = extract_all_faces(image_path, model=model)
        if encodings:
            faces_by_image[image_path] = encodings
            total_faces += len(encodings)
            # Store face data with locations and images for skin extraction
            for enc, loc in zip(encodings, locations):
                all_faces.append(FaceData(
                    encoding=enc,
                    source_image=image_path,
                    location=loc,
                    image=image,
                ))
        if progress_callback:
            progress_callback(image_path, len(encodings) if encodings else 0, model_used, pf_info)

    if not all_faces:
        return None

    result = find_common_face(all_faces, tolerance)
    if result is None:
        return None

    encoding, matching_images, matching_faces = result
    matching_set = set(matching_images)

    # Compute distance from each image's closest face to the common encoding
    distances_to_common: dict[Path, float] = {}
    for image_path, face_encodings in faces_by_image.items():
        if face_encodings:
            distances = face_recognition.face_distance(face_encodings, encoding)
            distances_to_common[image_path] = float(np.min(distances))

    # Extract skin color from matching faces
    skin_color = _aggregate_skin_colors(matching_faces)

    return AnalysisResult(
        encoding=encoding,
        images_with_face=matching_images,
        images_without_face=[p for p in all_images if p not in matching_set],
        total_images=len(all_images),
        total_faces_detected=total_faces,
        faces_per_image={p: len(faces_by_image.get(p, [])) for p in all_images},
        distances_to_common=distances_to_common,
        prefilter_info=prefilter_info_map,
        skin_color=skin_color,
    )


def _aggregate_skin_colors(matching_faces: list[FaceData]) -> dict | None:
    """
    Aggregate skin color from multiple matching faces.

    Computes weighted average of median RGB values and takes max std dev for tolerance.

    Args:
        matching_faces: List of FaceData with matching faces

    Returns:
        Dict with "median_rgb", "std_rgb", "sample_count", "image_count" or None if insufficient data.
    """
    skin_samples: list[dict] = []

    for face in matching_faces:
        if face.image is not None and face.location is not None:
            skin_info = extract_skin_color_from_face(face.image, face.location)
            if skin_info is not None:
                skin_samples.append(skin_info)

    if not skin_samples:
        return None

    # Weight by sample count for averaging median RGB
    total_samples = sum(s["sample_count"] for s in skin_samples)
    weighted_r = sum(s["median_rgb"][0] * s["sample_count"] for s in skin_samples) / total_samples
    weighted_g = sum(s["median_rgb"][1] * s["sample_count"] for s in skin_samples) / total_samples
    weighted_b = sum(s["median_rgb"][2] * s["sample_count"] for s in skin_samples) / total_samples

    # Take max std dev for tolerance (most permissive)
    max_std_r = max(s["std_rgb"][0] for s in skin_samples)
    max_std_g = max(s["std_rgb"][1] for s in skin_samples)
    max_std_b = max(s["std_rgb"][2] for s in skin_samples)

    return {
        "median_rgb": (int(round(weighted_r)), int(round(weighted_g)), int(round(weighted_b))),
        "std_rgb": (round(max_std_r, 1), round(max_std_g, 1), round(max_std_b, 1)),
        "sample_count": total_samples,
        "image_count": len(skin_samples),
    }


def load_reference_encodings(
    directory: Path,
    tolerance: float = 0.6,
    model: str = "hog",
) -> tuple[NDArray[np.float64], int, int] | None:
    """
    Load reference images and find the common face across all of them.

    Analyzes all faces in all reference images and identifies the person
    who appears in the most images. This handles cases where reference
    photos contain multiple people.

    Args:
        directory: Directory containing reference images
        tolerance: Distance threshold for face matching
        model: Face detection model - "hog" (fast) or "cnn" (better for profiles)

    Returns:
        Tuple of (encoding, images_with_face, total_images) or None if no common face found.
    """
    result = analyze_reference_images(directory, tolerance, model=model)
    if result is None:
        return None
    return result.encoding, result.match_count, result.total_images


def save_encoding(
    encoding: NDArray[np.float64],
    output_path: Path,
    skin_color: dict | None = None,
) -> None:
    """
    Save a face encoding to a .npy file.

    If skin_color is provided, saves in v2 format with versioning.
    Otherwise saves in v1 format (raw array) for backward compatibility.
    """
    if skin_color:
        data = {"version": 2, "encoding": encoding, "skin_color": skin_color}
        np.save(output_path, data, allow_pickle=True)
    else:
        np.save(output_path, encoding)


def load_encoding(input_path: Path) -> tuple[NDArray[np.float64], dict | None]:
    """
    Load a face encoding from a .npy file.

    Returns:
        Tuple of (encoding, skin_color).
        skin_color is None for v1 (legacy) format files.
    """
    data = np.load(input_path, allow_pickle=True)

    # v1: legacy format - raw 128-dim array
    if isinstance(data, np.ndarray) and data.shape == (128,):
        return data, None

    # v2: dict format with version, encoding, skin_color
    data_dict = data.item()
    return data_dict["encoding"], data_dict.get("skin_color")


def crop_face(
    image: NDArray[np.uint8],
    face_location: tuple[int, int, int, int],
    padding: float = 0.3,
) -> NDArray[np.uint8]:
    """
    Crop a face region from an image with padding.

    Args:
        image: RGB image as numpy array
        face_location: (top, right, bottom, left) face bounding box
        padding: Extra padding as fraction of face size (0.3 = 30%)

    Returns:
        Cropped face region as numpy array
    """
    top, right, bottom, left = face_location
    height = bottom - top
    width = right - left

    # Add padding
    pad_h = int(height * padding)
    pad_w = int(width * padding)

    # Clamp to image bounds
    top = max(0, top - pad_h)
    bottom = min(image.shape[0], bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(image.shape[1], right + pad_w)

    return image[top:bottom, left:right]


def extract_faces_from_image(
    image_path: Path,
    output_dir: Path,
    model: str = "hog",
    padding: float = 0.3,
) -> int:
    """
    Extract all faces from an image and save thumbnails + encodings.

    Files are named: {source_stem}_face_{N}.jpg/npy
    Face crops are taken from the original resolution image.

    Args:
        image_path: Path to input image
        output_dir: Directory to save face files
        model: Face detection model ("hog", "cnn", "auto")
        padding: Padding around face crop (0.3 = 30%)

    Returns:
        Number of faces extracted
    """
    # Load original image for cropping at full resolution
    original_image = load_image_with_exif_rotation(image_path)

    # Run face detection (may use resized image for CNN)
    encodings, locations, _, detection_image = extract_all_faces(image_path, model)

    # Calculate scale factor if detection was on resized image
    scale = original_image.shape[0] / detection_image.shape[0]

    source_stem = image_path.stem  # filename without extension

    for i, (encoding, location) in enumerate(zip(encodings, locations), start=1):
        # Scale face location back to original image coordinates
        if scale != 1.0:
            top, right, bottom, left = location
            location = (
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale),
            )

        # Crop face from original image
        face_img = crop_face(original_image, location, padding)

        # Save thumbnail: {source}_face_001.jpg
        thumbnail_path = output_dir / f"{source_stem}_face_{i:03d}.jpg"
        Image.fromarray(face_img).save(thumbnail_path, quality=95)

        # Save encoding: {source}_face_001.npy
        encoding_path = output_dir / f"{source_stem}_face_{i:03d}.npy"
        save_encoding(encoding, encoding_path)

    return len(encodings)


def extract_face_thumbnail(
    image_path: Path,
    encoding: NDArray[np.float64],
    output_path: Path,
    tolerance: float = 0.6,
    padding: float = 0.3,
    model: str = "hog",
) -> bool:
    """
    Extract and save a thumbnail of the face matching the given encoding.

    Args:
        image_path: Path to source image
        encoding: Face encoding to match
        output_path: Where to save the thumbnail
        tolerance: Distance threshold for matching
        padding: Extra padding around face as fraction of face size
        model: Face detection model:
               - "hog" (fast, frontal faces only)
               - "cnn" (slower, better for angles/profiles)
               - "auto" (try hog first, fall back to cnn if no faces found)

    Returns:
        True if face was found and saved, False otherwise.
    """
    image = load_image_with_exif_rotation(image_path)

    # Resize large images for CNN to prevent hangs/crashes
    if model in ("cnn", "auto"):
        image = resize_for_cnn(image)

    if model == "auto":
        face_locations = face_recognition.face_locations(image, model="hog")
        if not face_locations:
            face_locations = face_recognition.face_locations(image, model="cnn")
    else:
        face_locations = face_recognition.face_locations(image, model=model)

    face_encodings = face_recognition.face_encodings(image, face_locations)

    for i, face_enc in enumerate(face_encodings):
        distance = face_recognition.face_distance([encoding], face_enc)[0]
        if distance <= tolerance:
            top, right, bottom, left = face_locations[i]

            # Add padding
            height = bottom - top
            width = right - left
            pad_h = int(height * padding)
            pad_w = int(width * padding)

            # Expand bounds with padding, clamp to image size
            top = max(0, top - pad_h)
            bottom = min(image.shape[0], bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(image.shape[1], right + pad_w)

            # Extract and save
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(output_path)
            return True

    return False
