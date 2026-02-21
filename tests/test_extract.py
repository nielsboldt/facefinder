"""Tests for face extraction functionality."""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from face_finder.detector import (
    crop_face,
    extract_faces_from_image,
    extract_all_faces,
)


class TestCropFace:
    """Test the crop_face function."""

    def test_crop_face_basic(self):
        """Basic crop with no padding."""
        # Create 100x100 test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:60, 30:70] = 255  # White box in middle

        # Face location: (top, right, bottom, left)
        face_location = (25, 65, 55, 35)
        result = crop_face(image, face_location, padding=0.0)

        # Should match exact bounds
        assert result.shape[0] == 30  # height: 55 - 25
        assert result.shape[1] == 30  # width: 65 - 35

    def test_crop_face_with_padding(self):
        """Crop with 30% padding expands the region."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_location = (30, 60, 60, 30)  # 30x30 face

        result = crop_face(image, face_location, padding=0.3)

        # With 30% padding on 30x30 face: 9px padding each side
        # Expected: ~48x48 (depending on rounding)
        assert result.shape[0] > 30
        assert result.shape[1] > 30

    def test_crop_face_clamps_to_bounds(self):
        """Crop should clamp to image boundaries."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Face near edge
        face_location = (5, 95, 25, 80)

        result = crop_face(image, face_location, padding=0.5)

        # Should not exceed image bounds
        assert result.shape[0] <= 100
        assert result.shape[1] <= 100
        # Top and left should clamp to 0
        # Right should clamp to 100

    def test_crop_face_preserves_pixel_values(self):
        """Cropped region should have correct pixel values."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Mark the face region with distinct color
        image[40:60, 40:60] = [255, 128, 64]

        face_location = (40, 60, 60, 40)
        result = crop_face(image, face_location, padding=0.0)

        # All pixels should be the marked color
        assert np.all(result == [255, 128, 64])


class TestExtractFacesFromImage:
    """Test extract_faces_from_image function."""

    @pytest.fixture
    def test_image(self):
        """Get a test image from the project."""
        image_path = Path(__file__).parent.parent / "face.jpg"
        if not image_path.exists():
            pytest.skip("face.jpg not found")
        return image_path

    def test_extract_creates_output_files(self, test_image, tmp_path):
        """Extract should create .jpg and .npy files for each face."""
        count = extract_faces_from_image(test_image, tmp_path, model="hog")

        assert count >= 1, "Should find at least one face in test image"

        # Check files were created
        jpg_files = list(tmp_path.glob("*.jpg"))
        npy_files = list(tmp_path.glob("*.npy"))

        assert len(jpg_files) == count
        assert len(npy_files) == count

    def test_extract_file_naming(self, test_image, tmp_path):
        """Files should be named {source_stem}_face_{N}.ext."""
        extract_faces_from_image(test_image, tmp_path, model="hog")

        expected_stem = test_image.stem
        files = list(tmp_path.iterdir())

        for f in files:
            assert f.name.startswith(expected_stem)
            assert "_face_" in f.name

    def test_extract_encoding_format(self, test_image, tmp_path):
        """Encodings should be 128-dimensional numpy arrays."""
        extract_faces_from_image(test_image, tmp_path, model="hog")

        npy_files = list(tmp_path.glob("*.npy"))
        assert len(npy_files) >= 1

        for npy_file in npy_files:
            encoding = np.load(npy_file, allow_pickle=True)
            assert encoding.shape == (128,), f"Expected 128-dim encoding, got {encoding.shape}"

    def test_extract_thumbnail_is_valid_image(self, test_image, tmp_path):
        """Thumbnails should be valid JPEG images."""
        extract_faces_from_image(test_image, tmp_path, model="hog")

        jpg_files = list(tmp_path.glob("*.jpg"))
        assert len(jpg_files) >= 1

        for jpg_file in jpg_files:
            img = Image.open(jpg_file)
            assert img.mode == "RGB"
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_extract_no_faces_returns_zero(self, tmp_path):
        """Image with no faces should return 0 and create no files."""
        # Create solid color image (no faces)
        no_face_image = tmp_path / "solid.jpg"
        img = Image.new('RGB', (200, 200), color=(100, 100, 100))
        img.save(no_face_image)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = extract_faces_from_image(no_face_image, output_dir, model="hog")

        assert count == 0
        assert len(list(output_dir.iterdir())) == 0

    def test_extract_padding_affects_crop_size(self, test_image, tmp_path):
        """Different padding values should produce different crop sizes."""
        output_small = tmp_path / "small"
        output_large = tmp_path / "large"
        output_small.mkdir()
        output_large.mkdir()

        extract_faces_from_image(test_image, output_small, padding=0.1)
        extract_faces_from_image(test_image, output_large, padding=0.5)

        small_jpg = list(output_small.glob("*.jpg"))[0]
        large_jpg = list(output_large.glob("*.jpg"))[0]

        small_img = Image.open(small_jpg)
        large_img = Image.open(large_jpg)

        # Larger padding should produce larger image
        small_area = small_img.size[0] * small_img.size[1]
        large_area = large_img.size[0] * large_img.size[1]
        assert large_area > small_area

    def test_extract_multiple_faces(self, tmp_path):
        """Image with multiple faces should extract all of them."""
        # Use reference image that has multiple faces
        multi_face_image = Path("/mnt/c/temp/test-data-face-finder/reference/2012-08-12 00.02.51.jpg")
        if not multi_face_image.exists():
            pytest.skip("Multi-face test image not found")

        count = extract_faces_from_image(multi_face_image, tmp_path, model="hog")

        assert count >= 2, "Should find multiple faces"

        # Verify sequential numbering
        for i in range(1, count + 1):
            jpg_path = tmp_path / f"{multi_face_image.stem}_face_{i:03d}.jpg"
            npy_path = tmp_path / f"{multi_face_image.stem}_face_{i:03d}.npy"
            assert jpg_path.exists(), f"Missing {jpg_path}"
            assert npy_path.exists(), f"Missing {npy_path}"


class TestExtractWithModels:
    """Test extraction with different face detection models."""

    @pytest.fixture
    def test_image(self):
        """Get a test image from the project."""
        image_path = Path(__file__).parent.parent / "face.jpg"
        if not image_path.exists():
            pytest.skip("face.jpg not found")
        return image_path

    def test_hog_model(self, test_image, tmp_path):
        """HOG model should work for extraction."""
        count = extract_faces_from_image(test_image, tmp_path, model="hog")
        assert count >= 0

    def test_cnn_model(self, test_image, tmp_path):
        """CNN model should work for extraction."""
        count = extract_faces_from_image(test_image, tmp_path, model="cnn")
        assert count >= 0

    def test_auto_model(self, test_image, tmp_path):
        """Auto model should work for extraction."""
        count = extract_faces_from_image(test_image, tmp_path, model="auto")
        assert count >= 0
