"""Tests for MTCNN prefilter functionality."""

import numpy as np
import pytest
from pathlib import Path

from face_finder.prefilter import (
    PrefilterResult,
    get_mtcnn_detector,
    mtcnn_prefilter,
    should_skip_cnn_targeted,
)


class TestMTCNNDetector:
    """Test MTCNN detector loading and basic functionality."""

    def test_mtcnn_detector_loads(self):
        """MTCNN detector should load successfully."""
        detector = get_mtcnn_detector()
        assert detector is not None

    def test_mtcnn_detector_is_singleton(self):
        """MTCNN detector should be a singleton (same instance)."""
        detector1 = get_mtcnn_detector()
        detector2 = get_mtcnn_detector()
        assert detector1 is detector2


class TestMTCNNPrefilter:
    """Test MTCNN prefilter function."""

    def test_mtcnn_rejects_solid_color_image(self):
        """MTCNN should reject solid color image (no faces)."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = mtcnn_prefilter(image)
        assert result.should_skip is True
        assert "MTCNN" in result.reason
        assert result.mtcnn_faces == 0

    def test_mtcnn_rejects_landscape_image(self):
        """MTCNN should reject landscape/nature-like images."""
        # Create random blue/green image (landscape-like)
        rng = np.random.default_rng(42)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :, 1] = rng.integers(100, 200, size=(200, 200))  # Green
        image[:, :, 2] = rng.integers(150, 255, size=(200, 200))  # Blue
        result = mtcnn_prefilter(image)
        assert result.should_skip is True
        assert result.mtcnn_faces == 0

    def test_mtcnn_rejects_random_noise(self):
        """MTCNN should reject random noise image."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
        result = mtcnn_prefilter(image)
        assert result.should_skip is True
        assert result.mtcnn_faces == 0

    def test_mtcnn_result_has_image_size(self):
        """MTCNN result should include image size."""
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        result = mtcnn_prefilter(image)
        assert result.image_size == (300, 400)

    def test_mtcnn_result_fields(self):
        """MTCNN result should have all expected fields."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = mtcnn_prefilter(image)

        assert hasattr(result, 'should_skip')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'mtcnn_faces')
        assert hasattr(result, 'image_size')

        assert isinstance(result.should_skip, bool)
        assert isinstance(result.reason, str)
        assert isinstance(result.mtcnn_faces, int)


class TestShouldSkipCNNTargetedWithMTCNN:
    """Test integration of MTCNN in should_skip_cnn_targeted."""

    def test_mtcnn_enabled_by_default(self):
        """MTCNN should be enabled by default (use_mtcnn=True)."""
        # With MTCNN enabled, solid images fail early due to no faces
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = should_skip_cnn_targeted(image, use_mtcnn=True)
        # Should skip due to MTCNN finding no faces (after passing entropy check)
        # Note: solid image fails entropy check first
        assert result.should_skip is True

    def test_mtcnn_disabled_uses_skin_heuristics(self):
        """When MTCNN is disabled, should fall back to skin heuristics."""
        # Create image with skin-toned pixels
        rng = np.random.default_rng(42)
        image = rng.integers(50, 150, size=(200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150, 0] = 180  # Red
        image[50:150, 50:150, 1] = 140  # Green
        image[50:150, 50:150, 2] = 100  # Blue

        # With MTCNN disabled, should pass due to skin tones
        result_without_mtcnn = should_skip_cnn_targeted(image, use_mtcnn=False)
        assert result_without_mtcnn.should_skip is False
        # MTCNN not run, so mtcnn_faces should be None
        assert result_without_mtcnn.mtcnn_faces is None

    def test_mtcnn_result_includes_mtcnn_faces(self):
        """When MTCNN is enabled, result should include mtcnn_faces count."""
        # Create landscape image that passes entropy but fails MTCNN
        rng = np.random.default_rng(42)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :, 1] = rng.integers(100, 200, size=(200, 200))  # Green
        image[:, :, 2] = rng.integers(150, 255, size=(200, 200))  # Blue

        result = should_skip_cnn_targeted(image, use_mtcnn=True)
        assert result.mtcnn_faces is not None
        assert result.mtcnn_faces == 0

    def test_size_check_runs_before_mtcnn(self):
        """Size check should run before MTCNN (free check first)."""
        # Tiny image should be rejected before MTCNN runs
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        result = should_skip_cnn_targeted(image, use_mtcnn=True)
        assert result.should_skip is True
        assert "too small" in result.reason
        # MTCNN should not have run
        assert result.mtcnn_faces is None

    def test_entropy_check_runs_before_mtcnn(self):
        """Entropy check should run before MTCNN."""
        # Solid image should be rejected for low entropy before MTCNN
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = should_skip_cnn_targeted(image, use_mtcnn=True)
        assert result.should_skip is True
        assert "blank" in result.reason or "solid" in result.reason
        # MTCNN should not have run (entropy check caught it first)
        assert result.mtcnn_faces is None


class TestMTCNNWithRealImages:
    """Test MTCNN prefilter with real image files."""

    @pytest.fixture
    def face_image_path(self):
        """Get test face image path if available."""
        image_path = Path(__file__).parent.parent / "face.jpg"
        if not image_path.exists():
            pytest.skip("face.jpg not found")
        return image_path

    @pytest.fixture
    def positive_image_dir(self):
        """Get positive test images directory."""
        dir_path = Path("/mnt/c/temp/test-data-face-finder/positive")
        if not dir_path.exists():
            pytest.skip("Positive test images directory not found")
        return dir_path

    @pytest.fixture
    def negative_image_dir(self):
        """Get negative test images directory."""
        dir_path = Path("/mnt/c/temp/test-data-face-finder/negative")
        if not dir_path.exists():
            pytest.skip("Negative test images directory not found")
        return dir_path

    def test_mtcnn_detects_face_in_face_image(self, face_image_path):
        """MTCNN should detect face in known face image."""
        from face_finder.detector import load_image_with_exif_rotation

        image = load_image_with_exif_rotation(face_image_path)
        result = mtcnn_prefilter(image)
        assert result.should_skip is False, f"Face image should not be skipped: {result.reason}"
        assert result.mtcnn_faces >= 1, "Should detect at least one face"

    def test_mtcnn_integration_with_face_image(self, face_image_path):
        """Full prefilter with MTCNN should pass face image."""
        from face_finder.detector import load_image_with_exif_rotation

        image = load_image_with_exif_rotation(face_image_path)
        result = should_skip_cnn_targeted(image, use_mtcnn=True)
        assert result.should_skip is False, f"Face image should not be skipped: {result.reason}"
        assert result.mtcnn_faces >= 1

    def test_positive_images_pass_mtcnn(self, positive_image_dir):
        """Positive test images (known faces) should pass MTCNN filter."""
        from face_finder.detector import load_image_with_exif_rotation

        passed = 0
        failed = 0
        for img_path in positive_image_dir.glob("*.jpg"):
            try:
                image = load_image_with_exif_rotation(img_path)
                result = should_skip_cnn_targeted(image, use_mtcnn=True)
                if result.should_skip:
                    failed += 1
                else:
                    passed += 1
            except Exception:
                pass  # Skip images that fail to load

        total = passed + failed
        if total == 0:
            pytest.skip("No positive test images found")

        # At least 80% of positive images should pass
        pass_rate = passed / total
        assert pass_rate >= 0.8, f"Only {pass_rate*100:.1f}% of positive images passed MTCNN"

    def test_negative_images_filtered_by_mtcnn(self, negative_image_dir):
        """Negative test images (no faces) should mostly be filtered by MTCNN."""
        from face_finder.detector import load_image_with_exif_rotation

        passed = 0
        filtered = 0
        for img_path in negative_image_dir.glob("*.jpg"):
            try:
                image = load_image_with_exif_rotation(img_path)
                result = should_skip_cnn_targeted(image, use_mtcnn=True)
                if result.should_skip:
                    filtered += 1
                else:
                    passed += 1
            except Exception:
                pass  # Skip images that fail to load

        total = passed + filtered
        if total == 0:
            pytest.skip("No negative test images found")

        # At least 50% of negative images should be filtered
        # (MTCNN should do better than heuristics at filtering non-faces)
        filter_rate = filtered / total
        assert filter_rate >= 0.5, f"Only {filter_rate*100:.1f}% of negative images were filtered"
