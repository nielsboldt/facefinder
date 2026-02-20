"""Tests for the prefilter module."""

import numpy as np
import pytest

from face_finder.prefilter import (
    PrefilterResult,
    calculate_entropy,
    calculate_skin_percentage,
    should_skip_cnn,
)


class TestCalculateEntropy:
    """Test entropy calculation."""

    def test_solid_black_image_low_entropy(self):
        """Solid black image should have very low entropy."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        entropy = calculate_entropy(image)
        assert entropy < 1.0, f"Solid black should have near-zero entropy, got {entropy}"

    def test_solid_white_image_low_entropy(self):
        """Solid white image should have very low entropy."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        entropy = calculate_entropy(image)
        assert entropy < 1.0, f"Solid white should have near-zero entropy, got {entropy}"

    def test_random_noise_high_entropy(self):
        """Random noise should have high entropy."""
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        entropy = calculate_entropy(image)
        assert entropy > 7.0, f"Random noise should have high entropy, got {entropy}"

    def test_gradient_image_high_entropy(self):
        """Gradient image with 256 distinct values should have max entropy (~8 bits)."""
        # Create horizontal gradient with all 256 values
        gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        image = np.stack([gradient, gradient, gradient], axis=2)
        entropy = calculate_entropy(image)
        # 256 distinct values = log2(256) = 8.0 bits
        assert entropy >= 7.9, f"Full gradient should have ~8 bits entropy, got {entropy}"

    def test_grayscale_image(self):
        """Should handle grayscale images."""
        gray = np.random.default_rng(42).integers(0, 256, size=(100, 100), dtype=np.uint8)
        entropy = calculate_entropy(gray)
        assert entropy > 0, "Grayscale image should have positive entropy"


class TestCalculateSkinPercentage:
    """Test skin tone detection."""

    def test_no_skin_in_blue_image(self):
        """Pure blue image should have no skin tones."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # Blue channel
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct < 0.1, f"Blue image should have no skin, got {skin_pct}%"

    def test_no_skin_in_green_image(self):
        """Pure green image should have no skin tones."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 1] = 255  # Green channel
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct < 0.1, f"Green image should have no skin, got {skin_pct}%"

    def test_skin_tone_detected(self):
        """Skin-toned pixels should be detected."""
        # Create image with skin-toned pixels (R > G > B, values in skin range)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 180  # Red
        image[:, :, 1] = 140  # Green
        image[:, :, 2] = 100  # Blue
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct > 50, f"Skin-toned image should detect skin, got {skin_pct}%"

    def test_dark_skin_tone_detected(self):
        """Darker skin tones should also be detected."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 120  # Red
        image[:, :, 1] = 80   # Green
        image[:, :, 2] = 50   # Blue
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct > 50, f"Dark skin-toned image should detect skin, got {skin_pct}%"

    def test_black_image_no_skin(self):
        """Black image should have no skin tones."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct == 0, f"Black image should have 0% skin, got {skin_pct}%"

    def test_white_image_no_skin(self):
        """White image should have no skin tones (R-G difference too small)."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        skin_pct, _ = calculate_skin_percentage(image)
        assert skin_pct < 1, f"White image should have minimal skin, got {skin_pct}%"


class TestShouldSkipCnn:
    """Test the main prefilter decision function."""

    def test_skip_tiny_image(self):
        """Very small images should be skipped."""
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        result = should_skip_cnn(image)
        assert isinstance(result, PrefilterResult)
        assert result.should_skip is True
        assert "too small" in result.reason
        assert result.image_size == (30, 30)
        # Entropy/skin not computed for tiny images
        assert result.entropy is None
        assert result.skin_percentage is None

    def test_skip_narrow_image(self):
        """Images with one dimension < 60 should be skipped."""
        image = np.zeros((200, 50, 3), dtype=np.uint8)
        result = should_skip_cnn(image)
        assert result.should_skip is True
        assert "too small" in result.reason
        assert result.image_size == (200, 50)

    def test_skip_solid_color_image(self):
        """Solid color images should be skipped (low entropy)."""
        image = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = should_skip_cnn(image)
        assert result.should_skip is True
        assert "blank" in result.reason or "solid" in result.reason
        # Entropy computed but skin not (early exit)
        assert result.entropy is not None
        assert result.entropy < 3.5

    def test_skip_landscape_no_skin(self):
        """Landscape image (high entropy, no skin) should be skipped."""
        # Create a colorful landscape-like image (blues/greens)
        rng = np.random.default_rng(42)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :, 1] = rng.integers(100, 200, size=(200, 200), dtype=np.uint8)  # Green
        image[:, :, 2] = rng.integers(150, 255, size=(200, 200), dtype=np.uint8)  # Blue
        image[:, :, 0] = rng.integers(0, 100, size=(200, 200), dtype=np.uint8)    # Low red
        result = should_skip_cnn(image)
        assert result.should_skip is True
        assert "skin" in result.reason
        # Both entropy and skin computed
        assert result.entropy is not None
        assert result.entropy > 3.5  # High entropy landscape
        assert result.skin_percentage is not None
        assert result.skin_percentage < 1.0

    def test_do_not_skip_image_with_skin_tones(self):
        """Image with skin-toned areas should NOT be skipped."""
        # Create image with varied content including skin-toned region
        rng = np.random.default_rng(42)
        image = rng.integers(50, 150, size=(200, 200, 3), dtype=np.uint8)
        # Add a skin-toned region
        image[50:150, 50:150, 0] = 180  # Red
        image[50:150, 50:150, 1] = 140  # Green
        image[50:150, 50:150, 2] = 100  # Blue
        result = should_skip_cnn(image)
        assert result.should_skip is False
        assert result.reason == ""
        # Scores should be populated
        assert result.entropy is not None
        assert result.skin_percentage is not None
        assert result.skin_percentage >= 1.0

    def test_do_not_skip_photo_like_image(self):
        """Photo-like image with high entropy and skin should not be skipped."""
        # Simulate a photo with varied content
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
        # Add skin-toned area
        image[80:120, 80:120, 0] = 160
        image[80:120, 80:120, 1] = 120
        image[80:120, 80:120, 2] = 80
        result = should_skip_cnn(image)
        assert result.should_skip is False

    def test_edge_case_exactly_60px(self):
        """Image exactly 60x60 should NOT be skipped for size."""
        # Create image that passes size check but may fail others
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        image[:, :, 0] = 180
        image[:, :, 1] = 140
        image[:, :, 2] = 100
        result = should_skip_cnn(image)
        # Should not skip for size, but might skip for entropy (solid)
        assert "too small" not in result.reason
        assert result.image_size == (60, 60)

    def test_boundary_entropy_value(self):
        """Test entropy at the boundary (3.5 bits)."""
        # This tests that the threshold is properly applied
        # Create an image with approximately 3.5 bits of entropy
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Use only ~12 distinct values (2^3.5 ≈ 11.3)
        values = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250], dtype=np.uint8)
        rng = np.random.default_rng(42)
        image[:, :, 0] = rng.choice(values, size=(100, 100))
        image[:, :, 1] = image[:, :, 0]
        image[:, :, 2] = image[:, :, 0]
        entropy = calculate_entropy(image)
        # The entropy should be around 3.5 bits
        assert 3.0 < entropy < 4.0, f"Expected entropy around 3.5, got {entropy}"

    def test_prefilter_result_fields(self):
        """Test that PrefilterResult has all expected fields."""
        # Create image that passes all checks
        rng = np.random.default_rng(42)
        image = rng.integers(50, 150, size=(200, 200, 3), dtype=np.uint8)
        image[50:150, 50:150, 0] = 180
        image[50:150, 50:150, 1] = 140
        image[50:150, 50:150, 2] = 100
        result = should_skip_cnn(image)

        # Verify all fields are present
        assert hasattr(result, 'should_skip')
        assert hasattr(result, 'reason')
        assert hasattr(result, 'entropy')
        assert hasattr(result, 'skin_percentage')
        assert hasattr(result, 'image_size')

        # Verify types
        assert isinstance(result.should_skip, bool)
        assert isinstance(result.reason, str)
        assert isinstance(result.entropy, float)
        assert isinstance(result.skin_percentage, float)
        assert isinstance(result.image_size, tuple)
        assert len(result.image_size) == 2


class TestPrefilterWithRealImages:
    """Test prefilter with real image files if available."""

    @pytest.fixture
    def test_image_path(self):
        """Get test image path if available."""
        from pathlib import Path
        image_path = Path(__file__).parent.parent / "face.jpg"
        if not image_path.exists():
            pytest.skip("face.jpg not found")
        return image_path

    def test_real_face_image_not_skipped(self, test_image_path):
        """Real face image should NOT be skipped."""
        from face_finder.detector import load_image_with_exif_rotation

        image = load_image_with_exif_rotation(test_image_path)
        result = should_skip_cnn(image)
        assert result.should_skip is False, f"Face image should not be skipped, reason: {result.reason}"
        # Real face images should have both entropy and skin populated
        assert result.entropy is not None
        assert result.skin_percentage is not None
