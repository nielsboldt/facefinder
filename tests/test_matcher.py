"""Tests for the matcher module - specifically testing model options."""

import inspect
import pytest
from pathlib import Path
import numpy as np

from face_finder.matcher import is_match, check_match, MatchInfo, MatchError, process_images
from face_finder.detector import load_encoding


@pytest.fixture
def reference_encoding():
    """Load pre-computed reference encoding."""
    encoding_path = Path(__file__).parent.parent / "person.npy"
    if not encoding_path.exists():
        pytest.skip("person.npy not found - run 'analyze' first")
    encoding, _ = load_encoding(encoding_path)  # Handles both v1 and v2 format
    return encoding


@pytest.fixture
def test_image():
    """Get a test image from the project."""
    image_path = Path(__file__).parent.parent / "face.jpg"
    if not image_path.exists():
        pytest.skip("face.jpg not found")
    return image_path


class TestIsMatchModels:
    """Test is_match with different face detection models."""

    def test_hog_model(self, reference_encoding, test_image):
        """HOG model should work (baseline test)."""
        result = is_match(test_image, reference_encoding, model="hog")
        assert isinstance(result, bool)

    def test_cnn_model(self, reference_encoding, test_image):
        """CNN model - this may crash if dlib CNN support is missing."""
        result = is_match(test_image, reference_encoding, model="cnn")
        assert isinstance(result, bool)

    def test_auto_model(self, reference_encoding, test_image):
        """Auto model - tries HOG first, falls back to CNN."""
        result = is_match(test_image, reference_encoding, model="auto")
        assert isinstance(result, bool)


class TestCNNModelDirect:
    """Direct tests of CNN model to isolate the issue."""

    def test_cnn_face_locations_directly(self, test_image):
        """Test face_recognition.face_locations with CNN directly."""
        import face_recognition
        from face_finder.detector import load_image_with_exif_rotation

        image = load_image_with_exif_rotation(test_image)
        # This call may crash if CNN support is missing
        locations = face_recognition.face_locations(image, model="cnn")
        assert isinstance(locations, list)

    def test_cnn_available(self):
        """Check if CNN model files are available."""
        import face_recognition_models
        try:
            cnn_model_path = face_recognition_models.cnn_face_detector_model_location()
            assert Path(cnn_model_path).exists(), f"CNN model not found at {cnn_model_path}"
        except Exception as e:
            pytest.fail(f"CNN model not available: {e}")


class TestCNNProblematicImages:
    """Test CNN with images that previously caused crashes."""

    def test_cnn_with_problematic_image(self, reference_encoding):
        """Test CNN with image that previously caused segfault due to non-contiguous array."""
        problem_image = Path(__file__).parent / "fixtures" / "problem_image.jpg"
        if not problem_image.exists():
            pytest.skip("problem_image.jpg not found in fixtures")
        # This would segfault before the fix due to non-contiguous array from PIL conversion
        result = is_match(problem_image, reference_encoding, model="cnn")
        assert isinstance(result, bool)

    def test_hog_with_problematic_image(self, reference_encoding):
        """Test HOG with the same problematic image (baseline comparison)."""
        problem_image = Path(__file__).parent / "fixtures" / "problem_image.jpg"
        if not problem_image.exists():
            pytest.skip("problem_image.jpg not found in fixtures")
        result = is_match(problem_image, reference_encoding, model="hog")
        assert isinstance(result, bool)


class TestMultiImageProcessing:
    """Test processing multiple images to reproduce crashes."""

    def test_process_multiple_images_with_cnn(self, reference_encoding):
        """Process multiple images with CNN model."""
        from face_finder.matcher import process_images
        import tempfile

        # Search the project root for images (includes face.jpg and any others)
        search_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_images = []

            def track_pre_process(image_path: Path, model: str) -> None:
                processed_images.append((image_path, model))

            result = process_images(
                image_source=search_dir,
                output_dir=Path(tmpdir),
                reference_encoding=reference_encoding,
                model="cnn",
                pre_process_callback=track_pre_process,
            )
            assert result.total_scanned >= 0
            # Verify pre_process_callback was called for each image
            assert len(processed_images) == result.total_scanned

    def test_process_multiple_images_with_auto(self, reference_encoding):
        """Process multiple images with auto model."""
        from face_finder.matcher import process_images
        import tempfile

        search_dir = Path(__file__).parent.parent
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_images = []

            def track_pre_process(image_path: Path, model: str) -> None:
                processed_images.append((image_path, model))

            result = process_images(
                image_source=search_dir,
                output_dir=Path(tmpdir),
                reference_encoding=reference_encoding,
                model="auto",
                pre_process_callback=track_pre_process,
            )
            assert result.total_scanned >= 0
            assert len(processed_images) == result.total_scanned

    def test_pre_process_callback_called_before_match(self, reference_encoding, test_image):
        """Verify pre_process_callback is called BEFORE is_match."""
        from face_finder.matcher import process_images
        import tempfile

        call_order = []

        def track_pre_process(image_path: Path, model: str) -> None:
            call_order.append(("pre", image_path.name))

        def track_progress(image_path: Path, matched: bool, error: bool = False, error_msg: str = None) -> None:
            call_order.append(("post", image_path.name))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = process_images(
                image_source=test_image.parent,
                output_dir=Path(tmpdir),
                reference_encoding=reference_encoding,
                model="hog",
                pre_process_callback=track_pre_process,
                progress_callback=track_progress,
            )

            # Verify for each image, pre comes before post
            for i in range(0, len(call_order), 2):
                if i + 1 < len(call_order):
                    assert call_order[i][0] == "pre", "pre_process_callback should be called first"
                    assert call_order[i + 1][0] == "post", "progress_callback should be called after"
                    assert call_order[i][1] == call_order[i + 1][1], "callbacks should be for same image"


class TestDefaultTolerance:
    """Test that default tolerance is correctly set to 0.40."""

    def test_is_match_default_tolerance(self):
        """Verify is_match uses 0.40 as default tolerance."""
        sig = inspect.signature(is_match)
        assert sig.parameters['tolerance'].default == 0.40

    def test_check_match_default_tolerance(self):
        """Verify check_match uses 0.40 as default tolerance."""
        sig = inspect.signature(check_match)
        assert sig.parameters['tolerance'].default == 0.40

    def test_process_images_default_tolerance(self):
        """Verify process_images uses 0.40 as default tolerance."""
        sig = inspect.signature(process_images)
        assert sig.parameters['tolerance'].default == 0.40


class TestMatchInfo:
    """Test the MatchInfo dataclass for verbose matching."""

    def test_match_info_fields(self):
        """Verify MatchInfo has required fields."""
        info = MatchInfo(matched=True, min_distance=0.35, num_faces=2)
        assert info.matched is True
        assert info.min_distance == 0.35
        assert info.num_faces == 2

    def test_match_info_no_faces(self):
        """MatchInfo with no faces detected."""
        info = MatchInfo(matched=False, min_distance=None, num_faces=0)
        assert info.matched is False
        assert info.min_distance is None
        assert info.num_faces == 0

    def test_match_info_prefilter_fields(self):
        """Verify MatchInfo has prefilter fields."""
        info = MatchInfo(
            matched=False,
            min_distance=None,
            num_faces=0,
            prefilter_skipped=True,
            prefilter_reason="no skin tones detected",
        )
        assert info.prefilter_skipped is True
        assert info.prefilter_reason == "no skin tones detected"

    def test_match_info_prefilter_defaults(self):
        """Verify MatchInfo prefilter fields have correct defaults."""
        info = MatchInfo(matched=False, min_distance=None, num_faces=0)
        assert info.prefilter_skipped is False
        assert info.prefilter_reason == ""

    def test_match_info_prefilter_score_fields(self):
        """Verify MatchInfo has prefilter score fields."""
        info = MatchInfo(
            matched=False,
            min_distance=None,
            num_faces=0,
            prefilter_skipped=True,
            prefilter_reason="no skin tones detected",
            prefilter_entropy=6.5,
            prefilter_skin_pct=0.3,
        )
        assert info.prefilter_entropy == 6.5
        assert info.prefilter_skin_pct == 0.3

    def test_match_info_prefilter_score_defaults(self):
        """Verify MatchInfo prefilter score fields have correct defaults."""
        info = MatchInfo(matched=False, min_distance=None, num_faces=0)
        assert info.prefilter_entropy is None
        assert info.prefilter_skin_pct is None

    def test_match_info_with_all_fields(self):
        """MatchInfo with all fields populated (typical CNN match result)."""
        info = MatchInfo(
            matched=True,
            min_distance=0.35,
            num_faces=2,
            prefilter_skipped=False,
            prefilter_reason="",
            prefilter_entropy=7.2,
            prefilter_skin_pct=18.5,
        )
        assert info.matched is True
        assert info.min_distance == 0.35
        assert info.num_faces == 2
        assert info.prefilter_skipped is False
        assert info.prefilter_entropy == 7.2
        assert info.prefilter_skin_pct == 18.5


class TestCheckMatch:
    """Test check_match function that returns detailed match info."""

    def test_check_match_returns_match_info(self, reference_encoding, test_image):
        """check_match should return MatchInfo with distance values."""
        result = check_match(test_image, reference_encoding, model="hog")
        assert hasattr(result, 'matched')
        assert hasattr(result, 'min_distance')
        assert hasattr(result, 'num_faces')
        assert isinstance(result.matched, bool)

    def test_check_match_distance_within_tolerance(self, reference_encoding, test_image):
        """When matched=True, min_distance should be <= tolerance."""
        tolerance = 0.45
        result = check_match(test_image, reference_encoding, tolerance=tolerance, model="hog")
        if result.matched:
            assert result.min_distance is not None
            assert result.min_distance <= tolerance

    def test_check_match_distance_above_tolerance(self, reference_encoding, test_image):
        """When matched=False and faces detected, min_distance should be > tolerance."""
        # Use very strict tolerance to force a non-match
        result = check_match(test_image, reference_encoding, tolerance=0.01, model="hog")
        if result.num_faces > 0 and not result.matched:
            assert result.min_distance > 0.01


class TestToleranceAffectsMatching:
    """Verify that tolerance parameter correctly affects matching."""

    def test_strict_tolerance_reduces_matches(self, reference_encoding, test_image):
        """Stricter tolerance (lower value) should not increase matches."""
        # Get result with permissive tolerance
        permissive = check_match(test_image, reference_encoding, tolerance=0.8, model="hog")
        # Get result with strict tolerance
        strict = check_match(test_image, reference_encoding, tolerance=0.3, model="hog")

        # If strict matches, permissive must also match
        if strict.matched:
            assert permissive.matched, "Strict match implies permissive match"

        # Distance should be same regardless of tolerance
        if permissive.min_distance is not None and strict.min_distance is not None:
            assert abs(permissive.min_distance - strict.min_distance) < 0.001


class TestFalsePositiveRejection:
    """Test that known false positives are rejected at default tolerance."""

    def test_false_positive_rejected_at_default_tolerance(self, reference_encoding):
        """Image with distance ~0.411 should NOT match at tolerance=0.40."""
        fp_image = Path(__file__).parent / "fixtures" / "false_positive_cnn.jpg"
        if not fp_image.exists():
            pytest.skip("false_positive_cnn.jpg not found in fixtures")

        result = check_match(fp_image, reference_encoding, tolerance=0.40, model="cnn")

        assert not result.matched, f"False positive matched with distance {result.min_distance}"
        assert result.min_distance is not None
        assert result.min_distance > 0.40


class TestPrefilterIntegration:
    """Test prefilter integration with matching functions."""

    def test_check_match_with_cnn_returns_prefilter_info(self, reference_encoding, test_image):
        """check_match with CNN should include prefilter info (even if not skipped)."""
        result = check_match(test_image, reference_encoding, model="cnn")
        assert hasattr(result, 'prefilter_skipped')
        assert hasattr(result, 'prefilter_reason')
        assert hasattr(result, 'prefilter_entropy')
        assert hasattr(result, 'prefilter_skin_pct')
        # Real face image should not be skipped
        assert result.prefilter_skipped is False
        # Scores should be populated for CNN mode
        assert result.prefilter_entropy is not None
        assert result.prefilter_skin_pct is not None

    def test_check_match_with_hog_no_prefilter(self, reference_encoding, test_image):
        """check_match with HOG should not use prefilter."""
        result = check_match(test_image, reference_encoding, model="hog")
        # HOG doesn't use prefilter, so these should be defaults
        assert result.prefilter_skipped is False
        assert result.prefilter_reason == ""
        # Scores should be None for HOG mode (no prefilter run)
        assert result.prefilter_entropy is None
        assert result.prefilter_skin_pct is None

    def test_is_match_skips_tiny_image_cnn(self, reference_encoding, tmp_path):
        """is_match with CNN should skip tiny images via prefilter."""
        from PIL import Image

        # Create tiny 30x30 image
        tiny_image = tmp_path / "tiny.jpg"
        img = Image.new('RGB', (30, 30), color='red')
        img.save(tiny_image)

        result = is_match(tiny_image, reference_encoding, model="cnn")
        assert result is False  # Should return False without running CNN

    def test_check_match_skips_solid_color_cnn(self, reference_encoding, tmp_path):
        """check_match with CNN should skip solid color images."""
        from PIL import Image

        # Create solid color image
        solid_image = tmp_path / "solid.jpg"
        img = Image.new('RGB', (200, 200), color=(100, 100, 100))
        img.save(solid_image)

        result = check_match(solid_image, reference_encoding, model="cnn")
        assert result.prefilter_skipped is True
        assert "blank" in result.prefilter_reason or "solid" in result.prefilter_reason
        # Entropy should be populated (computed before skip)
        assert result.prefilter_entropy is not None
        assert result.prefilter_entropy < 3.5  # Low entropy = solid

    def test_check_match_skips_landscape_no_skin_cnn(self, reference_encoding, tmp_path):
        """check_match with CNN should skip landscape (blue/green) images."""
        from PIL import Image
        import numpy as np

        # Create varied blue/green landscape-like image (high entropy, no skin)
        landscape_image = tmp_path / "landscape.jpg"
        rng = np.random.default_rng(42)
        # Create image with varied blue/green tones (no red dominance = no skin)
        pixels = np.zeros((200, 200, 3), dtype=np.uint8)
        pixels[:, :, 0] = rng.integers(0, 80, size=(200, 200), dtype=np.uint8)    # Low red
        pixels[:, :, 1] = rng.integers(100, 200, size=(200, 200), dtype=np.uint8) # Green
        pixels[:, :, 2] = rng.integers(150, 255, size=(200, 200), dtype=np.uint8) # Blue
        img = Image.fromarray(pixels)
        img.save(landscape_image)

        result = check_match(landscape_image, reference_encoding, model="cnn")
        assert result.prefilter_skipped is True
        assert "skin" in result.prefilter_reason
        # Both entropy and skin should be populated
        assert result.prefilter_entropy is not None
        assert result.prefilter_entropy > 3.5  # High entropy landscape
        assert result.prefilter_skin_pct is not None
        assert result.prefilter_skin_pct < 1.0  # No skin detected
