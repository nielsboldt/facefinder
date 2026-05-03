"""Tests for duplicate file detection functionality."""

import pytest
from pathlib import Path

from face_finder.matcher import compute_file_hash


class TestComputeFileHash:
    """Test the compute_file_hash function."""

    def test_same_content_same_hash(self, tmp_path):
        """Files with identical content should have the same hash."""
        file1 = tmp_path / "file1.jpg"
        file2 = tmp_path / "file2.jpg"
        content = b"identical content here"
        file1.write_bytes(content)
        file2.write_bytes(content)

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Files with different content should have different hashes."""
        file1 = tmp_path / "file1.jpg"
        file2 = tmp_path / "file2.jpg"
        file1.write_bytes(b"content A")
        file2.write_bytes(b"content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)

    def test_hash_is_md5_hex(self, tmp_path):
        """Hash should be a 32-character MD5 hex string."""
        file1 = tmp_path / "file1.jpg"
        file1.write_bytes(b"test content")

        hash_value = compute_file_hash(file1)
        assert len(hash_value) == 32
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_empty_file(self, tmp_path):
        """Empty file should have a consistent hash."""
        file1 = tmp_path / "empty.jpg"
        file1.write_bytes(b"")

        # MD5 of empty string is d41d8cd98f00b204e9800998ecf8427e
        assert compute_file_hash(file1) == "d41d8cd98f00b204e9800998ecf8427e"

    def test_large_file_chunked(self, tmp_path):
        """Large files should be read in chunks without memory issues."""
        file1 = tmp_path / "large.jpg"
        # Write a file larger than the 8192 byte chunk size
        file1.write_bytes(b"x" * 50000)

        hash_value = compute_file_hash(file1)
        assert len(hash_value) == 32
