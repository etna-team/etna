import hashlib
import os
import pathlib
import tempfile
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from etna.core import load
from etna.core.utils import download_with_integrity_check, get_known_hash, verify_file_hash, KNOWN_MODEL_HASHES
from etna.models import NaiveModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform


def test_load_fail_file_not_found():
    non_existent_path = pathlib.Path("archive.zip")
    with pytest.raises(FileNotFoundError):
        load(non_existent_path)


def test_load_ok():
    transform = AddConstTransform(in_column="target", value=10.0, inplace=False)
    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = pathlib.Path(_temp_dir)
        save_path = temp_dir / "transform.zip"
        transform.save(save_path)

        new_transform = load(save_path)

        assert type(new_transform) == type(transform)
        for attribute in ["in_column", "value", "inplace"]:
            assert getattr(new_transform, attribute) == getattr(transform, attribute)


def test_load_ok_with_params(example_tsds):
    pipeline = Pipeline(model=NaiveModel(), horizon=7)
    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = pathlib.Path(_temp_dir)
        save_path = temp_dir / "pipeline.zip"
        pipeline.fit(ts=example_tsds)
        pipeline.save(save_path)

        new_pipeline = load(save_path, ts=example_tsds)

        assert new_pipeline.ts is not None
        assert type(new_pipeline) == type(pipeline)
        pd.testing.assert_frame_equal(new_pipeline.ts.to_pandas(), example_tsds.to_pandas())


class TestVerifyFileHash:
    def test_verify_file_hash_no_expected_hash(self):
        """Test that verification returns True when no expected hash is provided."""
        with tempfile.NamedTemporaryFile() as temp_file:
            result = verify_file_hash(temp_file.name, expected_hash=None)
            assert result is True

    def test_verify_file_hash_file_not_exists(self):
        """Test that verification returns False when file doesn't exist."""
        result = verify_file_hash("/nonexistent/file.txt", expected_hash="dummy_hash")
        assert result is False

    def test_verify_file_hash_correct_hash(self):
        """Test that verification returns True when hash matches."""
        test_content = b"test content"
        expected_hash = hashlib.md5(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            result = verify_file_hash(temp_file.name, expected_hash=expected_hash)
            assert result is True

    def test_verify_file_hash_incorrect_hash(self):
        """Test that verification returns False when hash doesn't match."""
        test_content = b"test content"
        wrong_hash = "wrong_hash"
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            result = verify_file_hash(temp_file.name, expected_hash=wrong_hash)
            assert result is False

    def test_verify_file_hash_chunked_reading(self):
        """Test that chunked reading works correctly for large files."""
        # Create content larger than chunk size (4096 bytes)
        test_content = b"x" * 10000
        expected_hash = hashlib.md5(test_content).hexdigest()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(test_content)
            temp_file.flush()
            
            result = verify_file_hash(temp_file.name, expected_hash=expected_hash)
            assert result is True

    def test_verify_file_hash_exception_handling(self):
        """Test that verification returns False when an exception occurs."""
        with patch("builtins.open", side_effect=IOError("File read error")):
            result = verify_file_hash("dummy_path", expected_hash="dummy_hash")
            assert result is False


class TestGetKnownHash:
    def test_get_known_hash_existing_url(self):
        """Test retrieving a known hash for an existing URL."""
        test_url = "http://example.com/model.ckpt"
        test_hash = "abcd1234"
        
        # Temporarily add to known hashes
        original_hashes = KNOWN_MODEL_HASHES.copy()
        KNOWN_MODEL_HASHES[test_url] = test_hash
        
        try:
            result = get_known_hash(test_url)
            assert result == test_hash
        finally:
            # Restore original hashes
            KNOWN_MODEL_HASHES.clear()
            KNOWN_MODEL_HASHES.update(original_hashes)

    def test_get_known_hash_nonexistent_url(self):
        """Test retrieving hash for a non-existent URL returns None."""
        result = get_known_hash("http://nonexistent.com/model.ckpt")
        assert result is None


class TestDownloadWithIntegrityCheck:
    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('etna.core.utils.verify_file_hash')
    def test_download_file_exists_and_valid(self, mock_verify, mock_exists, mock_urlretrieve):
        """Test that download is skipped when file exists and is valid."""
        mock_exists.return_value = True
        mock_verify.return_value = True
        
        download_with_integrity_check(
            url="http://example.com/model.ckpt",
            destination_path="/path/to/model.ckpt",
            expected_hash="abcd1234"
        )
        
        mock_urlretrieve.assert_not_called()

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    def test_download_file_exists_but_invalid(self, mock_makedirs, mock_verify, mock_remove, mock_exists, mock_urlretrieve):
        """Test that file is re-downloaded when existing file fails verification."""
        mock_exists.return_value = True
        mock_verify.side_effect = [False, True]  # First call (existing file) fails, second call (after download) succeeds
        
        with patch('warnings.warn') as mock_warn:
            download_with_integrity_check(
                url="http://example.com/model.ckpt",
                destination_path="/path/to/model.ckpt",
                expected_hash="abcd1234"
            )
        
        mock_remove.assert_called_once_with("/path/to/model.ckpt")
        mock_urlretrieve.assert_called_once_with(url="http://example.com/model.ckpt", filename="/path/to/model.ckpt")
        mock_warn.assert_called_once()

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    def test_download_file_not_exists(self, mock_makedirs, mock_verify, mock_exists, mock_urlretrieve):
        """Test that file is downloaded when it doesn't exist."""
        mock_exists.return_value = False
        mock_verify.return_value = True
        
        download_with_integrity_check(
            url="http://example.com/model.ckpt",
            destination_path="/path/to/model.ckpt",
            expected_hash="abcd1234"
        )
        
        mock_urlretrieve.assert_called_once_with(url="http://example.com/model.ckpt", filename="/path/to/model.ckpt")

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('os.remove')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    def test_download_fails_integrity_check(self, mock_makedirs, mock_verify, mock_remove, mock_exists, mock_urlretrieve):
        """Test that RuntimeError is raised when downloaded file fails integrity check."""
        mock_exists.return_value = False
        mock_verify.return_value = False
        
        with pytest.raises(RuntimeError, match="Downloaded file from .* failed integrity check"):
            download_with_integrity_check(
                url="http://example.com/model.ckpt",
                destination_path="/path/to/model.ckpt",
                expected_hash="abcd1234"
            )
        
        mock_remove.assert_called_once_with("/path/to/model.ckpt")

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    def test_download_no_expected_hash(self, mock_makedirs, mock_verify, mock_exists, mock_urlretrieve):
        """Test that download works without integrity checking when no hash is provided."""
        mock_exists.return_value = False
        mock_verify.return_value = True  # Should return True when no hash is provided
        
        download_with_integrity_check(
            url="http://example.com/model.ckpt",
            destination_path="/path/to/model.ckpt",
            expected_hash=None
        )
        
        mock_urlretrieve.assert_called_once_with(url="http://example.com/model.ckpt", filename="/path/to/model.ckpt")

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    def test_force_redownload(self, mock_makedirs, mock_verify, mock_exists, mock_urlretrieve):
        """Test that file is re-downloaded when force_redownload is True."""
        mock_exists.return_value = True
        mock_verify.return_value = True
        
        download_with_integrity_check(
            url="http://example.com/model.ckpt",
            destination_path="/path/to/model.ckpt",
            expected_hash="abcd1234",
            force_redownload=True
        )
        
        mock_urlretrieve.assert_called_once_with(url="http://example.com/model.ckpt", filename="/path/to/model.ckpt")

    @patch('etna.core.utils.request.urlretrieve')
    @patch('os.path.exists')
    @patch('etna.core.utils.verify_file_hash')
    @patch('os.makedirs')
    @patch('os.path.dirname')
    def test_creates_directory(self, mock_dirname, mock_makedirs, mock_verify, mock_exists, mock_urlretrieve):
        """Test that destination directory is created if it doesn't exist."""
        mock_exists.return_value = False
        mock_verify.return_value = True
        mock_dirname.return_value = "/path/to"
        
        download_with_integrity_check(
            url="http://example.com/model.ckpt",
            destination_path="/path/to/model.ckpt",
            expected_hash="abcd1234"
        )
        
        mock_makedirs.assert_called_once_with("/path/to", exist_ok=True)
