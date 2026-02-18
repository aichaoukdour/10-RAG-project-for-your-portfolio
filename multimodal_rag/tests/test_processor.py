"""
Tests for document processing and image captioning.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


class TestDocProcessor:
    """Tests for document processing utilities."""

    def test_download_sample_pdf_creates_file(self, tmp_path):
        """download_sample_pdf should create a file in the destination."""
        from src.doc_processor import download_sample_pdf
        # Use a small publicly available PDF for testing
        test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        filepath = download_sample_pdf(url=test_url, dest_dir=str(tmp_path))
        assert filepath.endswith(".pdf")

    def test_download_sample_pdf_skips_existing(self, tmp_path):
        """Should not re-download if file already exists."""
        from src.doc_processor import download_sample_pdf
        # Create a dummy file
        dummy = tmp_path / "dummy.pdf"
        dummy.write_text("fake pdf")
        test_url = f"https://example.com/dummy.pdf"
        filepath = download_sample_pdf(url=test_url, dest_dir=str(tmp_path))
        assert filepath == str(dummy)


class TestImageCaptioner:
    """Tests for image captioning."""

    def test_caption_single_image(self):
        """caption_single_image should return a non-empty string."""
        import PIL.Image
        from src.image_captioner import caption_single_image
        # Create a simple test image
        image = PIL.Image.new("RGB", (100, 100), color="red")
        caption = caption_single_image(image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_caption_images_empty_list(self):
        """caption_images with empty list should return empty list."""
        from src.image_captioner import caption_images
        result = caption_images([])
        assert result == []

    def test_caption_images_returns_documents(self):
        """caption_images should return LangChain Document objects."""
        import PIL.Image
        from src.image_captioner import caption_images
        image = PIL.Image.new("RGB", (100, 100), color="blue")
        images = [("test_ref", image)]
        result = caption_images(images, start_id=0)
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].metadata["type"] == "image"
        assert len(result[0].page_content) > 0
