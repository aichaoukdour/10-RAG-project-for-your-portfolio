"""
Image Captioner: Generate text descriptions for images.

Uses Salesforce/blip-image-captioning-base to run locally,
replacing the Granite Vision model from the original tutorial.
"""

from langchain_core.documents import Document
from src.config import VISION_MODEL_NAME


_captioner = None


def get_captioner():
    """Load the BLIP image captioning pipeline (cached)."""
    global _captioner
    if _captioner is None:
        from transformers import pipeline as hf_pipeline
        print(f"👁️  Loading vision model: {VISION_MODEL_NAME}...")
        _captioner = hf_pipeline(
            "image-to-text",
            model=VISION_MODEL_NAME,
        )
    return _captioner


def caption_single_image(image):
    """
    Generate a text caption for a single PIL Image.
    Returns the caption string.
    """
    captioner = get_captioner()
    results = captioner(image)
    caption = results[0]["generated_text"] if results else "Image content"
    return caption


def caption_images(images, start_id: int = 0):
    """
    Generate text descriptions for a list of (ref, PIL.Image) tuples.
    Returns a list of LangChain Document objects with captions as page_content.

    This replaces the Granite Vision model approach from the tutorial,
    using BLIP for fully local image-to-text generation.
    """
    if not images:
        print("🖼️  No images to caption.")
        return []

    print(f"🖼️  Captioning {len(images)} images...")
    captioner = get_captioner()
    documents = []
    doc_id = start_id

    for ref, image in images:
        try:
            results = captioner(image)
            caption = results[0]["generated_text"] if results else "Image content"
            document = Document(
                page_content=caption,
                metadata={
                    "doc_id": (doc_id := doc_id + 1),
                    "type": "image",
                    "ref": ref,
                },
            )
            documents.append(document)
            print(f"  ✓ {ref}: {caption[:80]}...")
        except Exception as e:
            print(f"  ✗ {ref}: Failed to caption — {e}")

    print(f"✅ Created {len(documents)} image descriptions.")
    return documents
