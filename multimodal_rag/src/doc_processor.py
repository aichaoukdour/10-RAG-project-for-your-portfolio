"""
Document Processor: Parse PDFs with Docling.

Converts PDF documents into text chunks, tables (markdown), and images
using Docling's DocumentConverter and HybridChunker.
"""

import os
import urllib.request
from langchain_core.documents import Document

from src.config import SAMPLE_PDF_URL, DATA_DIR


def download_sample_pdf(url: str = SAMPLE_PDF_URL, dest_dir: str = DATA_DIR):
    """Download the sample PDF if not already present."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath):
        print(f"📄 Sample PDF already exists: {filepath}")
        return filepath

    print(f"⬇️  Downloading sample PDF from {url}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✅ Saved to {filepath}")
    return filepath


def convert_pdf(source: str):
    """
    Use Docling to convert a PDF into a structured document.
    Returns a Docling document object with text, tables, and images.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    print(f"📖 Converting PDF: {source}")

    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
    )
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
    }
    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(source=source)
    print("✅ PDF conversion complete.")
    return result.document


def chunk_texts(docling_document, tokenizer=None):
    """
    Split text content from the Docling document into chunks.
    Returns a list of LangChain Document objects.
    """
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.types.doc.document import TableItem

    chunker = HybridChunker(tokenizer=tokenizer) if tokenizer else HybridChunker()
    texts = []
    doc_id = 0

    for chunk in chunker.chunk(docling_document):
        items = chunk.meta.doc_items
        # Skip standalone tables — we handle those separately
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue

        refs = " ".join(map(lambda item: item.get_ref().cref, items))
        document = Document(
            page_content=chunk.text,
            metadata={
                "doc_id": (doc_id := doc_id + 1),
                "type": "text",
                "ref": refs,
            },
        )
        texts.append(document)

    print(f"✂️  Created {len(texts)} text chunks.")
    return texts


def extract_tables(docling_document, start_id: int = 0):
    """
    Extract tables from the Docling document as markdown.
    Returns a list of LangChain Document objects.
    """
    from docling_core.types.doc.labels import DocItemLabel

    tables = []
    doc_id = start_id

    for table in docling_document.tables:
        if table.label in [DocItemLabel.TABLE]:
            ref = table.get_ref().cref
            text = table.export_to_markdown()
            document = Document(
                page_content=text,
                metadata={
                    "doc_id": (doc_id := doc_id + 1),
                    "type": "table",
                    "ref": ref,
                },
            )
            tables.append(document)

    print(f"📊 Extracted {len(tables)} tables.")
    return tables


def extract_images(docling_document):
    """
    Extract images from the Docling document as PIL Image objects.
    Returns a list of (ref, PIL.Image) tuples.
    """
    images = []
    for picture in docling_document.pictures:
        ref = picture.get_ref().cref
        image = picture.get_image(docling_document)
        if image:
            images.append((ref, image))

    print(f"🖼️  Found {len(images)} images.")
    return images


def process_pdf(source: str, tokenizer=None):
    """
    Full document processing pipeline:
    PDF → Docling → text chunks + tables + images.

    Returns (text_docs, table_docs, images) where images
    are (ref, PIL.Image) tuples ready for captioning.
    """
    docling_doc = convert_pdf(source)
    text_docs = chunk_texts(docling_doc, tokenizer=tokenizer)
    table_docs = extract_tables(docling_doc, start_id=len(text_docs))
    images = extract_images(docling_doc)
    return text_docs, table_docs, images
