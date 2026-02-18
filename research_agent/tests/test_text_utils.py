import pytest
from src.text_utils import chunk_passages, split_sentences

def test_chunk_passages():
    text = "one two three four five six"
    # Chunk by 2 words
    chunks = chunk_passages(text, max_words=2)
    assert len(chunks) == 3
    assert chunks[0] == "one two"
    assert chunks[1] == "three four"
    assert chunks[2] == "five six"

def test_chunk_passages_empty():
    assert chunk_passages("") == []

def test_split_sentences():
    text = "Hello world. This is a test! Is it working?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test!"
    assert sentences[2] == "Is it working?"

def test_split_sentences_no_punctuation():
    text = "hello world"
    assert split_sentences(text) == ["hello world"]
