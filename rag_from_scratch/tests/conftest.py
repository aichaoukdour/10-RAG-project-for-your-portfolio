"""
Pytest configuration and shared fixtures.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'work_year': [2024, 2024, 2023],
        'experience_level': ['SE', 'MI', 'EN'],
        'employment_type': ['FT', 'FT', 'CT'],
        'job_title': ['Data Scientist', 'ML Engineer', 'Data Analyst'],
        'salary_in_usd': [150000, 120000, 80000],
        'employee_residence': ['US', 'US', 'UK'],
        'remote_ratio': [100, 50, 0],
        'company_location': ['US', 'US', 'UK'],
        'company_size': ['M', 'L', 'S']
    })


@pytest.fixture
def sample_embeddings():
    """Create sample normalized embeddings for testing."""
    dim = 384
    vectors = np.random.random((5, dim)).astype('float32')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def sample_text_chunks():
    """Create sample text chunks for testing."""
    return [
        "In 2024, a Senior-level Data Scientist working Full-time in US earned 150,000 USD.",
        "In 2024, a Mid-level ML Engineer working Full-time in US earned 120,000 USD.",
        "In 2023, an Entry-level Data Analyst working Contract in UK earned 80,000 USD."
    ]
