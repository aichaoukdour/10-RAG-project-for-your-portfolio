"""
Tests for the ingestion module.
"""
import pytest
import pandas as pd

from ingestion import clean_data
from chunks import create_text_chunks
from config import EXPERIENCE_MAPPING, EMPLOYMENT_MAPPING, COMPANY_SIZE_MAPPING


class TestCleanData:
    """Tests for the clean_data function."""
    
    def test_removes_duplicates(self, sample_dataframe):
        """Test that duplicate rows are removed."""
        # Create a DataFrame with duplicates
        df_with_dups = pd.concat([sample_dataframe, sample_dataframe.iloc[[0]]])
        assert len(df_with_dups) == 4  # 3 original + 1 duplicate
        
        cleaned = clean_data(df_with_dups)
        assert len(cleaned) == 3  # Duplicates removed
    
    def test_maps_experience_levels(self, sample_dataframe):
        """Test that experience level codes are mapped to full names."""
        cleaned = clean_data(sample_dataframe)
        
        assert cleaned.loc[0, 'experience_level'] == 'Senior-level'
        assert cleaned.loc[1, 'experience_level'] == 'Mid-level'
        assert cleaned.loc[2, 'experience_level'] == 'Entry-level'
    
    def test_maps_employment_types(self, sample_dataframe):
        """Test that employment type codes are mapped correctly."""
        cleaned = clean_data(sample_dataframe)
        
        assert cleaned.loc[0, 'employment_type'] == 'Full-time'
        assert cleaned.loc[2, 'employment_type'] == 'Contract'
    
    def test_maps_company_sizes(self, sample_dataframe):
        """Test that company size codes are mapped correctly."""
        cleaned = clean_data(sample_dataframe)
        
        assert cleaned.loc[0, 'company_size'] == 'medium'
        assert cleaned.loc[1, 'company_size'] == 'large'
        assert cleaned.loc[2, 'company_size'] == 'small'
    
    def test_handles_unknown_codes(self):
        """Test that unknown codes are preserved."""
        df = pd.DataFrame({
            'experience_level': ['UNKNOWN'],
            'employment_type': ['XX'],
            'company_size': ['XL']
        })
        cleaned = clean_data(df)
        
        assert cleaned.loc[0, 'experience_level'] == 'UNKNOWN'
        assert cleaned.loc[0, 'employment_type'] == 'XX'


class TestCreateTextChunks:
    """Tests for the create_text_chunks function."""
    
    def test_creates_text_chunk_column(self, sample_dataframe):
        """Test that text_chunk column is created."""
        cleaned = clean_data(sample_dataframe)
        chunked = create_text_chunks(cleaned)
        
        assert 'text_chunk' in chunked.columns
        assert len(chunked) == 3
    
    def test_chunk_contains_salary(self, sample_dataframe):
        """Test that text chunks contain salary information."""
        cleaned = clean_data(sample_dataframe)
        chunked = create_text_chunks(cleaned)
        
        assert '150,000 USD' in chunked.loc[0, 'text_chunk']
    
    def test_chunk_contains_job_title(self, sample_dataframe):
        """Test that text chunks contain job title."""
        cleaned = clean_data(sample_dataframe)
        chunked = create_text_chunks(cleaned)
        
        assert 'Data Scientist' in chunked.loc[0, 'text_chunk']
    
    def test_remote_status_mapping(self, sample_dataframe):
        """Test that remote ratio is correctly mapped to text."""
        cleaned = clean_data(sample_dataframe)
        chunked = create_text_chunks(cleaned)
        
        assert 'remote' in chunked.loc[0, 'text_chunk'].lower()
        assert 'hybrid' in chunked.loc[1, 'text_chunk'].lower()
        assert 'on-site' in chunked.loc[2, 'text_chunk'].lower()
