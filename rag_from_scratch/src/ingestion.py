"""
Data ingestion module for loading, cleaning, and chunking salary data.
Transforms raw CSV data into RAG-ready text chunks.
"""
import logging
import os
from typing import Callable

import pandas as pd

from config import (
    LOG_FORMAT, LOG_LEVEL,
    RAW_SALARIES_PATH, PROCESSED_SALARIES_PATH
)

# Setup module logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Mapping Constants ---
EXPERIENCE_MAPPING = {
    'EN': 'Entry-level',
    'MI': 'Mid-level',
    'SE': 'Senior-level',
    'EX': 'Executive-level'
}

EMPLOYMENT_MAPPING = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

COMPANY_SIZE_MAPPING = {
    'S': 'small',
    'M': 'medium',
    'L': 'large'
}

REMOTE_RATIO_MAPPING = {
    100: 'remote',
    50: 'hybrid',
    0: 'on-site'
}


def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the raw salary data from CSV.
    
    Args:
        file_path: Path to the CSV file. Defaults to config path.
        
    Returns:
        pd.DataFrame: Raw salary data.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
    """
    path = file_path or str(RAW_SALARIES_PATH)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}")
    
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data: remove duplicates and map codes to full names.
    
    Args:
        df: Raw DataFrame with coded values.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with human-readable values.
    """
    initial_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate records")
    
    # Map experience levels
    if 'experience_level' in df.columns:
        df['experience_level'] = (
            df['experience_level']
            .map(EXPERIENCE_MAPPING)
            .fillna(df['experience_level'])
        )
    
    # Map employment types
    if 'employment_type' in df.columns:
        df['employment_type'] = (
            df['employment_type']
            .map(EMPLOYMENT_MAPPING)
            .fillna(df['employment_type'])
        )
    
    # Map company sizes
    if 'company_size' in df.columns:
        df['company_size'] = (
            df['company_size']
            .map(COMPANY_SIZE_MAPPING)
            .fillna(df['company_size'])
        )
    
    logger.info(f"Data cleaning complete: {len(df)} records")
    return df


def create_text_chunks(
    df: pd.DataFrame, 
    chunk_fn: Callable[[pd.Series], str] = None
) -> pd.DataFrame:
    """
    Convert each row into a descriptive text chunk for RAG.
    
    Args:
        df: Cleaned DataFrame.
        chunk_fn: Optional custom function to convert a row to text.
        
    Returns:
        pd.DataFrame: DataFrame with added 'text_chunk' column.
    """
    def default_row_to_text(row: pd.Series) -> str:
        """Default chunking function for salary data."""
        remote_status = REMOTE_RATIO_MAPPING.get(row.get('remote_ratio', 0), 'on-site')
        
        salary = row.get('salary_in_usd', 0)
        salary_str = f"{salary:,}" if isinstance(salary, (int, float)) else str(salary)
        
        return (
            f"In {row.get('work_year', 'unknown year')}, "
            f"a {row.get('experience_level', '')} {row.get('job_title', 'professional')} "
            f"working {row.get('employment_type', '')} in {row.get('employee_residence', 'unknown location')} "
            f"earned a salary of {salary_str} USD. "
            f"The role was {remote_status} for a {row.get('company_size', 'unknown')}-sized company "
            f"located in {row.get('company_location', 'unknown location')}."
        )
    
    chunk_function = chunk_fn or default_row_to_text
    
    logger.info(f"Creating text chunks for {len(df)} records...")
    df['text_chunk'] = df.apply(chunk_function, axis=1)
    
    # Log sample chunk
    if len(df) > 0:
        logger.debug(f"Sample chunk: {df['text_chunk'].iloc[0][:100]}...")
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Save the cleaned and chunked data to CSV.
    
    Args:
        df: Processed DataFrame to save.
        output_path: Path to save the file. Defaults to config path.
        
    Returns:
        str: Path where the file was saved.
    """
    path = output_path or str(PROCESSED_SALARIES_PATH)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path} ({len(df)} records)")
    return path


def run_ingestion(raw_path: str = None, processed_path: str = None) -> pd.DataFrame:
    """
    Run the complete ingestion pipeline.
    
    Args:
        raw_path: Path to raw data file.
        processed_path: Path to save processed data.
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for embedding.
    """
    logger.info("--- Starting Data Ingestion Pipeline ---")
    
    df = load_data(raw_path)
    df = clean_data(df)
    df = create_text_chunks(df)
    save_processed_data(df, processed_path)
    
    logger.info("--- Data Ingestion Complete ---")
    return df


if __name__ == "__main__":
    # Test the ingestion pipeline
    try:
        df = run_ingestion()
        print(f"\nProcessed {len(df)} records")
        print(f"\nSample text chunk:\n{df['text_chunk'].iloc[0]}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure raw data exists at the expected path.")
