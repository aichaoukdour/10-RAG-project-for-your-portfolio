import logging
import os
from typing import Callable
import pandas as pd
from config import (RAW_SALARIES_PATH, PROCESSED_SALARIES_PATH)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


EXPERIENCE_MAPPING = {'EN': 'Entry-level', 'MI': 'Mid-level', 
                      'SE': 'Senior-level', 'EX':'Executive-level'}

EMPLOYMENT_MAPPING = {'FT': 'Full-time', 'PT': 'Part-time',
                      'CT': 'Contract', 'FL': 'Freelance'}

COMPANY_SIZE_MAPPING = {'S': 'small', 'M': 'medium', 'L': 'large'}

REMOTE_RATIO_MAPPING = {100: 'remote', 50: 'hybrid', 0: 'on-site'}


def load_data(file_path: str | None = None) -> pd.DataFrame:
    path = file_path or str(RAW_SALARIES_PATH)
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records")
    return df



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    removed = df.duplicated().sum()
    if removed:
        logger.info(f"Removed {removed} duplicate records")

    for col, mapping in {
        'experience_level': EXPERIENCE_MAPPING,
        'employment_type': EMPLOYMENT_MAPPING,
        'company_size': COMPANY_SIZE_MAPPING
    }.items():
        if col in df:
            df[col] = df[col].map(mapping).fillna(df[col])

    logger.info(f"Data cleaning complete: {len(df)} records")
    return df


def _serialize_row(row: pd.Series) -> str:
    salary = row.get("salary_in_usd", 0)
    salary_str = f"{salary:,}" if isinstance(salary, (int, float)) else str(salary)

    return (
        f"In {row.get('work_year', 'unknown year')}, "
        f"a {row.get('experience_level', '')} {row.get('job_title', 'professional')} "
        f"working {row.get('employment_type', '')} in {row.get('employee_residence', 'unknown location')} "
        f"earned a salary of {salary_str} USD. "
        f"The role was {REMOTE_RATIO_MAPPING.get(row.get('remote_ratio', 0), 'on-site')} "
        f"for a {row.get('company_size', 'unknown')}-sized company "
        f"located in {row.get('company_location', 'unknown location')}."
    )



def create_text_chunks(
    df: pd.DataFrame,
    chunk_fn: Callable[[pd.Series], str] | None = None
) -> pd.DataFrame:
    chunk_function = chunk_fn or _serialize_row
    df = df.copy() 

    logger.info(f"Creating text chunks for {len(df)} records...")
    df["text_chunk"] = df.apply(chunk_function, axis=1)

    if not df.empty:
        logger.debug(f"Sample chunk: {df['text_chunk'].iloc[0][:100]}...")

    return df



def save_processed_data(df: pd.DataFrame, output_path: str = None) -> str:
    path = output_path or str(PROCESSED_SALARIES_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path} ({len(df)} records)")
    return path


def run_ingestion(raw_path: str = None, processed_path: str = None) -> pd.DataFrame:
    logger.info("--- Starting Data Ingestion Pipeline ---")
    
    df = load_data(raw_path)
    df = clean_data(df)
    df = create_text_chunks(df)
    save_processed_data(df, processed_path)
    
    logger.info("--- Data Ingestion Complete ---")
    return df


if __name__ == "__main__":
     run_ingestion()
   