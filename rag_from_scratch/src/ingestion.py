import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from chunks import create_text_chunks
from config import (
    RAW_SALARIES_PATH, PROCESSED_SALARIES_PATH,
    EXPERIENCE_MAPPING, EMPLOYMENT_MAPPING, COMPANY_SIZE_MAPPING,
)

logger = logging.getLogger(__name__)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    path = file_path or str(RAW_SALARIES_PATH)
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
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


def save_processed_data(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    path = Path(output_path or str(PROCESSED_SALARIES_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path} ({len(df)} records)")
    return str(path)


def run_ingestion(raw_path: Optional[str] = None, processed_path: Optional[str] = None) -> pd.DataFrame:
    logger.info("--- Starting Data Ingestion Pipeline ---")

    df = load_data(raw_path)
    df = clean_data(df)
    df = create_text_chunks(df)
    save_processed_data(df, processed_path)

    logger.info("--- Data Ingestion Complete ---")
    return df


if __name__ == "__main__":
     run_ingestion()