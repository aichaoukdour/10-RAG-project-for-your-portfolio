import pandas as pd
from typing import Callable
from config import REMOTE_RATIO_MAPPING, setup_logging

# Setup module logger
logger = setup_logging(__name__)


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
