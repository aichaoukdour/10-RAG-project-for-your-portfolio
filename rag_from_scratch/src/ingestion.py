import pandas as pd
import os

# Mapping for experience levels
EXPERIENCE_MAPPING = {
    'EN': 'Entry-level',
    'MI': 'Mid-level',
    'SE': 'Senior-level',
    'EX': 'Executive-level'
}

# Mapping for employment types
EMPLOYMENT_MAPPING = {
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
}

def load_data(file_path):
    """Load the raw salary data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data not found at {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the data: remove duplicates and map codes to full names."""
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Map experience levels
    df['experience_level'] = df['experience_level'].map(EXPERIENCE_MAPPING).fillna(df['experience_level'])
    
    # Map employment types
    df['employment_type'] = df['employment_type'].map(EMPLOYMENT_MAPPING).fillna(df['employment_type'])
    
    return df

def create_text_chunks(df):
    """Convert each row into a descriptive sentence for RAG."""
    def row_to_text(row):
        remote_status = "remote" if row['remote_ratio'] == 100 else "hybrid" if row['remote_ratio'] == 50 else "on-site"
        return (f"In {row['work_year']}, a {row['experience_level']} {row['job_title']} "
                f"working {row['employment_type']} in {row['employee_residence']} "
                f"earned a salary of {row['salary_in_usd']:,} USD. "
                f"The role was {remote_status} for a {row['company_size']}-sized company "
                f"located in {row['company_location']}.")

    df['text_chunk'] = df.apply(row_to_text, axis=1)
    return df

def save_processed_data(df, output_path):
    """Save the cleaned and chunked data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
