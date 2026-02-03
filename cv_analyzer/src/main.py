from extractor import extract_text_from_pdf
from cleaner import clean_cv_text_advanced
from summarizer_llama import summarize_cv_llama
from json_extractor_llama import extract_cv_json_llama
import os

DATA_FOLDER = "./data"
FALLBACK_PDF = "./data/cv.pdf"

def find_first_pdf(folder: str, fallback: str) -> str:
    import glob
    pdf_files = glob.glob(os.path.join(folder, "*.pdf"))
    return pdf_files[0] if pdf_files else fallback


def main():
    pdf_path = find_first_pdf(DATA_FOLDER, FALLBACK_PDF)
    print(f"[*] Starting processing for: {pdf_path}")

    print("[*] Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    print("[*] Cleaning text...")
    cleaned_text = clean_cv_text_advanced(raw_text)

    print("[*] Generating summary with LLaMA...")
    # Summarize
    summary = summarize_cv_llama(cleaned_text)

    # Add summary before JSON extraction
    cleaned_text_with_summary = cleaned_text + "\n\nSummary:\n" + summary

    print("[*] Extracting structured JSON with LLaMA...")
    # Extract structured JSON
    cv_json = extract_cv_json_llama(cleaned_text_with_summary)

    print("[*] Done! Result:")
    import json
    print(json.dumps(cv_json, indent=4, ensure_ascii=False))

    return cv_json



if __name__ == "__main__":
    main()
