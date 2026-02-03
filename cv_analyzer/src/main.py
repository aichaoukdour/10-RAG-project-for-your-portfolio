from extractor import extract_text_from_pdf
from cleaner import clean_cv_text_advanced
from summarizer_llama import summarize_cv_llama
from json_extractor_llama import extract_cv_json_llama
from matcher_llama import compute_skill_match
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

    # --- Skill Matching Feature ---
    print("\n[*] --- Skill Matching Analysis ---")
    sample_jd = """
    We are looking for a Senior AI Engineer with experience in LLMs, RAG systems, and MLOps. 
    Required skills include Python, PyTorch, Docker, and experience with cloud platforms like AWS or GCP.
    Nice to have: Experience with agentic workflows and fine-tuning models.
    """
    
    print("[*] Comparing CV with Job Description...")
    match_result = compute_skill_match(cleaned_text, sample_jd)
    
    print("[*] Match Score:", match_result.get("match_score", "N/A"), "/ 100")
    print("[*] Matched Skills:", ", ".join(match_result.get("matched_skills", [])))
    print("[*] Missing Skills:", ", ".join(match_result.get("missing_skills", [])))
    print("[*] Extra Skills:", ", ".join(match_result.get("extra_skills", [])))

    print("\n[*] Done! Extraction completed.")
    return cv_json, match_result




if __name__ == "__main__":
    main()
