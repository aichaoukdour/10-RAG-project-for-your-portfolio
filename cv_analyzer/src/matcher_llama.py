from llm_client import query_llm
import json
import re

def compute_skill_match(cv_text: str, job_description: str) -> dict:
    """
    Compares CV text with a Job Description and returns a JSON report.
    """
    prompt = f"""
    You are an expert recruiter AI assistant.

    Task: Compare the candidate's CV with the following job description and compute a skill match score.

    Instructions:
    1. Analyze the CV and extract the candidate’s key skills and tools.
    2. Compare these skills with the job description.
    3. Compute a matching score from 0 to 100.
    4. Provide a summary of matched, missing, and extra skills.

    Output format (JSON):
    {{
      "match_score": <integer 0-100>,
      "matched_skills": ["list of skills found in both CV and job description"],
      "missing_skills": ["list of skills in job description but not in CV"],
      "extra_skills": ["skills in CV not in job description"]
    }}

    Candidate CV: 
    {cv_text}

    Job Description:
    {job_description}

    Respond ONLY with valid JSON.
    """
    
    response = query_llm(prompt, max_tokens=1024)
    
    try:
        # Extract JSON block from potential surrounding text
        match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = match.group(0) if match else response
        return json.loads(json_str)
    except Exception as e:
        return {
            "error": "Failed to parse matching results",
            "details": str(e),
            "raw": response[:500]
        }
