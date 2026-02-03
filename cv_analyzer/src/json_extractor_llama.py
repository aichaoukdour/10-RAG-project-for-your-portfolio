from llm_client import query_llm
import json

def extract_cv_json_llama(text: str) -> dict:
    prompt = f"""
    Extract the following CV into JSON format.
    Keys: name, contact, email, linkedin, github, skills (list), experience (list of role, company, period, description), tools, summary

    CV Text:
    {text}

    Respond ONLY with valid JSON.
    """
    # Increase max_tokens for full JSON extraction
    response = query_llm(prompt, max_tokens=2048)
    
    # Advanced JSON extraction: Find the first '{' and last '}'
    import re
    try:
        match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
        if match:
            clean_response = match.group(1)
            return json.loads(clean_response)
        else:
            return json.loads(response) # Try original as fallback
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_response": response}


