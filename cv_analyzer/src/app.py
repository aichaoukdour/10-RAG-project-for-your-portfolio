import streamlit as st
import os
import json
import tempfile
from extractor import extract_text_from_pdf
from cleaner import clean_cv_text_advanced
from summarizer_llama import summarize_cv_llama
from json_extractor_llama import extract_cv_json_llama
from matcher_llama import compute_skill_match

# Page Configuration
st.set_page_config(
    page_title="AI CV Analyzer & Matcher",
    page_icon="📄",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        margin: 4px;
        font-size: 0.9em;
        font-weight: 500;
    }
    .matched { background-color: #d1fae5; color: #065f46; border: 1px solid #34d399; }
    .missing { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .extra { background-color: #e0f2fe; color: #075985; border: 1px solid #38bdf8; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🚀 AI CV Analyzer & Job Matcher")
    st.markdown("Upload your CV and the job description to get AI-powered insights.")

    # Sidebar for Inputs
    with st.sidebar:
        st.header("📋 Inputs")
        uploaded_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
        
        st.markdown("---")
        job_description = st.text_area("Job Description", height=300, placeholder="Paste the job description here...")
        
        analyze_button = st.button("🔍 Analyze CV", type="primary", use_container_width=True)

    if analyze_button:
        if not uploaded_file:
            st.error("Please upload a CV PDF file.")
            return
        if not job_description.strip():
            st.warning("Please provide a Job Description for skill matching.")
            # We can still proceed with just extraction if JD is missing, but skill matching won't work well
            
        with st.spinner("Processing CV... (This involves LLM calls, please wait)"):
            try:
                # 1. Save uploaded file to temp path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 2. Extract and Clean
                raw_text = extract_text_from_pdf(tmp_path)
                cleaned_text = clean_cv_text_advanced(raw_text)
                
                # Cleanup temp file
                os.unlink(tmp_path)

                # 3. Parallel Processing (Conceptual/Sequential here for simplicity)
                # Generate Summary
                summary = summarize_cv_llama(cleaned_text)
                
                # Extract JSON
                cv_json = extract_cv_json_llama(cleaned_text + "\n\nSummary:\n" + summary)
                
                # Compute Skill Match
                match_result = {}
                if job_description.strip():
                    match_result = compute_skill_match(cleaned_text, job_description)

                # --- Display Results ---
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📝 Professional Summary")
                    st.info(summary if summary else "No summary generated.")

                    st.subheader("👤 Candidate Info")
                    if isinstance(cv_json, dict) and "error" not in cv_json:
                        st.json(cv_json)
                    else:
                        st.warning("Could not structure CV data into JSON.")
                        st.text(cv_json.get("raw_response", "") if isinstance(cv_json, dict) else str(cv_json))

                with col2:
                    st.subheader("🎯 Skill Match Analysis")
                    if match_result and "error" not in match_result:
                        score = match_result.get("match_score", 0)
                        
                        # Big Score Metric
                        st.metric("Compatibility Score", f"{score}%")
                        st.progress(score / 100)

                        # Skills Visualization
                        st.markdown("#### Matched Skills")
                        if match_result.get("matched_skills"):
                            for s in match_result["matched_skills"]:
                                st.markdown(f'<span class="skill-tag matched">{s}</span>', unsafe_allow_html=True)
                        else:
                            st.write("No direct matches found.")

                        st.markdown("#### Missing Skills")
                        if match_result.get("missing_skills"):
                            for s in match_result["missing_skills"]:
                                st.markdown(f'<span class="skill-tag missing">{s}</span>', unsafe_allow_html=True)
                        else:
                            st.write("No missing skills identified.")

                        st.markdown("#### Extra Skills")
                        if match_result.get("extra_skills"):
                            for s in match_result["extra_skills"]:
                                st.markdown(f'<span class="skill-tag extra">{s}</span>', unsafe_allow_html=True)
                        else:
                            st.write("No additional skills highlighted.")
                    else:
                        st.info("Paste a Job Description in the sidebar to see the skill match analysis.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
