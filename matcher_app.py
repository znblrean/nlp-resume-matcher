import json
import spacy
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple

# Load the large English NLP model
nlp = spacy.load("en_core_web_lg")

def load_data() -> Tuple[Dict, Dict]:
    """Load resumes and job opportunities from JSON files."""
    with open("resumes.json", "r") as f:
        resumes = json.load(f)
    
    with open("job_opportunities.json", "r") as f:
        jobs = json.load(f)
    
    return resumes, jobs

def preprocess_text(text: str) -> List[str]:
    """Extract keywords from text using NLP."""
    doc = nlp(text.lower())
    keywords = [
        token.text for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return keywords

def get_embedding_vector(keywords: List[str]) -> np.ndarray:
    """Convert keywords to their embedding vectors and return mean vector."""
    vectors = [nlp(keyword).vector for keyword in keywords]
    return np.mean(vectors, axis=0) if vectors else np.zeros((nlp.meta["vectors"]["width"],))

def calculate_similarity(job_vector: np.ndarray, resume_vector: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors (lower is more similar)."""
    return np.linalg.norm(job_vector - resume_vector)

def find_best_matches(job_desc: str, resumes: Dict) -> List[Tuple[str, float]]:
    """Find top 3 matching resumes for a job description."""
    job_keywords = preprocess_text(job_desc)
    job_vector = get_embedding_vector(job_keywords)
    
    similarities = []
    for resume_id, resume_data in resumes.items():
        resume_text = f"{resume_data['skills']} {resume_data['experience']}"
        resume_keywords = preprocess_text(resume_text)
        resume_vector = get_embedding_vector(resume_keywords)
        
        similarity = calculate_similarity(job_vector, resume_vector)
        similarities.append((resume_id, similarity))
    
    # Sort by similarity (ascending order for Euclidean distance)
    similarities.sort(key=lambda x: x[1])
    return similarities[:3]

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="NLP Resume-to-Job Matcher", layout="wide")
    
    st.title("🧠 NLP Resume-to-Job Matcher")
    st.markdown("""
        This app matches **job opportunities** to **resumes** using spaCy embeddings and vector similarity.
        Select a job from the dropdown to see the top 3 matching resumes.
    """)
    
    # Load data
    resumes, jobs = load_data()
    
    # Job selection dropdown
    job_titles = [job["title"] for job in jobs.values()]
    selected_job_title = st.selectbox("Select a Job Opportunity:", job_titles)
    
    # Find selected job details
    selected_job = next(job for job in jobs.values() if job["title"] == selected_job_title)
    
    if st.button("Find Matching Resumes"):
        with st.spinner("Finding best matches..."):
            top_matches = find_best_matches(selected_job["description"], resumes)
        
        st.success("Found top 3 matching resumes!")
        st.subheader(f"Job: {selected_job_title}")
        st.write(selected_job["description"])
        
        st.subheader("Top 3 Matching Resumes:")
        
        for i, (resume_id, similarity_score) in enumerate(top_matches, 1):
            resume = resumes[resume_id]
            
            with st.expander(f"Match #{i} (Score: {similarity_score:.2f}): {resume['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Skills:**")
                    st.write(resume["skills"])
                    
                with col2:
                    st.markdown("**Experience:**")
                    st.write(resume["experience"])
                
                st.markdown(f"**Similarity Score:** `{similarity_score:.2f}`")

if __name__ == "__main__":
    main()

    import re
re.escape("your_text")  # برای escaping کاراکترهای خاص

# مثال در کد:
st.markdown(re.escape("""
این یک متن **Markdown** است با کاراکترهای خاص!
"""))