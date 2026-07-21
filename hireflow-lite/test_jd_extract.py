import re

def extract_jd_keywords(jd_text: str) -> dict:
    STOP_SKILLS = {"developer", "candidate", "service", "api", "system", "application", "project", "experience", "engineer", "work", "team", "skills", "knowledge", "ability", "responsibilities"}
    
    # Simple regex to split JD into Required and Preferred if possible
    required_text = jd_text
    preferred_text = ""
    
    pref_match = re.search(r"(preferred|nice to have|bonus) skills?:?(.*?)(?:\n\n|\Z)", jd_text, re.IGNORECASE | re.DOTALL)
    if pref_match:
        preferred_text = pref_match.group(2)
        # remove preferred text from required_text to avoid overlap
        required_text = required_text.replace(pref_match.group(0), "")
        
    req_match = re.search(r"(required|core|must have) skills?:?(.*?)(?:\n\n|\Z)", required_text, re.IGNORECASE | re.DOTALL)
    if req_match:
        required_text = req_match.group(2)
        
    def extract_from_text(text):
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            return []
        
        doc = nlp(text)
        tech_keywords = {"python", "java", "c++", "fastapi", "langchain", "llms", "genai", "spring boot", "hibernate", "microservices", "mysql", "postgresql", "react.js", "javascript", "docker", "aws", "git", "junit"}
        seen = set()
        keywords = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space: continue
            lower = token.lemma_.lower()
            if lower in STOP_SKILLS: continue
            if token.pos_ in ("NOUN", "PROPN") or lower in tech_keywords:
                if lower not in seen and len(lower) > 1:
                    seen.add(lower)
                    keywords.append(lower)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if chunk_text.startswith("a "): chunk_text = chunk_text[2:]
            if chunk_text.startswith("the "): chunk_text = chunk_text[4:]
            if chunk_text in STOP_SKILLS: continue
            if chunk_text not in seen and len(chunk_text.split()) > 1:
                if not any(stop in chunk_text.split() for stop in STOP_SKILLS):
                    seen.add(chunk_text)
                    keywords.append(chunk_text)
        return keywords

    req_kws = extract_from_text(required_text)
    pref_kws = extract_from_text(preferred_text)
    all_kws = list(set(req_kws + pref_kws))
    
    return {"required": req_kws, "preferred": pref_kws, "all": all_kws}

if __name__ == "__main__":
    jd = \"\"\"Position: Machine Learning Engineer
Required Skills: Python, NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Machine Learning, Deep Learning, Data Preprocessing, Feature Engineering, Model Evaluation, SQL, Git
Preferred Skills: LangChain, LLMs, RAG, FastAPI, Hugging Face Transformers, Vector Databases, AWS, GCP, Azure, MLOps
Responsibilities: Build and train ML/DL models, data cleaning, feature engineering, NLP, Computer Vision, fine-tune transformers, deploy with FastAPI and Docker.
Experience: 0-3 years. Strong knowledge of Statistics, Probability, Linear Algebra.\"\"\"
    print(extract_jd_keywords(jd))
