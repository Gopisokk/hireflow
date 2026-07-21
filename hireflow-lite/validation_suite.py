import os
import sys

# Ensure imports work from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ats_engine import run_ats

ML_JD = """We are hiring an AI Engineer to build intelligent applications powered by Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Agentic AI systems.
Required Skills: Python, FastAPI, LangChain, LLMs, Generative AI, Prompt Engineering, RAG, Vector Databases, REST APIs, Git
Preferred Skills: Hugging Face, PyTorch, TensorFlow, Docker, AWS, Agentic AI, MCP, Open Source Contributions
Qualifications: Strong knowledge of NLP and Transformer architectures. Experience building AI applications from concept to deployment."""

RESUMES = {
    "A. ML Engineer": {
        "text": """Experienced AI Engineer with 4 years of experience.
        Built multiple RAG pipelines using LangChain and Vector Databases.
        Developed backend REST APIs with FastAPI in Python.
        Experience fine-tuning LLMs with Hugging Face and PyTorch.
        Deployed models on AWS using Docker. Designed Agentic AI systems.
        Education: MS in Computer Science.""",
        "skills": ["Python", "FastAPI", "LangChain", "LLMs", "Generative AI", "Prompt Engineering", "RAG", "Vector Databases", "REST APIs", "Git", "Hugging Face", "PyTorch", "Docker", "AWS", "Agentic AI"],
        "expected_min": 80,
        "expected_max": 100
    },
    "B. Python Backend Developer": {
        "text": """Backend Developer with 5 years of Python experience.
        Strong expertise in FastAPI, Django, and Flask.
        Built scalable REST APIs and microservices deployed via Docker and AWS.
        Used Git for version control. No direct ML or LLM experience, but familiar with integrating external APIs.
        Education: BS in Computer Science.""",
        "skills": ["Python", "FastAPI", "Django", "Flask", "REST APIs", "Docker", "AWS", "Git", "Microservices"],
        "expected_min": 45,
        "expected_max": 75
    },
    "C. Java Full Stack Developer": {
        "text": """Senior Java Developer with 8 years of experience.
        Built enterprise web applications using Java, Spring Boot, and Hibernate.
        Frontend development with React.js and JavaScript.
        Database management with MySQL and PostgreSQL.
        Used Docker and AWS for deployment, and Git for source control.
        Wrote extensive test suites using JUnit.""",
        "skills": ["Java", "Spring Boot", "Hibernate", "Microservices", "MySQL", "PostgreSQL", "React.js", "JavaScript", "Docker", "AWS", "Git", "JUnit"],
        "expected_min": 10,
        "expected_max": 40
    },
    "D. Cybersecurity Engineer": {
        "text": """Information Security Analyst.
        Conducted penetration testing, vulnerability assessments, and risk mitigation.
        Proficient with Kali Linux, Wireshark, Metasploit, and SIEM tools.
        Certified Ethical Hacker (CEH) and CISSP.
        Strong knowledge of network security protocols and firewalls.""",
        "skills": ["Penetration Testing", "Vulnerability Assessment", "Kali Linux", "Wireshark", "Metasploit", "SIEM", "CEH", "CISSP", "Network Security", "Firewalls"],
        "expected_min": 0,
        "expected_max": 30
    }
}

def run_suite():
    print("=================================================================")
    print("  ATS SCORING VALIDATION SUITE")
    print("=================================================================\n")
    
    passed = 0
    total = len(RESUMES)
    
    for name, data in RESUMES.items():
        print(f"Testing Profile: {name}")
        print(f"Expected Range : {data['expected_min']} - {data['expected_max']}")
        
        result = run_ats(
            resume_text=data["text"],
            jd_text=ML_JD,
            resume_skills=data["skills"],
            algo="hybrid_efficient",
            device="cpu"
        )
        
        ats_base_score = result["score"]
        
        # 10% Project Relevance Heuristic (pre-verification)
        project_score = 10.0 # Mocked for validation
        
        # 10% Experience/Education Heuristic
        exp_score = 10.0 # Mocked for validation
        
        score = ats_base_score + project_score + exp_score
        
        status = "✅ PASS" if data['expected_min'] <= score <= data['expected_max'] else "❌ FAIL"
        
        print(f"Actual Score   : {score:.1f} / 100")
        print(f"Matched Skills : {len(result.get('matched_skills', []))} | Missing Skills: {len(result.get('missing_skills', []))}")
        print(f"Status         : {status}\n")
        
        if "✅ PASS" in status:
            passed += 1
            
    print("=================================================================")
    print(f"  RESULTS: {passed}/{total} Passed")
    print("=================================================================")
    
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
