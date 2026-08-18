# 🚀 HireFlow-Lite: AI-Powered ATS & Candidate Verification Platform

HireFlow-Lite is an end-to-end Applicant Tracking System (ATS) that parses candidate resumes using a local LLM (**Qwen 2.5 1.5B via Ollama**), indexes candidates using hybrid vector search (**MiniLM + BM25 FTS5**), and verifies candidate projects against their public GitHub profile.

---

## 🏗️ System Architecture

```
PDF / DOCX Resume
       │
       ▼
 PyMuPDF / docx Text Extractor
       │
       ▼
 Plain Resume Text (Truncated to 3.5k chars)
       │
       ▼
 Qwen 2.5 1.5B-Instruct (Ollama on GPU)
       │
       ▼
 Structured Canonical JSON (Candidate, Skills, Projects, Education)
       │
       ▼
 Deterministic Python Evidence Validator
       │
       ▼
 ┌──────────────────────────────────────┬──────────────────────────────────────┐
 ▼                                      ▼                                      ▼
MiniLM Vector Embedding                SQLite Database (FTS5 + vec)           GitHub Profile Verifier
(Hybrid RRF Search)                    (Persistent Storage)                   (Free REST API / GraphQL)
```

---

## 📋 Prerequisites

Before setting up, make sure your machine has:

1. **Python 3.10 or 3.11** ([Download Python](https://www.python.org/downloads/))
2. **Git** ([Download Git](https://git-scm.com/downloads))
3. **Ollama** ([Download Ollama](https://ollama.com/download))

> 💡 **GPU Note:** An NVIDIA GPU (4GB+ VRAM, e.g., GTX 1650 or RTX 2060+) is recommended for fast LLM parsing, but the system will automatically fall back to CPU if no CUDA GPU is detected.

---

## ⚡ Quick Setup Guide (Any Laptop)

### Step 1: Install & Pull Ollama Model

Install Ollama, open your terminal (or Command Prompt), and run:

```bash
ollama pull qwen2.5:1.5b-instruct
```

*Verify Ollama is running in the background.*

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/Gopisokk/hireflow.git
cd hireflow
```

---

### Step 3: Create & Activate Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Start the HireFlow Server

```bash
python server.py
```

You should see:
```text
════════════════════════════════════════════════════════════
  HireFlow-Lite ATS Server v4.0
  GPU    : NVIDIA GeForce GTX 1650 (4.0 GB)
  Device : cuda
  DB     : hireflow.db
  URL    : http://localhost:8000
════════════════════════════════════════════════════════════
```

Now open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🖥️ How to Use the Application

The web interface is divided into 3 main workflows:

### 1️⃣ Phase 1: Upload & Ingest Resumes
- **Student CSV**: Upload a `.csv` file containing candidate details.
  - Required column headers: `roll_number`, `name`, `github_url`, `github_token` (optional).
- **Resumes ZIP**: Upload a `.zip` file containing resume files (`.pdf` or `.docx`).
  - Filename format: File stem must contain or start with the candidate's `roll_number` (e.g. `23AD044.pdf` or `23AD044 - Gopi.docx`).
- Click **Start Ingestion**. Live status logs will show each resume being parsed sequentially.

### 2️⃣ Phase 2: JD Search & Candidate Ranking
- Paste a **Job Description** (JD) in the text box.
- Select maximum results (`Top K`).
- Click **Run JD Search**. Candidates are scored and ranked based on:
  - Skill match (Required & Preferred)
  - Project relevance
  - Vector similarity (MiniLM) + BM25 keyword matching (RRF Fusion)

### 3️⃣ Phase 3: Quick Score & GitHub Verification
- Paste a candidate's resume text and their **GitHub URL**.
- Click **Quick Score + GitHub Verify**.
- The system verifies claimed projects against candidate's public repositories using the GitHub REST API (60 free requests/hr fallback).

---

## 📁 File Structure

```text
hireflow-lite/
├── server.py              # FastAPI server & background job runner
├── ingest.py              # Ingestion pipeline (CSV + ZIP -> LLM parse -> DB)
├── search.py              # Hybrid RRF search & ATS candidate scoring
├── qwen_resume_parser.py  # Ollama Qwen2.5 1.5B LLM resume parser
├── minilm.py              # MiniLM embedding encoder (pure transformers)
├── document_extractor.py  # PyMuPDF PDF & python-docx text extractor
├── ats_engine.py          # 8-component evidence aggregation scoring engine
├── github_verifier.py     # GitHub profile & repository verifier
├── project_verifier.py    # SBERT project similarity matching
├── database.py            # SQLite database schema (FTS5 + vec)
├── ui/
│   └── index.html         # Flat single-page web UI
├── requirements.txt       # Python package dependencies
└── README.md              # Documentation & setup guide
```

---

## 🔧 Troubleshooting & FAQ

<details>
<summary><b>Q: Ingestion feels slow or freezes?</b></summary>
Ollama runs LLM inference sequentially on GPU to prevent VRAM deadlocks. Each resume takes ~10–25 seconds. A live heartbeat log updates every 8 seconds in the UI.
</details>

<details>
<summary><b>Q: GitHub API rate limits?</b></summary>
Without a GitHub Personal Access Token, GitHub allows 60 free API calls/hour. If candidate tokens are provided in the CSV, HireFlow uses them automatically for higher rate limits.
</details>

<details>
<summary><b>Q: No GPU / CPU Mode?</b></summary>
HireFlow automatically detects CUDA. If no GPU is present, it runs safely on CPU.
</details>

---

## 📄 License
MIT License. Built for HireFlow-Lite ATS Project.
