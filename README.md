# HireFlow — Automated Developer Hiring Platform

A high-throughput, automated developer hiring platform that replaces manual resume screening with an AI-powered pipeline combining BM25/SBERT ATS scoring and GitHub profile verification.

## 🏗️ Architecture

```
hireflow/
├── backend/          # FastAPI + Python ML pipeline
│   ├── main.py       # FastAPI app entry point
│   ├── config.py     # Settings
│   ├── database/     # SQLite + sqlite-vec schema & CRUD
│   ├── pipeline/     # Parser, embedder, ATS scorer, GitHub verifier
│   └── routers/      # API route handlers
│
├── app/              # Next.js App Router (frontend)
│   ├── page.js       # Landing / Configuration page
│   ├── dashboard/    # HR Results Dashboard
│   ├── explorer/     # GitHub Profile Explorer
│   ├── repo/         # Repository Deep Dive
│   └── api/          # Next.js API routes (GitHub proxy)
│
└── components/       # Shared React components
    ├── Sidebar.jsx
    ├── ConfigPanel.jsx
    ├── PipelineProgress.jsx
    ├── CandidateTable.jsx
    ├── CandidateModal.jsx
    └── ScoreChart.jsx
```

## 🚀 Running the Platform

### Prerequisites
- Node.js ≥ 18
- Python ≥ 3.11
- pip

### 1. Start the Backend (FastAPI)

```bash
cd backend

# Create virtual environment (first time)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies (first time)
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```

Backend will be available at: **http://localhost:8000**
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 2. Start the Frontend (Next.js)

```bash
# From the hireflow root directory
npm install   # first time only
npm run dev
```

Frontend will be available at: **http://localhost:3000**

---

## 📋 How to Use

### Step 1 — Prepare Your Data
Create a CSV with these columns:
```csv
roll_number,name,github_username,resume_filename
CS001,Alice Johnson,alicejohnson,alice_johnson.pdf
CS002,Bob Smith,bobsmith99,bob_smith.pdf
```

Place all resume PDFs/DOCXs in a single folder.

### Step 2 — Configure Pipeline
1. Open **http://localhost:3000**
2. Click **"🚀 Start Hiring Pipeline"**
3. Upload your CSV
4. Enter the resume folder path
5. Paste the Job Description
6. Set ATS/GitHub weight ratio
7. Choose scoring algorithm
8. Enter GitHub Personal Access Token (optional, for GitHub verification)

### Step 3 — View Results
Navigate to **http://localhost:3000/dashboard** to see:
- Real-time pipeline progress
- Ranked candidates with scores
- Click any candidate for detailed score breakdown

---

## 🧠 Scoring Algorithms

| Algorithm | Description | Speed |
|-----------|-------------|-------|
| Classic BM25 | TF-IDF keyword matching | ⚡⚡⚡ |
| Neural Fast | Sentence-BERT cosine similarity | ⚡⚡ |
| Hybrid Efficient | BM25 pre-filter → SBERT re-rank | ⚡⚡ |

## 📊 Pipeline Stages

1. **Parse** — Extract text from PDF/DOCX resumes using PyMuPDF and python-docx
2. **ATS Score** — Score against JD using selected algorithm
3. **Shortlist** — Keep top 10% for GitHub verification
4. **GitHub Verify** — Check repos, commits, forks, active days, language alignment
5. **Rank** — Merge ATS + GitHub scores with configured weights

## 🔑 GitHub Token Setup

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `read:user`, `public_repo`
4. Copy the token (starts with `ghp_...`)

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16 (App Router), Vanilla CSS |
| Charts | Chart.js + react-chartjs-2 |
| Backend | FastAPI + Python 3.11 |
| Database | SQLite + sqlite-vec (vector search) |
| ML/NLP | sentence-transformers (all-MiniLM-L6-v2) |
| Lexical | rank-bm25 |
| Resume Parsing | PyMuPDF + python-docx |
| GitHub API | GraphQL + httpx |
| Streaming | SSE (sse-starlette) |
