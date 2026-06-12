'use client';

import { useState, useRef } from 'react';

const ALGORITHMS = [
  { id: 'bm25',     name: 'Classic BM25',       desc: 'Traditional keyword matching with TF-IDF weighting. Fast and reliable.' },
  { id: 'neural',   name: 'Neural Fast',        desc: 'Sentence-transformer embeddings for semantic similarity. GPU accelerated.' },
  { id: 'hybrid',   name: 'Hybrid Efficient',   desc: 'BM25 + Neural combined scoring. Best accuracy-speed tradeoff.' },
  { id: 'colbert',  name: 'SOTA ColBERT',       desc: 'Late interaction model for SOTA accuracy. Slower but most precise.' },
];

export default function ConfigPanel({ onConfigChange, onStartPipeline }) {
  const [csvFile, setCsvFile] = useState(null);
  const [resumeFolder, setResumeFolder] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [atsWeight, setAtsWeight] = useState(60);
  const [algorithm, setAlgorithm] = useState('hybrid');
  const [githubToken, setGithubToken] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);

  const githubWeight = 100 - atsWeight;
  const jdLineCount = jobDescription.split('\n').length;

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length && files[0].name.endsWith('.csv')) {
      setCsvFile(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files.length) {
      setCsvFile(e.target.files[0]);
    }
  };

  const handleStart = async () => {
    if (!csvFile) return alert('Please upload a CSV file.');
    if (!jobDescription.trim()) return alert('Please enter a Job Description.');

    setIsUploading(true);
    try {
      // Step 1: Update pipeline config (weights + algorithm)
      const configRes = await fetch('/api/backend/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ats_weight: atsWeight / 100,
          github_weight: githubWeight / 100,
          algorithm: algorithm === 'bm25' ? 'classic_bm25' :
                     algorithm === 'neural' ? 'neural_fast' :
                     algorithm === 'hybrid' ? 'hybrid_efficient' : 'hybrid_efficient',
        }),
      });
      // Config update is best-effort; continue even if it fails
      if (!configRes.ok) {
        console.warn('Config update warning:', await configRes.text());
      }

      // Step 2: Upload batch CSV to seed the database
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      if (resumeFolder) formData.append('resume_folder', resumeFolder);
      formData.append('job_description', jobDescription);

      const uploadRes = await fetch('/api/backend/batch/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json().catch(() => ({}));
        throw new Error(err.detail || `Upload failed: HTTP ${uploadRes.status}`);
      }

      // Step 3: Start the pipeline
      const startRes = await fetch('/api/backend/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_description: jobDescription,
          github_token: githubToken || null,
          resume_folder: resumeFolder || null,
        }),
      });

      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({}));
        throw new Error(err.detail || `Pipeline start failed: HTTP ${startRes.status}`);
      }

      const startData = await startRes.json();
      if (onStartPipeline) onStartPipeline(startData.run_id || null);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const triggerChange = () => {
    if (onConfigChange) {
      onConfigChange({ csvFile, resumeFolder, jobDescription, atsWeight, githubWeight, algorithm, githubToken });
    }
  };

  return (
    <div className="config-panel">
      {/* CSV Upload */}
      <div className="config-section">
        <h3 className="config-section-title">📁 Candidate Data</h3>
        <div
          className={`upload-zone ${isDragging ? 'upload-zone-active' : ''} ${csvFile ? 'upload-zone-done' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          {csvFile ? (
            <div className="upload-zone-content">
              <span className="upload-zone-icon">✅</span>
              <span className="upload-zone-filename">{csvFile.name}</span>
              <span className="upload-zone-size">{(csvFile.size / 1024).toFixed(1)} KB</span>
            </div>
          ) : (
            <div className="upload-zone-content">
              <span className="upload-zone-icon">📄</span>
              <span className="upload-zone-text">Drop CSV file here or click to browse</span>
              <span className="upload-zone-hint">Supports .csv with Name, Roll Number, GitHub URL columns</span>
            </div>
          )}
        </div>
      </div>

      {/* Resume Folder */}
      <div className="config-section">
        <div className="field">
          <label>📂 Resume Folder Path</label>
          <input
            type="text"
            placeholder="/path/to/resume/folder"
            value={resumeFolder}
            onChange={(e) => { setResumeFolder(e.target.value); triggerChange(); }}
          />
        </div>
      </div>

      {/* Job Description */}
      <div className="config-section">
        <h3 className="config-section-title">📝 Job Description</h3>
        <div className="jd-container">
          <div className="jd-line-numbers">
            {Array.from({ length: jdLineCount }, (_, i) => (
              <span key={i}>{i + 1}</span>
            ))}
          </div>
          <textarea
            className="jd-textarea"
            placeholder="Paste the full job description here...&#10;&#10;Required Skills:&#10;- Python, JavaScript&#10;- React, Node.js&#10;- SQL, MongoDB&#10;&#10;Experience: 2+ years"
            value={jobDescription}
            onChange={(e) => { setJobDescription(e.target.value); triggerChange(); }}
            rows={10}
          />
        </div>
        <div className="jd-stats">
          <span>{jdLineCount} lines</span>
          <span>{jobDescription.length} chars</span>
          <span>{jobDescription.trim().split(/\s+/).filter(Boolean).length} words</span>
        </div>
      </div>

      {/* Weights */}
      <div className="config-section">
        <h3 className="config-section-title">⚖️ Scoring Weights</h3>
        <div className="weight-sliders">
          <div className="weight-slider-group">
            <div className="weight-label">
              <span>ATS Score Weight</span>
              <span className="weight-value">{atsWeight}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={atsWeight}
              onChange={(e) => { setAtsWeight(Number(e.target.value)); triggerChange(); }}
              className="weight-slider"
            />
          </div>
          <div className="weight-slider-group">
            <div className="weight-label">
              <span>GitHub Score Weight</span>
              <span className="weight-value">{githubWeight}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={githubWeight}
              onChange={(e) => { setAtsWeight(100 - Number(e.target.value)); triggerChange(); }}
              className="weight-slider github-slider"
            />
          </div>
          <div className="weight-bar-preview">
            <div className="weight-bar-ats" style={{ width: `${atsWeight}%` }}>
              {atsWeight > 15 && `ATS ${atsWeight}%`}
            </div>
            <div className="weight-bar-github" style={{ width: `${githubWeight}%` }}>
              {githubWeight > 15 && `GitHub ${githubWeight}%`}
            </div>
          </div>
        </div>
      </div>

      {/* Algorithm */}
      <div className="config-section">
        <h3 className="config-section-title">🧠 Algorithm</h3>
        <div className="algo-selector">
          {ALGORITHMS.map(algo => (
            <button
              key={algo.id}
              className={`algo-option ${algorithm === algo.id ? 'algo-active' : ''}`}
              onClick={() => { setAlgorithm(algo.id); triggerChange(); }}
            >
              <span className="algo-name">{algo.name}</span>
              <span className="algo-desc">{algo.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* GitHub Token */}
      <div className="config-section">
        <div className="field">
          <label>🔑 GitHub Token</label>
          <input
            type="password"
            placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
            value={githubToken}
            onChange={(e) => { setGithubToken(e.target.value); triggerChange(); }}
          />
          <span className="hint">Required for GitHub verification. Needs read:user + public_repo scopes.</span>
        </div>
      </div>

      {/* Start Button */}
      <button
        className="btn-pipeline-start"
        onClick={handleStart}
        disabled={isUploading || !csvFile}
      >
        {isUploading ? (
          <><span className="spinner" style={{ width: 16, height: 16 }} /> Starting Pipeline...</>
        ) : (
          '🚀 Start Pipeline'
        )}
      </button>
    </div>
  );
}
