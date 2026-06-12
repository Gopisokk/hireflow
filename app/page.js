'use client';

import { useState } from 'react';
import ConfigPanel from '../components/ConfigPanel';

const FEATURES = [
  { icon: '📄', title: 'Smart ATS Parsing', desc: 'BM25, Neural, and Hybrid algorithms parse resumes against your JD with semantic understanding.' },
  { icon: '🐙', title: 'GitHub Verification', desc: 'Automated verification of candidate repos — fork detection, active days, commit history analysis.' },
  { icon: '📊', title: 'Intelligent Ranking', desc: 'Weighted scoring combining ATS and GitHub signals for fair, data-driven candidate ranking.' },
  { icon: '🔍', title: 'Clone Detection', desc: 'Identifies forked repositories and AI-generated code patterns to ensure candidate authenticity.' },
  { icon: '⚡', title: 'Real-time Pipeline', desc: 'Watch candidates get processed in real-time with SSE-powered progress tracking.' },
  { icon: '🎯', title: 'Configurable Weights', desc: 'Fine-tune ATS vs GitHub scoring weights to match your hiring priorities.' },
];

export default function HomePage() {
  const [showConfig, setShowConfig] = useState(false);
  const [pipelineStarted, setPipelineStarted] = useState(false);

  const handlePipelineStart = (runId) => {
    setPipelineStarted(true);
    // Navigate to dashboard
    window.location.href = '/dashboard';
  };

  return (
    <div className="landing-page">
      {/* Animated background */}
      <div className="landing-bg">
        <div className="landing-gradient-1" />
        <div className="landing-gradient-2" />
        <div className="landing-particles">
          {Array.from({ length: 20 }).map((_, i) => (
            <div key={i} className="particle" style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 8}s`,
              animationDuration: `${6 + Math.random() * 8}s`,
            }} />
          ))}
        </div>
      </div>

      {/* Hero */}
      <section className="landing-hero">
        <div className="landing-hero-badge">⚡ Automated Developer Hiring Platform</div>
        <h1 className="landing-hero-title">
          <span className="landing-hero-gradient">HireFlow</span>
          <span className="landing-hero-sub">Hire Smarter. Verify Deeper.</span>
        </h1>
        <p className="landing-hero-desc">
          Upload candidate data, let AI score resumes against your JD, automatically verify GitHub profiles,
          and get a ranked shortlist — all in one pipeline.
        </p>
        <div className="landing-hero-actions">
          <button className="btn-hero-primary" onClick={() => setShowConfig(true)}>
            🚀 Start Hiring Pipeline
          </button>
          <a href="/dashboard" className="btn-hero-secondary">
            📊 View Dashboard
          </a>
        </div>
        <div className="landing-hero-stats">
          <div className="hero-stat">
            <span className="hero-stat-value">4</span>
            <span className="hero-stat-label">Pipeline Stages</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">4</span>
            <span className="hero-stat-label">AI Algorithms</span>
          </div>
          <div className="hero-stat-divider" />
          <div className="hero-stat">
            <span className="hero-stat-value">∞</span>
            <span className="hero-stat-label">Candidates</span>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="landing-features">
        <h2 className="landing-section-title">Why HireFlow?</h2>
        <div className="features-grid">
          {FEATURES.map(f => (
            <div className="feature-card" key={f.title}>
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline Diagram */}
      <section className="landing-pipeline">
        <h2 className="landing-section-title">How It Works</h2>
        <div className="pipeline-flow">
          {[
            { icon: '📤', title: 'Upload', desc: 'CSV + Resumes + JD' },
            { icon: '📄', title: 'Parse', desc: 'Extract & match skills' },
            { icon: '📝', title: 'ATS Score', desc: 'BM25 / Neural / Hybrid' },
            { icon: '🐙', title: 'Verify GitHub', desc: 'Repos, commits, forks' },
            { icon: '🏆', title: 'Rank', desc: 'Weighted final scores' },
          ].map((step, i) => (
            <div className="pipeline-step" key={step.title}>
              <div className="pipeline-step-num">{i + 1}</div>
              <div className="pipeline-step-icon">{step.icon}</div>
              <div className="pipeline-step-title">{step.title}</div>
              <div className="pipeline-step-desc">{step.desc}</div>
              {i < 4 && <div className="pipeline-step-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      {/* Navigation links */}
      <section className="landing-nav-cards">
        <h2 className="landing-section-title">Quick Navigation</h2>
        <div className="nav-cards-grid">
          <a href="/dashboard" className="nav-card">
            <span className="nav-card-icon">📊</span>
            <span className="nav-card-title">Dashboard</span>
            <span className="nav-card-desc">View candidate rankings and scores</span>
          </a>
          <a href="/explorer" className="nav-card">
            <span className="nav-card-icon">🔍</span>
            <span className="nav-card-title">Explorer</span>
            <span className="nav-card-desc">Deep dive into any GitHub profile</span>
          </a>
          <a href="/repo" className="nav-card">
            <span className="nav-card-icon">📦</span>
            <span className="nav-card-title">Repo Analyzer</span>
            <span className="nav-card-desc">Analyze repo commit history & forks</span>
          </a>
        </div>
      </section>

      {/* Config Panel Modal */}
      {showConfig && (
        <div className="config-overlay" onClick={(e) => {
          if (e.target.className === 'config-overlay') setShowConfig(false);
        }}>
          <div className="config-modal-container">
            <div className="config-modal-header">
              <h2>🚀 Pipeline Configuration</h2>
              <button className="modal-close" onClick={() => setShowConfig(false)}>✕</button>
            </div>
            <ConfigPanel
              onStartPipeline={handlePipelineStart}
            />
          </div>
        </div>
      )}
    </div>
  );
}
