'use client';

import { useEffect, useRef, useCallback } from 'react';

/* ─── Score Gauge (Canvas) ──────────────────────────────── */
function ScoreGauge({ score, label, color, size = 110 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + 'px';
    canvas.style.height = size + 'px';
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const radius = (size - 16) / 2;
    const lineWidth = 8;
    const startAngle = Math.PI * 0.75;
    const endAngle = Math.PI * 2.25;
    const pct = Math.min((score || 0) / 100, 1);
    const sweepAngle = startAngle + (endAngle - startAngle) * pct;

    ctx.clearRect(0, 0, size, size);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Foreground arc
    if (score !== null && score !== undefined) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius, startAngle, sweepAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, sweepAngle - 0.05, sweepAngle);
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Score text
    ctx.fillStyle = '#e2e8f0';
    ctx.font = `bold ${size * 0.24}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(score !== null && score !== undefined ? score.toFixed(1) : '—', cx, cy - 4);

    // Label
    ctx.fillStyle = '#64748b';
    ctx.font = `600 ${size * 0.1}px Inter, sans-serif`;
    ctx.fillText(label, cx, cy + size * 0.18);
  }, [score, label, color, size]);

  return <canvas ref={canvasRef} className="score-gauge" />;
}

/* ─── Skill Pill ──────────────────────────────────────────── */
function SkillPill({ skill, matched }) {
  return (
    <span className={`skill-pill ${matched ? 'skill-matched' : 'skill-missing'}`}>
      {matched ? '✓' : '✗'} {skill}
    </span>
  );
}

/* ─── Project Row ──────────────────────────────────────────── */
function ProjectRow({ project }) {
  return (
    <tr className="project-row">
      <td className="project-name">{project.name || '—'}</td>
      <td>
        <span className={`source-badge source-${project.source}`}>
          {project.source === 'github' ? '🐙 GitHub' : '📄 Resume'}
        </span>
      </td>
      <td>
        {project.verified ? (
          <span className="verified-badge verified-yes">✓ Verified</span>
        ) : (
          <span className="verified-badge verified-no">✗ Not Found</span>
        )}
      </td>
      <td>
        {project.is_fork ? (
          <span className="fork-indicator">⑂ Fork</span>
        ) : (
          <span className="original-indicator">✦ Original</span>
        )}
      </td>
    </tr>
  );
}

/* ─── Main Modal ──────────────────────────────────────────── */
export default function CandidateModal({ candidate, onClose }) {
  const modalRef = useRef(null);

  const handleEsc = useCallback((e) => {
    if (e.key === 'Escape') onClose();
  }, [onClose]);

  useEffect(() => {
    document.addEventListener('keydown', handleEsc);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handleEsc);
      document.body.style.overflow = '';
    };
  }, [handleEsc]);

  const handleBackdrop = (e) => {
    if (e.target === modalRef.current) onClose();
  };

  if (!candidate) return null;

  const ats = candidate.ats_breakdown || {};
  const github = candidate.github_breakdown || {};
  const projects = candidate.projects || [];
  const matchedSkills = ats.matched_skills || [];
  const missingSkills = ats.missing_skills || [];

  return (
    <div className="modal-overlay" ref={modalRef} onClick={handleBackdrop}>
      <div className="modal-content">
        {/* Close button */}
        <button className="modal-close" onClick={onClose}>✕</button>

        {/* Header */}
        <div className="modal-header">
          <div className="modal-header-bg" />
          <div className="modal-header-content">
            <div className="modal-avatar">
              <span className="modal-avatar-text">
                {(candidate.name || '?').charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="modal-header-info">
              <h2 className="modal-name">{candidate.name || 'Unknown'}</h2>
              <div className="modal-meta">
                <span className="modal-roll">🎓 {candidate.roll_number || '—'}</span>
                {candidate.github_url && (
                  <a
                    href={candidate.github_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="modal-github-link"
                  >
                    🐙 GitHub Profile →
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Score Gauges */}
        <div className="modal-gauges">
          <ScoreGauge score={candidate.ats_score} label="ATS" color="#6366f1" />
          <ScoreGauge score={candidate.github_score} label="GitHub" color="#06b6d4" />
          <ScoreGauge score={candidate.final_score} label="Final" color="#10b981" size={130} />
        </div>

        {/* ATS Breakdown */}
        <div className="modal-section">
          <h3 className="modal-section-title">📝 ATS Breakdown</h3>
          <div className="modal-section-body">
            <div className="ats-scores-row">
              {ats.bm25_score !== undefined && (
                <div className="ats-score-item">
                  <span className="ats-score-label">BM25 Score</span>
                  <span className="ats-score-value">{ats.bm25_score?.toFixed(2) ?? '—'}</span>
                </div>
              )}
              {ats.semantic_score !== undefined && (
                <div className="ats-score-item">
                  <span className="ats-score-label">Semantic Score</span>
                  <span className="ats-score-value">{ats.semantic_score?.toFixed(2) ?? '—'}</span>
                </div>
              )}
              {ats.keyword_match !== undefined && (
                <div className="ats-score-item">
                  <span className="ats-score-label">Keyword Match</span>
                  <span className="ats-score-value">{ats.keyword_match?.toFixed(1) ?? '—'}%</span>
                </div>
              )}
            </div>
            <div className="skills-section">
              <h4 className="skills-heading">Matched Skills</h4>
              <div className="skills-grid">
                {matchedSkills.length > 0 ? matchedSkills.map(s => (
                  <SkillPill key={s} skill={s} matched={true} />
                )) : <span className="no-data">No matched skills data</span>}
              </div>
            </div>
            <div className="skills-section">
              <h4 className="skills-heading">Missing Skills</h4>
              <div className="skills-grid">
                {missingSkills.length > 0 ? missingSkills.map(s => (
                  <SkillPill key={s} skill={s} matched={false} />
                )) : <span className="no-data">No missing skills data</span>}
              </div>
            </div>
          </div>
        </div>

        {/* GitHub Breakdown */}
        <div className="modal-section">
          <h3 className="modal-section-title">🐙 GitHub Breakdown</h3>
          <div className="modal-section-body">
            <div className="github-stats-grid">
              <div className="github-stat-card">
                <span className="github-stat-icon">📅</span>
                <span className="github-stat-val">{github.active_days ?? '—'}</span>
                <span className="github-stat-label">Active Days</span>
              </div>
              <div className="github-stat-card">
                <span className="github-stat-icon">⑂</span>
                <span className="github-stat-val">{github.fork_ratio !== undefined ? (github.fork_ratio * 100).toFixed(0) + '%' : '—'}</span>
                <span className="github-stat-label">Fork Ratio</span>
              </div>
              <div className="github-stat-card">
                <span className="github-stat-icon">📝</span>
                <span className="github-stat-val">{github.total_commits ?? '—'}</span>
                <span className="github-stat-label">Total Commits</span>
              </div>
              <div className="github-stat-card">
                <span className="github-stat-icon">⭐</span>
                <span className="github-stat-val">{github.total_stars ?? '—'}</span>
                <span className="github-stat-label">Total Stars</span>
              </div>
            </div>

            {/* Language alignment */}
            {github.languages && Object.keys(github.languages).length > 0 && (
              <div className="lang-alignment">
                <h4 className="skills-heading">Language Alignment</h4>
                {Object.entries(github.languages).sort(([,a],[,b]) => b - a).slice(0, 8).map(([lang, pct]) => (
                  <div className="lang-align-row" key={lang}>
                    <span className="lang-align-name">{lang}</span>
                    <div className="lang-align-bar-bg">
                      <div className="lang-align-bar-fill" style={{ width: `${Math.min(pct, 100)}%` }} />
                    </div>
                    <span className="lang-align-pct">{typeof pct === 'number' ? pct.toFixed(1) : pct}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Verified projects */}
            {github.verified_projects && github.verified_projects.length > 0 && (
              <div className="verified-projects">
                <h4 className="skills-heading">Verified Projects</h4>
                <div className="verified-list">
                  {github.verified_projects.map((p, i) => (
                    <span key={i} className="verified-project-pill">✓ {p}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Projects Table */}
        {projects.length > 0 && (
          <div className="modal-section">
            <h3 className="modal-section-title">📋 Projects</h3>
            <div className="modal-section-body">
              <table className="projects-table">
                <thead>
                  <tr>
                    <th>Project Name</th>
                    <th>Source</th>
                    <th>Verified</th>
                    <th>Fork Status</th>
                  </tr>
                </thead>
                <tbody>
                  {projects.map((p, i) => <ProjectRow key={i} project={p} />)}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
