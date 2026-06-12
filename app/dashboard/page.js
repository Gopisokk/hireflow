'use client';

import { useState, useEffect, useMemo } from 'react';
import CandidateTable from '../../components/CandidateTable';
import CandidateModal from '../../components/CandidateModal';
import PipelineProgress from '../../components/PipelineProgress';
import ScoreChart from '../../components/ScoreChart';

export default function DashboardPage() {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [chartType, setChartType] = useState('distribution');
  const [error, setError] = useState('');

  // Fetch candidates
  const fetchCandidates = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/backend/candidates');
      if (res.ok) {
        const data = await res.json();
        const list = Array.isArray(data) ? data : data.candidates || [];
        // Add rank based on final_score
        const ranked = list
          .sort((a, b) => (b.final_score || 0) - (a.final_score || 0))
          .map((c, i) => ({ ...c, rank: i + 1 }));
        setCandidates(ranked);
      } else {
        setCandidates([]);
      }
    } catch {
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCandidates();
    // Poll every 10 seconds
    const interval = setInterval(fetchCandidates, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch detailed candidate when clicked
  const handleCandidateClick = async (candidate) => {
    try {
      const res = await fetch(`/api/backend/candidates/${candidate.roll_number}`);
      if (res.ok) {
        const detail = await res.json();
        setSelectedCandidate(detail);
      } else {
        setSelectedCandidate(candidate);
      }
    } catch {
      setSelectedCandidate(candidate);
    }
  };

  // Summary stats
  const stats = useMemo(() => {
    const total = candidates.length;
    const shortlisted = candidates.filter(c => c.stage === 'shortlisted' || c.stage === 'verified' || c.stage === 'ranked').length;
    const avgAts = total ? candidates.reduce((s, c) => s + (c.ats_score || 0), 0) / total : 0;
    const avgGithub = total ? candidates.reduce((s, c) => s + (c.github_score || 0), 0) / total : 0;
    const avgFinal = total ? candidates.reduce((s, c) => s + (c.final_score || 0), 0) / total : 0;
    const maxFinal = total ? Math.max(...candidates.map(c => c.final_score || 0)) : 0;

    return { total, shortlisted, avgAts, avgGithub, avgFinal, maxFinal };
  }, [candidates]);

  return (
    <div className="dashboard-page">
      {/* Dashboard Header */}
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">HR Dashboard</h1>
          <p className="dashboard-subtitle">Candidate ranking and analysis results</p>
        </div>
        <div className="dashboard-actions">
          <button className="btn-refresh" onClick={fetchCandidates} disabled={loading}>
            {loading ? '⏳' : '↺'} Refresh
          </button>
        </div>
      </div>

      {/* Pipeline Progress */}
      <div className="dashboard-section">
        <PipelineProgress isRunning={pipelineRunning} />
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card summary-card-total">
          <div className="summary-card-icon">👥</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.total}</div>
            <div className="summary-card-label">Total Processed</div>
          </div>
        </div>
        <div className="summary-card summary-card-shortlisted">
          <div className="summary-card-icon">✅</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.shortlisted}</div>
            <div className="summary-card-label">Shortlisted</div>
          </div>
        </div>
        <div className="summary-card summary-card-ats">
          <div className="summary-card-icon">📝</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.avgAts.toFixed(1)}</div>
            <div className="summary-card-label">Avg ATS Score</div>
          </div>
        </div>
        <div className="summary-card summary-card-github">
          <div className="summary-card-icon">🐙</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.avgGithub.toFixed(1)}</div>
            <div className="summary-card-label">Avg GitHub Score</div>
          </div>
        </div>
        <div className="summary-card summary-card-final">
          <div className="summary-card-icon">🏆</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.avgFinal.toFixed(1)}</div>
            <div className="summary-card-label">Avg Final Score</div>
          </div>
        </div>
        <div className="summary-card summary-card-top">
          <div className="summary-card-icon">⭐</div>
          <div className="summary-card-data">
            <div className="summary-card-value">{stats.maxFinal.toFixed(1)}</div>
            <div className="summary-card-label">Top Score</div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="dashboard-charts">
        <div className="chart-tabs">
          <button
            className={`chart-tab ${chartType === 'distribution' ? 'chart-tab-active' : ''}`}
            onClick={() => setChartType('distribution')}
          >
            📊 Distribution
          </button>
          <button
            className={`chart-tab ${chartType === 'radar' ? 'chart-tab-active' : ''}`}
            onClick={() => setChartType('radar')}
          >
            🎯 Comparison
          </button>
          <button
            className={`chart-tab ${chartType === 'comparison' ? 'chart-tab-active' : ''}`}
            onClick={() => setChartType('comparison')}
          >
            📈 Overview
          </button>
        </div>
        <ScoreChart candidates={candidates} type={chartType} />
      </div>

      {/* Candidates Table */}
      <div className="dashboard-section">
        <h2 className="dashboard-section-title">🏆 Ranked Candidates</h2>
        <CandidateTable
          candidates={candidates}
          onCandidateClick={handleCandidateClick}
          loading={loading}
        />
      </div>

      {/* Candidate Modal */}
      {selectedCandidate && (
        <CandidateModal
          candidate={selectedCandidate}
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </div>
  );
}
