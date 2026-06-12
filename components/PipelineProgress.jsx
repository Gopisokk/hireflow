'use client';

import { useState, useEffect, useRef } from 'react';

const STAGES = [
  { id: 'parsing',      icon: '📄', label: 'Parsing' },
  { id: 'ats_scoring',  icon: '📝', label: 'ATS Scoring' },
  { id: 'github_verify', icon: '🔍', label: 'GitHub Verification' },
  { id: 'ranking',      icon: '🏆', label: 'Ranking' },
];

export default function PipelineProgress({ isRunning, pipelineRunId }) {
  const [currentStage, setCurrentStage] = useState('');
  const [progress, setProgress] = useState(0);
  const [totalCandidates, setTotalCandidates] = useState(0);
  const [processedCandidates, setProcessedCandidates] = useState(0);
  const [currentCandidate, setCurrentCandidate] = useState('');
  const [startTime, setStartTime] = useState(null);
  const [elapsed, setElapsed] = useState('00:00');
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState('');
  const eventSourceRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isRunning) return;

    setStartTime(Date.now());
    setCompleted(false);
    setError('');

    // Connect to SSE
    const url = pipelineRunId
      ? `/api/backend/pipeline/status?run_id=${pipelineRunId}`
      : '/api/backend/pipeline/status';

    try {
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.stage) setCurrentStage(data.stage);
          if (data.progress !== undefined) setProgress(data.progress);
          if (data.total !== undefined) setTotalCandidates(data.total);
          if (data.processed !== undefined) setProcessedCandidates(data.processed);
          if (data.current_candidate) setCurrentCandidate(data.current_candidate);
          if (data.stage === 'completed') {
            setCompleted(true);
            setProgress(100);
            es.close();
          }
          if (data.error) {
            setError(data.error);
            es.close();
          }
        } catch {
          // ignore parse errors
        }
      };

      es.onerror = () => {
        // SSE connection closed or errored — could be normal end
        es.close();
      };
    } catch {
      // SSE not available - poll instead
    }

    return () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, [isRunning, pipelineRunId]);

  // Timer
  useEffect(() => {
    if (!startTime || completed) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = setInterval(() => {
      const diff = Math.floor((Date.now() - startTime) / 1000);
      const mins = String(Math.floor(diff / 60)).padStart(2, '0');
      const secs = String(diff % 60).padStart(2, '0');
      setElapsed(`${mins}:${secs}`);
    }, 1000);

    return () => clearInterval(timerRef.current);
  }, [startTime, completed]);

  const getStageIndex = () => STAGES.findIndex(s => s.id === currentStage);

  if (!isRunning && !completed && !error) {
    return (
      <div className="pipeline-idle">
        <div className="pipeline-idle-icon">⏸️</div>
        <div className="pipeline-idle-text">Pipeline is idle. Configure and start to begin processing.</div>
      </div>
    );
  }

  return (
    <div className={`pipeline-progress ${completed ? 'pipeline-completed' : ''}`}>
      {/* Stage pills */}
      <div className="pipeline-stages">
        {STAGES.map((stage, i) => {
          const stageIdx = getStageIndex();
          let status = 'pending';
          if (completed) status = 'done';
          else if (i < stageIdx) status = 'done';
          else if (i === stageIdx) status = 'active';

          return (
            <div key={stage.id} className={`stage-pill stage-${status}`}>
              <span className="stage-pill-icon">{stage.icon}</span>
              <span className="stage-pill-label">{stage.label}</span>
              {status === 'done' && <span className="stage-check">✓</span>}
              {i < STAGES.length - 1 && <span className="stage-connector" />}
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }}>
          <div className="progress-glow" />
        </div>
        <span className="progress-label">{Math.round(progress)}%</span>
      </div>

      {/* Stats row */}
      <div className="pipeline-stats-row">
        {currentCandidate && (
          <div className="pipeline-stat">
            <span className="pipeline-stat-label">Processing</span>
            <span className="pipeline-stat-value pipeline-stat-candidate">{currentCandidate}</span>
          </div>
        )}
        <div className="pipeline-stat">
          <span className="pipeline-stat-label">Candidates</span>
          <span className="pipeline-stat-value">{processedCandidates} / {totalCandidates || '—'}</span>
        </div>
        <div className="pipeline-stat">
          <span className="pipeline-stat-label">Elapsed</span>
          <span className="pipeline-stat-value pipeline-stat-time">{elapsed}</span>
        </div>
        {completed && (
          <div className="pipeline-stat pipeline-stat-complete">
            <span className="pipeline-stat-value">✅ Complete</span>
          </div>
        )}
        {error && (
          <div className="pipeline-stat pipeline-stat-error">
            <span className="pipeline-stat-value">❌ {error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
