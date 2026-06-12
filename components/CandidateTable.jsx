'use client';

import { useState, useMemo } from 'react';

const STAGE_CONFIG = {
  pending:      { label: 'Pending',      color: '#64748b', bg: 'rgba(100,116,139,0.12)' },
  parsed:       { label: 'Parsed',       color: '#3b82f6', bg: 'rgba(59,130,246,0.12)' },
  shortlisted:  { label: 'Shortlisted',  color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
  verified:     { label: 'Verified',     color: '#10b981', bg: 'rgba(16,185,129,0.12)' },
  ranked:       { label: 'Ranked',       color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
};

const COLUMNS = [
  { key: 'rank',         label: '#',            width: '56px',  sortable: true },
  { key: 'name',         label: 'Name',         width: '1fr',   sortable: true },
  { key: 'roll_number',  label: 'Roll Number',  width: '120px', sortable: true },
  { key: 'ats_score',    label: 'ATS Score',     width: '130px', sortable: true },
  { key: 'github_score', label: 'GitHub Score',  width: '130px', sortable: true },
  { key: 'final_score',  label: 'Final Score',   width: '130px', sortable: true },
  { key: 'stage',        label: 'Stage',         width: '120px', sortable: true },
];

const PAGE_SIZE = 25;

function getScoreColor(score) {
  if (score >= 80) return '#10b981';
  if (score >= 60) return '#f59e0b';
  return '#ef4444';
}

function ScoreCell({ score }) {
  if (score === null || score === undefined) {
    return <span className="score-cell score-cell-na">—</span>;
  }
  const color = getScoreColor(score);
  const pct = Math.min(score, 100);
  return (
    <div className="score-cell">
      <div className="score-bar-bg">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-value" style={{ color }}>{score.toFixed(1)}</span>
    </div>
  );
}

function StageBadge({ stage }) {
  const config = STAGE_CONFIG[stage] || STAGE_CONFIG.pending;
  return (
    <span
      className="stage-badge"
      style={{ color: config.color, background: config.bg, borderColor: config.color + '40' }}
    >
      {config.label}
    </span>
  );
}

export default function CandidateTable({ candidates = [], onCandidateClick, loading }) {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState('final_score');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
    setPage(0);
  };

  const filtered = useMemo(() => {
    let result = [...candidates];

    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(c =>
        (c.name || '').toLowerCase().includes(q) ||
        (c.roll_number || '').toLowerCase().includes(q)
      );
    }

    result.sort((a, b) => {
      let aVal = a[sortKey];
      let bVal = b[sortKey];
      if (typeof aVal === 'string') aVal = aVal.toLowerCase();
      if (typeof bVal === 'string') bVal = bVal.toLowerCase();
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;
      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [candidates, search, sortKey, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="data-table-container">
      {/* Search */}
      <div className="table-toolbar">
        <div className="table-search">
          <span className="table-search-icon">🔍</span>
          <input
            type="text"
            placeholder="Search by name or roll number..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            className="table-search-input"
          />
          {search && (
            <button className="table-search-clear" onClick={() => setSearch('')}>✕</button>
          )}
        </div>
        <div className="table-info">
          {filtered.length} candidate{filtered.length !== 1 ? 's' : ''}
          {search && ` matching "${search}"`}
        </div>
      </div>

      {/* Table */}
      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr className="table-header">
              {COLUMNS.map(col => (
                <th
                  key={col.key}
                  className={`table-th ${col.sortable ? 'table-th-sortable' : ''} ${sortKey === col.key ? 'table-th-active' : ''}`}
                  style={{ width: col.width }}
                  onClick={() => col.sortable && handleSort(col.key)}
                >
                  <span>{col.label}</span>
                  {sortKey === col.key && (
                    <span className="sort-indicator">{sortDir === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="table-row table-row-loading">
                  {COLUMNS.map(col => (
                    <td key={col.key}><div className="shimmer-cell" /></td>
                  ))}
                </tr>
              ))
            ) : paginated.length === 0 ? (
              <tr>
                <td colSpan={COLUMNS.length} className="table-empty">
                  {search ? 'No candidates match your search.' : 'No candidates to display.'}
                </td>
              </tr>
            ) : (
              paginated.map((c, i) => (
                <tr
                  key={c.roll_number || i}
                  className="table-row"
                  onClick={() => onCandidateClick?.(c)}
                >
                  <td className="table-rank">
                    <span className="rank-badge">{c.rank || page * PAGE_SIZE + i + 1}</span>
                  </td>
                  <td className="table-name">{c.name || '—'}</td>
                  <td className="table-roll">{c.roll_number || '—'}</td>
                  <td><ScoreCell score={c.ats_score} /></td>
                  <td><ScoreCell score={c.github_score} /></td>
                  <td><ScoreCell score={c.final_score} /></td>
                  <td><StageBadge stage={c.stage} /></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="table-pagination">
          <button
            className="pagination-btn"
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            ← Prev
          </button>
          <div className="pagination-pages">
            {Array.from({ length: Math.min(totalPages, 7) }).map((_, i) => {
              let pageNum;
              if (totalPages <= 7) {
                pageNum = i;
              } else if (page < 3) {
                pageNum = i;
              } else if (page > totalPages - 4) {
                pageNum = totalPages - 7 + i;
              } else {
                pageNum = page - 3 + i;
              }
              return (
                <button
                  key={pageNum}
                  className={`pagination-num ${page === pageNum ? 'pagination-active' : ''}`}
                  onClick={() => setPage(pageNum)}
                >
                  {pageNum + 1}
                </button>
              );
            })}
          </div>
          <button
            className="pagination-btn"
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
