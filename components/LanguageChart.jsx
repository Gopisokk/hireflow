'use client';
import { langColor } from './RepoGrid';

function fmtBytes(b) {
  if (b >= 1_000_000) return (b / 1_000_000).toFixed(1) + ' MB';
  if (b >= 1_000)     return (b / 1_000).toFixed(0) + ' KB';
  return b + ' B';
}

export default function LanguageChart({ repos }) {
  // Aggregate bytes per language across all repos
  const totals = {};
  repos.forEach(r => {
    (r.languages?.edges || []).forEach(e => {
      totals[e.node.name] = (totals[e.node.name] || 0) + e.size;
    });
  });

  const sorted = Object.entries(totals).sort((a, b) => b[1] - a[1]);
  const total  = sorted.reduce((s, [, v]) => s + v, 0);
  const top    = sorted.slice(0, 15);

  if (!top.length) {
    return <p style={{ color: 'var(--muted)', fontSize: 13 }}>No language data available.</p>;
  }

  return (
    <div>
      {/* Stacked bar */}
      <div style={{
        display: 'flex',
        height: 10,
        borderRadius: 5,
        overflow: 'hidden',
        marginBottom: 20,
        gap: 1,
      }}>
        {top.map(([lang, bytes]) => (
          <div
            key={lang}
            style={{
              flex: bytes,
              background: langColor(lang),
            }}
            title={`${lang}: ${((bytes / total) * 100).toFixed(1)}%`}
          />
        ))}
      </div>

      {/* Bar rows */}
      {top.map(([lang, bytes]) => {
        const pct = total ? ((bytes / total) * 100).toFixed(1) : 0;
        return (
          <div className="lang-row" key={lang}>
            <div className="lang-name">
              <span style={{
                display: 'inline-block',
                width: 10, height: 10,
                borderRadius: '50%',
                background: langColor(lang),
                marginRight: 7,
                verticalAlign: 'middle',
              }} />
              {lang}
            </div>
            <div className="lang-bar-bg">
              <div
                className="lang-bar-fill"
                style={{ width: `${pct}%`, background: langColor(lang) }}
              />
            </div>
            <div className="lang-pct">{pct}%</div>
            <div className="lang-bytes">{fmtBytes(bytes)}</div>
          </div>
        );
      })}

      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 16 }}>
        Total code: <strong style={{ color: 'var(--text2)' }}>{fmtBytes(total)}</strong>
        {' '}across <strong style={{ color: 'var(--text2)' }}>{repos.length}</strong> repositories
      </p>
    </div>
  );
}
