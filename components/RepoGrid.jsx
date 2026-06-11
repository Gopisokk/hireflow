'use client';

const LANG_COLORS = {
  JavaScript:'#f7df1e', TypeScript:'#3178c6', Python:'#3572A5',
  Java:'#b07219', 'C++':'#f34b7d', C:'#555555', 'C#':'#178600',
  Rust:'#dea584', Go:'#00ADD8', Ruby:'#701516', PHP:'#4F5D95',
  Swift:'#F05138', Kotlin:'#A97BFF', HTML:'#e34c26', CSS:'#563d7c',
  Shell:'#89e051', Dart:'#00B4AB', Scala:'#c22d40', R:'#198CE7',
  Vue:'#41b883', 'Jupyter Notebook':'#DA5B0B', SCSS:'#c6538c',
  Makefile:'#427819', PowerShell:'#012456',
};

export function langColor(lang) {
  return LANG_COLORS[lang] || '#64748b';
}

function fmtDate(d) {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

function fmtBytes(b) {
  if (!b) return '?';
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return Math.round(b / 1024) + ' KB';
  return (b / 1024 / 1024).toFixed(1) + ' MB';
}

export default function RepoGrid({ repos, totalCount }) {
  return (
    <>
      <p style={{ color: 'var(--text2)', fontSize: 13, marginBottom: 16 }}>
        Showing <strong style={{ color: 'var(--text)' }}>{repos.length}</strong> of{' '}
        <strong style={{ color: 'var(--text)' }}>{totalCount}</strong> repositories (most recently updated)
      </p>
      <div className="repo-grid">
        {repos.map((r) => {
          const commits = r.defaultBranchRef?.target?.history?.totalCount ?? '?';
          const topics  = r.repositoryTopics?.nodes?.map(t => t.topic.name) ?? [];
          const langs   = r.languages?.edges ?? [];

          return (
            <div className="repo-card" key={r.name}>
              <div className="repo-card-header">
                <div className="repo-name">
                  <a href={r.url} target="_blank" rel="noopener noreferrer">{r.name}</a>
                </div>
                <span className={r.isFork ? 'fork-badge' : 'original-badge'}>
                  {r.isFork ? '⑂ Fork' : '✦ Original'}
                </span>
              </div>

              {r.isFork && r.parent && (
                <div className="repo-parent">from {r.parent.nameWithOwner}</div>
              )}

              <div className="repo-desc">{r.description || 'No description'}</div>

              <div className="repo-tags">
                {langs.slice(0, 3).map(e => (
                  <span
                    key={e.node.name}
                    className="lang-pill"
                    style={{
                      background: langColor(e.node.name) + '22',
                      color: langColor(e.node.name),
                      border: `1px solid ${langColor(e.node.name)}44`,
                    }}
                  >
                    {e.node.name}
                  </span>
                ))}
                {topics.slice(0, 2).map(t => (
                  <span key={t} className="topic-pill">{t}</span>
                ))}
              </div>

              <div className="repo-stats">
                <span>⭐ {r.stargazerCount}</span>
                <span>⑂ {r.forkCount}</span>
                <span>📝 {typeof commits === 'number' ? commits.toLocaleString() : commits} commits</span>
                <span>💾 {fmtBytes(r.diskUsage ? r.diskUsage * 1024 : null)}</span>
              </div>

              <div className="repo-dates">
                Created {fmtDate(r.createdAt)} · Updated {fmtDate(r.updatedAt)}
                {r.isArchived && <span style={{ color: 'var(--yellow)', marginLeft: 8 }}>📦 Archived</span>}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}
