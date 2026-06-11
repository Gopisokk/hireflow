'use client';

import { useState, useCallback, useRef, useEffect } from 'react';

/* ── helpers ──────────────────────────────────────────────── */
function fmtDate(d) {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}
function fmtDateShort(d) {
  if (!d) return '';
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}
function fmtBytes(kb) {
  if (!kb) return '0 B';
  const b = kb * 1024;
  if (b > 1_000_000) return (b / 1_000_000).toFixed(1) + ' MB';
  return (b / 1024).toFixed(0) + ' KB';
}
function relativeTime(d) {
  if (!d) return '';
  const diff = Math.floor((Date.now() - new Date(d)) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const LANG_COLORS = {
  JavaScript:'#f7df1e',TypeScript:'#3178c6',Python:'#3572A5',Java:'#b07219',
  'C++':'#f34b7d',C:'#555',Rust:'#dea584',Go:'#00ADD8',Ruby:'#701516',
  PHP:'#4F5D95',Swift:'#F05138',Kotlin:'#A97BFF',HTML:'#e34c26',CSS:'#563d7c',
  Shell:'#89e051',Dart:'#00B4AB',
};
const lc = l => LANG_COLORS[l] || '#64748b';

/* ── CommitCalendar ───────────────────────────────────────── */
function CommitCalendar({ uniqueDaysList, commits }) {
  const canvasRef = useRef(null);

  // Build a map: date → commit count
  const dayCount = {};
  commits.forEach(c => {
    const d = (c.date || '').slice(0, 10);
    if (d) dayCount[d] = (dayCount[d] || 0) + 1;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !commits.length) return;

    // Determine date range
    const dates = Object.keys(dayCount).sort();
    if (!dates.length) return;

    const start = new Date(dates[0]);
    const end   = new Date(dates[dates.length - 1]);

    // Snap start to Sunday of that week
    start.setDate(start.getDate() - start.getDay());

    const CELL = 13, GAP = 2, STEP = CELL + GAP;
    const LEFT = 28, TOP = 22;

    // Count weeks
    const totalDays = Math.ceil((end - start) / 86400000) + 7;
    const weeks = Math.ceil(totalDays / 7);
    const W = LEFT + weeks * STEP + 10;
    const H = TOP + 7 * STEP + 20;

    canvas.width = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const maxCount = Math.max(...Object.values(dayCount), 1);

    // Day labels
    ['S','M','T','W','T','F','S'].forEach((d, i) => {
      if (i % 2 === 1) {
        ctx.fillStyle = '#475569';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText(d, 6, TOP + i * STEP + CELL - 1);
      }
    });

    // Draw cells
    let lastMonth = -1;
    for (let w = 0; w < weeks; w++) {
      for (let dow = 0; dow < 7; dow++) {
        const cur = new Date(start);
        cur.setDate(cur.getDate() + w * 7 + dow);

        if (cur > end) continue;

        const dateStr = cur.toISOString().slice(0, 10);
        const count   = dayCount[dateStr] || 0;
        const x = LEFT + w * STEP;
        const y = TOP + dow * STEP;

        // Month label
        if (dow === 0) {
          const m = cur.getMonth();
          if (m !== lastMonth) {
            const names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            ctx.fillStyle = '#64748b';
            ctx.font = '9px Inter, sans-serif';
            ctx.fillText(names[m], x, 12);
            lastMonth = m;
          }
        }

        // Color
        let fill = '#161f30';
        if (count > 0) {
          const intensity = count / maxCount;
          if (intensity < 0.25)      fill = '#0e4429';
          else if (intensity < 0.5)  fill = '#006d32';
          else if (intensity < 0.75) fill = '#26a641';
          else                       fill = '#39d353';
        }

        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.roundRect(x, y, CELL, CELL, 2);
        ctx.fill();
      }
    }
  }, [commits]);

  return (
    <div>
      <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
        <canvas ref={canvasRef} style={{ display: 'block' }} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10, fontSize: 11, color: 'var(--muted)' }}>
        <span>Less</span>
        {['#161f30','#0e4429','#006d32','#26a641','#39d353'].map((c, i) => (
          <div key={i} style={{ width: 11, height: 11, borderRadius: 2, background: c }} />
        ))}
        <span>More commits</span>
      </div>
    </div>
  );
}

/* ── MonthlyBarChart ──────────────────────────────────────── */
function MonthlyBarChart({ monthlyActivity }) {
  const entries = Object.entries(monthlyActivity).sort(([a], [b]) => a.localeCompare(b));
  if (!entries.length) return null;
  const max = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 100, overflowX: 'auto', paddingBottom: 4 }}>
      {entries.map(([month, count]) => (
        <div key={month} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flex: '0 0 auto' }}>
          <div style={{ fontSize: 10, color: 'var(--accent)', fontFamily: 'monospace' }}>{count}</div>
          <div style={{
            width: 28,
            height: Math.max((count / max) * 72, 4),
            background: `linear-gradient(180deg, #6366f1, #8b5cf6)`,
            borderRadius: '4px 4px 0 0',
            transition: 'height .4s ease',
          }} title={`${month}: ${count} commits`} />
          <div style={{ fontSize: 9, color: 'var(--muted)', whiteSpace: 'nowrap' }}>
            {month.slice(5)}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Section ──────────────────────────────────────────────── */
function Section({ title, tag, children, accent, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="section" style={accent ? { borderColor: 'rgba(99,102,241,0.3)', boxShadow: '0 0 24px rgba(99,102,241,0.08)' } : {}}>
      <div className="section-head" onClick={() => setOpen(o => !o)}>
        <div className="section-title">
          <span>{title}</span>
          {tag && <span className="endpoint-tag">{tag}</span>}
        </div>
        <span className="chevron" style={{ transform: open ? '' : 'rotate(-90deg)', transition: 'transform .2s' }}>▼</span>
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

/* ── Main Page ────────────────────────────────────────────── */
export default function RepoPage() {
  const [username,  setUsername]  = useState('');
  const [token,     setToken]     = useState('');
  const [repoName,  setRepoName]  = useState('');
  const [loading,   setLoading]   = useState(false);
  const [step,      setStep]      = useState('');
  const [error,     setError]     = useState('');
  const [data,      setData]      = useState(null);
  const [tab,       setTab]       = useState('overview');
  const [commitFilter, setCommitFilter] = useState('');

  const analyze = useCallback(async () => {
    if (!username.trim() || !token.trim() || !repoName.trim()) {
      setError('Please fill in all three fields.');
      return;
    }
    setError(''); setData(null); setLoading(true);
    setStep('Fetching repo metadata…');

    try {
      setStep('Paginating through full commit history (may take a moment)…');
      const res  = await fetch('/api/repo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), token: token.trim(), repoName: repoName.trim() }),
      });
      setStep('Processing & computing stats…');
      const json = await res.json();
      if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false); setStep('');
    }
  }, [username, token, repoName]);

  const handleKey = e => { if (e.key === 'Enter') analyze(); };

  /* derived */
  const repo    = data?.repo;
  const stats   = data?.stats;
  const commits = data?.commits ?? [];
  const fa      = data?.forkAnalysis;

  const filteredCommits = commitFilter
    ? commits.filter(c =>
        c.message?.toLowerCase().includes(commitFilter.toLowerCase()) ||
        c.author?.toLowerCase().includes(commitFilter.toLowerCase()) ||
        c.sha?.includes(commitFilter)
      )
    : commits;

  const tabs = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'commits',     label: `📝 Commits (${commits.length})` },
    { id: 'calendar',    label: '📅 Activity' },
    { id: 'languages',   label: '💻 Languages' },
    { id: 'forks',       label: `🔍 Fork Check (${fa?.forkCount ?? 0} cloned)` },
    { id: 'contributors',label: '👥 Contributors' },
  ];

  return (
    <>
      {/* ── Header ── */}
      <header className="header">
        <a href="/" style={{ textDecoration: 'none' }}>
          <span className="logo">HireFlow</span>
        </a>
        <span className="badge">Repo Deep Dive</span>
        {data && (
          <div className="header-right">
            <span className="rate-badge">
              {repo?.isFork
                ? `⑂ Fork of ${repo.parent}`
                : '✦ Original Repository'}
            </span>
          </div>
        )}
      </header>

      {/* ── Input ── */}
      <div style={{ padding: '48px 24px 32px', maxWidth: 780, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <h1 style={{ fontSize: 'clamp(24px,3.5vw,40px)', fontWeight: 800, letterSpacing: '-0.03em',
            background: 'linear-gradient(135deg,#e2e8f0,#818cf8,#06b6d4)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 8 }}>
            Repository Analyzer
          </h1>
          <p style={{ color: 'var(--text2)', fontSize: 14 }}>
            Full commit history · unique days worked · fork/clone detection across your entire account
          </p>
        </div>

        <div className="glass-card">
          <div className="form-row" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
            <div className="field">
              <label>GitHub Username</label>
              <input id="username" type="text" placeholder="Gopisokk"
                value={username} onChange={e => setUsername(e.target.value)} onKeyDown={handleKey} />
            </div>
            <div className="field">
              <label>Repo Name</label>
              <input id="reponame" type="text" placeholder="funding-bot"
                value={repoName} onChange={e => setRepoName(e.target.value)} onKeyDown={handleKey}
                style={{ fontFamily: 'var(--font-mono,monospace)' }} />
            </div>
            <div className="field">
              <label>Personal Access Token</label>
              <input id="token" type="password" placeholder="ghp_…"
                value={token} onChange={e => setToken(e.target.value)} onKeyDown={handleKey} />
            </div>
          </div>
          <button className="btn-primary" onClick={analyze} disabled={loading}>
            {loading ? `⏳ ${step}` : '🔍 Analyze Repository'}
          </button>
        </div>

        {error && (
          <div className="error-box" style={{ marginTop: 16 }}>
            <span>❌</span>
            <div>
              <strong>Error:</strong> {error}
              {error.includes('404') && <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text2)' }}>
                Repo not found. Check the exact repo name (case-sensitive).
              </div>}
            </div>
          </div>
        )}
      </div>

      {/* ── Results ── */}
      {data && (
        <div className="results" style={{ paddingTop: 0 }}>

          {/* Big stat strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(160px,1fr))', gap: 12, marginBottom: 28 }}>
            {[
              { label: 'Total Commits',   value: stats.totalCommits.toLocaleString(), icon: '📝', color: '#6366f1' },
              { label: 'Unique Days',      value: stats.uniqueDays,                   icon: '📅', color: '#10b981' },
              { label: 'Date Range',       value: `${fmtDateShort(stats.firstCommit)} → ${fmtDateShort(stats.lastCommit)}`, icon: '🗓️', color: '#f59e0b', small: true },
              { label: 'Contributors',     value: data.contributors.length,            icon: '👥', color: '#06b6d4' },
              { label: 'Stars',            value: repo.stars,                          icon: '⭐', color: '#f59e0b' },
              { label: 'Open Issues',      value: repo.openIssues,                     icon: '🐛', color: '#ef4444' },
              { label: 'Repo Size',        value: fmtBytes(repo.size),                 icon: '💾', color: '#8b5cf6', small: true },
              { label: 'Status',           value: repo.isFork ? '⑂ Forked' : '✦ Original', icon: repo.isFork ? '⑂' : '✦', color: repo.isFork ? '#f59e0b' : '#10b981', small: true },
            ].map(s => (
              <div key={s.label} className="stat-card" style={{ borderColor: s.color + '33', background: s.color + '08' }}>
                <div style={{ fontSize: 22, marginBottom: 6 }}>{s.icon}</div>
                <div className="stat-value" style={{ fontSize: s.small ? 16 : 28, backgroundImage: `linear-gradient(135deg,${s.color},${s.color}99)` }}>
                  {s.value}
                </div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Fork alert */}
          {repo.isFork && (
            <div style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)', borderRadius: 12, padding: '14px 20px', marginBottom: 20, display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <span style={{ fontSize: 20 }}>⚠️</span>
              <div>
                <strong style={{ color: '#fbbf24' }}>This repository is a fork</strong>
                <p style={{ color: 'var(--text2)', fontSize: 13, marginTop: 4 }}>
                  Originally from: <a href={repo.parentUrl} target="_blank" rel="noopener noreferrer" style={{ color: '#818cf8' }}>{repo.parent}</a>
                </p>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="tab-bar">
            {tabs.map(t => (
              <button key={t.id} className={`tab-btn ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>

          {/* ── OVERVIEW ── */}
          {tab === 'overview' && (
            <>
              <Section title="Repository Info" tag={`REST /repos/${username}/${repoName}`} accent>
                <table className="kv-table">
                  <tbody>
                    <tr><td>Name</td><td><a href={repo.url} target="_blank" rel="noopener noreferrer" style={{color:'#818cf8'}}>{repo.fullName}</a></td></tr>
                    <tr><td>Description</td><td>{repo.description || <span className="null-val">null</span>}</td></tr>
                    <tr><td>Is Fork?</td><td style={{color: repo.isFork ? '#fbbf24':'#10b981'}}>{String(repo.isFork)} {repo.isFork ? `← forked from ${repo.parent}` : '← original work'}</td></tr>
                    <tr><td>Default Branch</td><td>{repo.defaultBranch}</td></tr>
                    <tr><td>Visibility</td><td>{repo.visibility}</td></tr>
                    <tr><td>License</td><td>{repo.license || <span className="null-val">null</span>}</td></tr>
                    <tr><td>Topics</td><td>{repo.topics.length ? repo.topics.join(', ') : <span className="null-val">none</span>}</td></tr>
                    <tr><td>Created at</td><td>{fmtDate(repo.createdAt)}</td></tr>
                    <tr><td>Last pushed</td><td>{fmtDate(repo.pushedAt)} ({relativeTime(repo.pushedAt)})</td></tr>
                    <tr><td>Stars / Forks / Watchers</td><td>⭐ {repo.stars} · ⑂ {repo.forks} · 👁 {repo.watchers}</td></tr>
                    <tr><td>Open Issues</td><td>{repo.openIssues}</td></tr>
                    <tr><td>Size</td><td>{fmtBytes(repo.size)}</td></tr>
                  </tbody>
                </table>
              </Section>

              <Section title="Work Summary — Days & Commits" accent>
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:20 }}>
                  <div style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:12, padding:20 }}>
                    <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--muted)', marginBottom:12 }}>Commit Timeline</div>
                    <div style={{ fontSize:13, color:'var(--text2)', lineHeight:1.8 }}>
                      <div>📅 First commit: <strong style={{color:'var(--text)'}}>{fmtDate(stats.firstCommit)}</strong></div>
                      <div>🏁 Latest commit: <strong style={{color:'var(--text)'}}>{fmtDate(stats.lastCommit)}</strong></div>
                      <div>📝 Total commits: <strong style={{color:'var(--text)'}}>{stats.totalCommits.toLocaleString()}</strong></div>
                      <div>📅 Days with commits: <strong style={{color:'#10b981', fontSize:16}}>{stats.uniqueDays}</strong></div>
                      <div>📊 Avg commits/active day: <strong style={{color:'var(--text)'}}>{stats.uniqueDays ? (stats.totalCommits / stats.uniqueDays).toFixed(1) : '—'}</strong></div>
                    </div>
                  </div>
                  <div style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:12, padding:20 }}>
                    <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--muted)', marginBottom:12 }}>Monthly Activity</div>
                    <MonthlyBarChart monthlyActivity={stats.monthlyActivity} />
                  </div>
                </div>
              </Section>

              {data.readme && (
                <Section title="README Preview" tag="REST /repos/{owner}/{repo}/readme" defaultOpen={false}>
                  <pre style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:10, padding:16, fontSize:12, color:'var(--text2)', overflowX:'auto', whiteSpace:'pre-wrap', lineHeight:1.65, maxHeight:300, overflowY:'auto' }}>
                    {data.readme}
                  </pre>
                </Section>
              )}
            </>
          )}

          {/* ── COMMITS ── */}
          {tab === 'commits' && (
            <Section title={`All ${stats.totalCommits} Commits`} tag={`REST /repos/${username}/${repoName}/commits (paginated)`}>
              <div style={{ marginBottom:16 }}>
                <input
                  type="text"
                  placeholder="Filter by message, author, or SHA…"
                  value={commitFilter}
                  onChange={e => setCommitFilter(e.target.value)}
                  style={{ width:'100%', background:'var(--surface2)', border:'1px solid var(--border2)', borderRadius:10, color:'var(--text)', fontFamily:'var(--font-mono,monospace)', fontSize:13, padding:'10px 14px', outline:'none' }}
                />
                {commitFilter && <div style={{ fontSize:12, color:'var(--muted)', marginTop:6 }}>Showing {filteredCommits.length} of {commits.length} commits</div>}
              </div>
              <div style={{ maxHeight:520, overflowY:'auto' }}>
                {filteredCommits.map((c, i) => (
                  <div key={i} style={{ display:'flex', gap:14, padding:'10px 0', borderBottom:'1px solid var(--border)', alignItems:'flex-start' }}>
                    <div style={{ fontFamily:'var(--font-mono,monospace)', fontSize:11, color:'#818cf8', background:'rgba(99,102,241,0.1)', border:'1px solid rgba(99,102,241,0.2)', padding:'2px 8px', borderRadius:6, flexShrink:0, marginTop:1 }}>
                      {c.sha}
                    </div>
                    <div style={{ flex:1, minWidth:0 }}>
                      <div style={{ fontSize:13, fontWeight:600, color:'var(--text)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                        <a href={c.url} target="_blank" rel="noopener noreferrer" style={{ color:'inherit', textDecoration:'none' }}>{c.message}</a>
                      </div>
                      <div style={{ fontSize:11, color:'var(--muted)', marginTop:3 }}>
                        {c.author} · {fmtDate(c.date)} · {relativeTime(c.date)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* ── CALENDAR ── */}
          {tab === 'calendar' && (
            <>
              <Section title="Commit Activity Calendar" tag={`${stats.uniqueDays} unique days worked`} accent>
                <CommitCalendar uniqueDaysList={stats.uniqueDaysList} commits={commits} />
                <div style={{ marginTop:20 }}>
                  <div style={{ fontSize:11, fontWeight:700, textTransform:'uppercase', letterSpacing:'0.08em', color:'var(--muted)', marginBottom:12 }}>
                    All {stats.uniqueDays} Days You Committed
                  </div>
                  <div style={{ display:'flex', flexWrap:'wrap', gap:6 }}>
                    {stats.uniqueDaysList.map(d => (
                      <span key={d} style={{ fontSize:11, fontFamily:'var(--font-mono,monospace)', background:'rgba(16,185,129,0.1)', color:'#10b981', border:'1px solid rgba(16,185,129,0.2)', padding:'3px 10px', borderRadius:20 }}>
                        {fmtDate(d)}
                      </span>
                    ))}
                  </div>
                </div>
              </Section>

              <Section title="Monthly Commit Breakdown">
                <MonthlyBarChart monthlyActivity={stats.monthlyActivity} />
                <table className="kv-table" style={{ marginTop:16 }}>
                  <tbody>
                    {Object.entries(stats.monthlyActivity).sort(([a],[b])=>b.localeCompare(a)).map(([m, c]) => (
                      <tr key={m}>
                        <td style={{ fontFamily:'var(--font-mono,monospace)' }}>{m}</td>
                        <td>
                          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                            <div style={{ height:8, width: Math.max((c/Math.max(...Object.values(stats.monthlyActivity)))*200,4), background:'linear-gradient(90deg,#6366f1,#8b5cf6)', borderRadius:4 }} />
                            <span style={{ color:'var(--text)', fontWeight:600 }}>{c} commits</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </>
          )}

          {/* ── LANGUAGES ── */}
          {tab === 'languages' && (
            <Section title="Language Breakdown" tag={`REST /repos/${username}/${repoName}/languages`}>
              {(() => {
                const langs = Object.entries(data.languages).sort(([,a],[,b])=>b-a);
                const total = langs.reduce((s,[,v])=>s+v,0);
                return (
                  <>
                    <div style={{ display:'flex', height:10, borderRadius:5, overflow:'hidden', marginBottom:20, gap:1 }}>
                      {langs.map(([l,b])=>(
                        <div key={l} style={{ flex:b, background:lc(l) }} title={`${l}: ${((b/total)*100).toFixed(1)}%`} />
                      ))}
                    </div>
                    {langs.map(([l,b])=>{
                      const pct = ((b/total)*100).toFixed(1);
                      return (
                        <div className="lang-row" key={l}>
                          <div className="lang-name">
                            <span style={{ display:'inline-block', width:10, height:10, borderRadius:'50%', background:lc(l), marginRight:7, verticalAlign:'middle' }} />
                            {l}
                          </div>
                          <div className="lang-bar-bg">
                            <div className="lang-bar-fill" style={{ width:`${pct}%`, background:lc(l) }} />
                          </div>
                          <div className="lang-pct">{pct}%</div>
                          <div className="lang-bytes">{(b/1024).toFixed(0)} KB</div>
                        </div>
                      );
                    })}
                  </>
                );
              })()}
            </Section>
          )}

          {/* ── FORKS ── */}
          {tab === 'forks' && (
            <>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:12, marginBottom:20 }}>
                {[
                  { label:'Total Repos', value:fa.totalRepos, color:'#6366f1' },
                  { label:'Original',    value:fa.originalCount, color:'#10b981' },
                  { label:'Cloned/Forked', value:fa.forkCount, color: fa.forkCount > 0 ? '#f59e0b' : '#10b981' },
                ].map(s=>(
                  <div className="stat-card" key={s.label} style={{ borderColor:s.color+'33' }}>
                    <div className="stat-value" style={{ backgroundImage:`linear-gradient(135deg,${s.color},${s.color}88)` }}>{s.value}</div>
                    <div className="stat-label">{s.label}</div>
                  </div>
                ))}
              </div>

              {fa.forkCount === 0 ? (
                <div style={{ background:'rgba(16,185,129,0.08)', border:'1px solid rgba(16,185,129,0.25)', borderRadius:12, padding:'20px 24px', display:'flex', gap:14, alignItems:'center' }}>
                  <span style={{ fontSize:28 }}>✅</span>
                  <div>
                    <strong style={{ color:'#10b981' }}>No cloned/forked repositories detected</strong>
                    <p style={{ color:'var(--text2)', fontSize:13, marginTop:4 }}>All {fa.totalRepos} repositories appear to be original work.</p>
                  </div>
                </div>
              ) : (
                <Section title={`⚠️ ${fa.forkCount} Forked / Cloned Repositories`} tag="REST /users/{username}/repos — fork:true" accent>
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))', gap:12 }}>
                    {fa.forkedRepos.map(r=>(
                      <div key={r.name} style={{ background:'rgba(245,158,11,0.06)', border:'1px solid rgba(245,158,11,0.25)', borderRadius:12, padding:16 }}>
                        <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:8 }}>
                          <span style={{ fontSize:14, fontWeight:700, color:'#fbbf24' }}>{r.name}</span>
                          <span className="fork-badge">⑂ Fork</span>
                        </div>
                        <div style={{ fontSize:12, color:'var(--text2)', marginBottom:8, lineHeight:1.5 }}>{r.description || 'No description'}</div>
                        <div style={{ fontSize:11, color:'var(--muted)' }}>Last updated: {fmtDate(r.updatedAt)}</div>
                        <a href={r.url} target="_blank" rel="noopener noreferrer" style={{ fontSize:11, color:'#818cf8', marginTop:6, display:'block' }}>View on GitHub →</a>
                      </div>
                    ))}
                  </div>
                </Section>
              )}

              <Section title="✦ Original Repositories" defaultOpen={false}>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))', gap:10 }}>
                  {fa.originalRepos.map(r=>(
                    <div key={r.name} style={{ background:'rgba(16,185,129,0.06)', border:'1px solid rgba(16,185,129,0.2)', borderRadius:10, padding:12, display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                      <div>
                        <div style={{ fontSize:13, fontWeight:600, color:'var(--text)' }}>{r.name}</div>
                        <div style={{ fontSize:11, color:'var(--muted)', marginTop:2 }}>{r.language || 'No language'}</div>
                      </div>
                      <div style={{ fontSize:12, color:'var(--text2)' }}>⭐ {r.stars}</div>
                    </div>
                  ))}
                </div>
              </Section>
            </>
          )}

          {/* ── CONTRIBUTORS ── */}
          {tab === 'contributors' && (
            <Section title="Contributors" tag={`REST /repos/${username}/${repoName}/contributors`}>
              {data.contributors.length === 0 ? (
                <p style={{color:'var(--muted)',fontSize:13}}>No contributor data available.</p>
              ) : (
                <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
                  {data.contributors.map((c,i)=>(
                    <div key={c.login} style={{ display:'flex', alignItems:'center', gap:14, background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:12, padding:'12px 16px' }}>
                      <div style={{ fontWeight:700, fontSize:18, color:'var(--muted)', width:28, textAlign:'center' }}>#{i+1}</div>
                      <img src={c.avatar} alt={c.login} style={{ width:40, height:40, borderRadius:'50%', border:'2px solid rgba(99,102,241,0.3)' }} />
                      <div style={{ flex:1 }}>
                        <div style={{ fontSize:14, fontWeight:700, color:'var(--text)' }}>
                          <a href={c.url} target="_blank" rel="noopener noreferrer" style={{ color:'#818cf8', textDecoration:'none' }}>{c.login}</a>
                        </div>
                        <div style={{ fontSize:12, color:'var(--muted)', marginTop:2 }}>{c.commits} commits</div>
                      </div>
                      <div style={{ width:120 }}>
                        <div style={{ height:6, background:'var(--surface3)', borderRadius:3, overflow:'hidden' }}>
                          <div style={{ height:'100%', background:'linear-gradient(90deg,#6366f1,#8b5cf6)', borderRadius:3, width:`${(c.commits/data.contributors[0].commits)*100}%` }} />
                        </div>
                      </div>
                      <div style={{ fontSize:13, fontWeight:700, color:'var(--text)', width:60, textAlign:'right' }}>
                        {((c.commits/stats.totalCommits)*100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Section>
          )}
        </div>
      )}
    </>
  );
}
