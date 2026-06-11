'use client';

import { useState, useCallback } from 'react';
import StatGrid        from '../components/StatGrid';
import RepoGrid        from '../components/RepoGrid';
import ContribCalendar from '../components/ContribCalendar';
import LanguageChart   from '../components/LanguageChart';
import EventsList      from '../components/EventsList';
import { langColor }   from '../components/RepoGrid';

/* ─── helpers ──────────────────────────────────────────── */
function fmtDate(d) {
  if (!d) return '—';
  return new Date(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}
function KV({ label, value }) {
  const isNull = value === null || value === undefined || value === '';
  return (
    <tr>
      <td>{label}</td>
      <td className={isNull ? 'null-val' : ''}>{isNull ? 'null' : String(value)}</td>
    </tr>
  );
}
function Section({ title, tag, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`section ${open ? '' : 'collapsed'}`}>
      <div className="section-head" onClick={() => setOpen(o => !o)}>
        <div className="section-title">
          <span>{title}</span>
          {tag && <span className="endpoint-tag">{tag}</span>}
        </div>
        <span className="chevron">▼</span>
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

const TABS = [
  { id: 'overview',       icon: '👤', label: 'Profile' },
  { id: 'repos',          icon: '📦', label: 'Repositories' },
  { id: 'contributions',  icon: '📅', label: 'Contributions' },
  { id: 'languages',      icon: '💻', label: 'Languages' },
  { id: 'hireflow',       icon: '🎯', label: 'HireFlow Fields' },
  { id: 'raw',            icon: '{ }', label: 'Raw JSON' },
];

const STEPS = [
  'Connecting to GitHub…',
  'Running GraphQL query (profile + repos + contributions)…',
  'Fetching REST profile…',
  'Fetching recent events…',
  'Rendering dashboard…',
];

const HIREFLOW_FIELDS = [
  { icon:'🔍', title:'Resume–Repo Matching',   field:'repositoryTopics, name',
    desc:'Repo names & topics matched against resume project list using fuzzy match.' },
  { icon:'⑂',  title:'Clone Detection',        field:'isFork + parent.nameWithOwner',
    desc:'Flags forked repos. Penalizes copied work in scoring.' },
  { icon:'📅', title:'Active Days Score',       field:'contributionCalendar.weeks',
    desc:'Days with ≥1 contribution count as active. Scored over 365 days.' },
  { icon:'📝', title:'Commit Volume',           field:'history.totalCount / totalCommitContributions',
    desc:'Per-repo commit count + total annual commit contributions.' },
  { icon:'💻', title:'Skill–JD Match',          field:'languages.edges[].size',
    desc:'Byte-weighted language usage compared against JD required stack.' },
  { icon:'🕐', title:'Timeline Fraud',          field:'createdAt vs resume date',
    desc:'Repos created after claimed end date are flagged.' },
  { icon:'🔀', title:'PR & Issue Activity',     field:'totalPullRequestContributions',
    desc:'Indicates collaborative coding habits beyond solo projects.' },
  { icon:'📌', title:'Pinned Quality',          field:'pinnedItems',
    desc:'Pinned repos show best work. Star count & language scored.' },
  { icon:'📊', title:'Engagement Score',        field:'stargazerCount, forkCount, topics',
    desc:'Stars, forks, topics, README presence — community traction.' },
  { icon:'🤖', title:'AI Code Probability',     field:'commit cadence + contributorCount',
    desc:'Uniform cadence + single contributor + low issues → elevated AI flag.' },
  { icon:'🌐', title:'Activity Recency',        field:'updatedAt + /events',
    desc:'Recent repo updates & events confirm active coding.' },
  { icon:'👥', title:'Community Presence',      field:'followers, following, totalPRContributions',
    desc:'Measures engagement beyond personal projects.' },
];

/* ─── Main Page ────────────────────────────────────────── */
export default function Home() {
  const [username, setUsername]   = useState('');
  const [token,    setToken]      = useState('');
  const [loading,  setLoading]    = useState(false);
  const [step,     setStep]       = useState(0);
  const [error,    setError]      = useState('');
  const [data,     setData]       = useState(null);
  const [tab,      setTab]        = useState('overview');

  const fetchData = useCallback(async () => {
    if (!username.trim() || !token.trim()) {
      setError('Please enter both a GitHub username and a personal access token.');
      return;
    }
    setError('');
    setData(null);
    setLoading(true);
    setStep(0);

    try {
      setStep(1);
      const res = await fetch('/api/github', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), token: token.trim() }),
      });
      setStep(2);
      const json = await res.json();
      if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
      setStep(4);
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [username, token]);

  const handleKey = (e) => { if (e.key === 'Enter') fetchData(); };

  const user  = data?.gql?.user;
  const rest  = data?.rest;
  const events = data?.events ?? [];
  const cc    = user?.contributionsCollection;
  const repos = user?.repositories?.nodes ?? [];

  return (
    <>
      {/* ── Header ── */}
      <header className="header">
        <span className="logo">HireFlow</span>
        <span className="badge">GitHub API Explorer</span>
        <a href="/repo" style={{ marginLeft: 8, fontSize: 12, fontWeight: 600, color: '#818cf8', background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.25)', padding: '4px 12px', borderRadius: 20, textDecoration: 'none' }}>
          🔍 Repo Analyzer →
        </a>
        {data && (
          <div className="header-right">
            <span className="rate-badge">
              ✅ Data loaded · {repos.length} repos · {events.length} events
            </span>
          </div>
        )}
      </header>

      {/* ── Hero + Input ── */}
      {!data && (
        <div className="hero">
          <h1>GitHub API Explorer</h1>
          <p>See exactly what data is available for any GitHub profile — REST + GraphQL in one dashboard.</p>

          <div className="input-form">
            <div className="glass-card">
              <div className="form-row">
                <div className="field">
                  <label>GitHub Username</label>
                  <input
                    id="username"
                    type="text"
                    placeholder="e.g. Gopisokk"
                    value={username}
                    onChange={e => setUsername(e.target.value)}
                    onKeyDown={handleKey}
                    autoComplete="off"
                    autoFocus
                  />
                </div>
                <div className="field">
                  <label>Personal Access Token</label>
                  <input
                    id="token"
                    type="password"
                    placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                    value={token}
                    onChange={e => setToken(e.target.value)}
                    onKeyDown={handleKey}
                  />
                  <span className="hint">Needs scopes: read:user · public_repo</span>
                </div>
              </div>

              <button
                className="btn-primary"
                id="fetch-btn"
                onClick={fetchData}
                disabled={loading}
              >
                {loading ? '⏳ Fetching from GitHub API…' : '🚀 Fetch Everything from GitHub API'}
              </button>
            </div>
          </div>

          {/* Loading status */}
          {loading && (
            <div className="status-bar">
              <div className="status-inner">
                <div className="spinner" />
                <span className="step-text">{STEPS[step]}</span>
                <div className="status-steps">
                  {STEPS.map((_, i) => (
                    <div
                      key={i}
                      className={`step-dot ${i < step ? 'done' : i === step ? 'active' : ''}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="status-bar">
              <div className="error-box">
                <span>❌</span>
                <div>
                  <strong>Error:</strong> {error}
                  {error.includes('401') && (
                    <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text2)' }}>
                      Your token may be invalid or expired. Generate a new one at github.com/settings/tokens
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Results ── */}
      {data && user && (
        <div className="results">
          {/* Re-search bar */}
          <div style={{ display:'flex', gap:10, marginBottom:28, alignItems:'center', flexWrap:'wrap' }}>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              style={{ background:'var(--surface)', border:'1px solid var(--border2)', borderRadius:10, color:'var(--text)', fontFamily:'var(--font-mono)', fontSize:13, padding:'9px 14px', outline:'none', width:200 }}
              onKeyDown={handleKey}
            />
            <button
              onClick={fetchData}
              disabled={loading}
              style={{ padding:'9px 20px', background:'linear-gradient(135deg,var(--accent),var(--accent2))', border:'none', borderRadius:10, color:'#fff', fontWeight:700, cursor:'pointer', fontSize:13 }}
            >
              {loading ? '…' : '↺ Refresh'}
            </button>
            {error && <span style={{ color:'var(--red)', fontSize:13 }}>❌ {error}</span>}
          </div>

          {/* Tabs */}
          <div className="tab-bar">
            {TABS.map(t => (
              <button
                key={t.id}
                className={`tab-btn ${tab === t.id ? 'active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                <span>{t.icon}</span> {t.label}
              </button>
            ))}
          </div>

          {/* ── OVERVIEW ── */}
          {tab === 'overview' && (
            <>
              <Section title="User Profile" tag="REST /users/{username} + GraphQL">
                <div className="profile-header">
                  <img className="avatar" src={user.avatarUrl} alt={user.login} />
                  <div className="profile-info">
                    <h2>{user.name || user.login}</h2>
                    <div className="login">@{user.login}</div>
                    {user.bio && <div className="bio">{user.bio}</div>}
                    <div className="meta-row">
                      {user.company    && <div className="meta-chip">🏢 <strong>{user.company}</strong></div>}
                      {user.location   && <div className="meta-chip">📍 <strong>{user.location}</strong></div>}
                      {user.email      && <div className="meta-chip">✉️ <strong>{user.email}</strong></div>}
                      {user.websiteUrl && <div className="meta-chip">🌐 <strong>{user.websiteUrl}</strong></div>}
                      <div className="meta-chip">📅 Joined <strong>{fmtDate(user.createdAt)}</strong></div>
                    </div>
                  </div>
                </div>

                <StatGrid items={[
                  { label: 'Followers',         value: user.followers.totalCount },
                  { label: 'Following',          value: user.following.totalCount },
                  { label: 'Public Repos',       value: user.repositories.totalCount },
                  { label: 'Stars Given',        value: user.starredRepositories.totalCount },
                  { label: 'Public Gists (REST)',value: rest?.public_gists },
                ]} />

                <table className="kv-table">
                  <tbody>
                    <KV label="GraphQL: login"      value={user.login} />
                    <KV label="GraphQL: name"       value={user.name} />
                    <KV label="GraphQL: email"      value={user.email} />
                    <KV label="GraphQL: company"    value={user.company} />
                    <KV label="GraphQL: location"   value={user.location} />
                    <KV label="GraphQL: websiteUrl" value={user.websiteUrl} />
                    <KV label="GraphQL: bio"        value={user.bio} />
                    <KV label="GraphQL: createdAt"  value={user.createdAt} />
                    <KV label="GraphQL: updatedAt"  value={user.updatedAt} />
                    <KV label="REST: public_repos"  value={rest?.public_repos} />
                    <KV label="REST: public_gists"  value={rest?.public_gists} />
                    <KV label="REST: site_admin"    value={String(rest?.site_admin)} />
                    <KV label="REST: hireable"      value={rest?.hireable} />
                    <KV label="REST: twitter_username" value={rest?.twitter_username} />
                    <KV label="REST: node_id"       value={rest?.node_id} />
                  </tbody>
                </table>
              </Section>

              <Section title="Contribution Summary" tag="GraphQL contributionsCollection">
                <StatGrid items={[
                  { label: 'Total Commits',   value: cc.totalCommitContributions },
                  { label: 'Pull Requests',   value: cc.totalPullRequestContributions },
                  { label: 'Issues',          value: cc.totalIssueContributions },
                  { label: 'Repos Created',   value: cc.totalRepositoryContributions },
                  { label: 'Total (Calendar)',value: cc.contributionCalendar.totalContributions },
                ]} />
              </Section>

              <Section title="Pinned Repositories" tag="GraphQL pinnedItems">
                {user.pinnedItems.nodes.length === 0 ? (
                  <p style={{ color:'var(--muted)', fontSize:13 }}>No pinned repositories.</p>
                ) : (
                  <div className="pinned-grid">
                    {user.pinnedItems.nodes.map(p => (
                      <div className="pinned-card" key={p.name}>
                        <div className="pinned-label">📌 Pinned</div>
                        <div className="pinned-name"><a href={p.url} target="_blank" rel="noopener noreferrer" style={{color:'inherit',textDecoration:'none'}}>{p.name}</a></div>
                        <div className="pinned-desc">{p.description || 'No description'}</div>
                        <div className="pinned-foot">
                          {p.primaryLanguage && (
                            <span style={{ display:'flex', alignItems:'center', gap:5 }}>
                              <span style={{ width:10, height:10, borderRadius:'50%', background:langColor(p.primaryLanguage.name), display:'inline-block' }} />
                              {p.primaryLanguage.name}
                            </span>
                          )}
                          <span>⭐ {p.stargazerCount}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Section>
            </>
          )}

          {/* ── REPOS ── */}
          {tab === 'repos' && (
            <Section title="All Public Repositories" tag="GraphQL repositories(first:30)">
              <RepoGrid repos={repos} totalCount={user.repositories.totalCount} />
            </Section>
          )}

          {/* ── CONTRIBUTIONS ── */}
          {tab === 'contributions' && (
            <>
              <Section title="Contribution Calendar — Last 12 Months" tag="GraphQL contributionCalendar">
                <ContribCalendar
                  weeks={cc.contributionCalendar.weeks}
                  stats={cc.contributionCalendar}
                />
              </Section>

              <Section title="Recent Public Events" tag="REST /users/{username}/events">
                <EventsList events={events} />
              </Section>
            </>
          )}

          {/* ── LANGUAGES ── */}
          {tab === 'languages' && (
            <Section title="Language Usage Across All Repos" tag="GraphQL languages(first:10)">
              <LanguageChart repos={repos} />
            </Section>
          )}

          {/* ── HIREFLOW FIELDS ── */}
          {tab === 'hireflow' && (
            <Section title="HireFlow — What Each API Field Is Used For">
              <div className="summary-grid">
                {HIREFLOW_FIELDS.map(f => (
                  <div className="summary-item" key={f.title}>
                    <div className="summary-icon">{f.icon}</div>
                    <div className="summary-text">
                      <h4>{f.title}</h4>
                      <p>{f.desc}</p>
                      <span className="field-tag">{f.field}</span>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* ── RAW JSON ── */}
          {tab === 'raw' && (
            <>
              <Section title="Raw GraphQL Response" tag="GraphQL">
                <pre className="raw-json">{JSON.stringify(data.gql, null, 2)}</pre>
              </Section>
              <Section title="Raw REST Response — /users/{username}" tag="REST">
                <pre className="raw-json">{JSON.stringify(data.rest, null, 2)}</pre>
              </Section>
              <Section title="Raw Events Response" tag="REST /events">
                <pre className="raw-json">{JSON.stringify(data.events, null, 2)}</pre>
              </Section>
            </>
          )}
        </div>
      )}
    </>
  );
}
