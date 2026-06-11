export const runtime = 'nodejs';

// Fetches deep repo data: all commits (paginated), languages, readme, contributors
async function ghFetch(url, token) {
  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status} ${res.statusText} — ${url}\n${txt}`);
  }
  return res.json();
}

// Paginate through all commits (GitHub caps at 100/page)
async function getAllCommits(owner, repo, token) {
  const allCommits = [];
  let page = 1;
  while (true) {
    const url = `https://api.github.com/repos/${owner}/${repo}/commits?per_page=100&page=${page}`;
    const data = await ghFetch(url, token);
    if (!Array.isArray(data) || data.length === 0) break;
    allCommits.push(...data);
    if (data.length < 100) break; // last page
    page++;
    if (page > 50) break; // safety cap at 5000 commits
  }
  return allCommits;
}

export async function POST(req) {
  try {
    const { username, token, repoName } = await req.json();
    if (!username || !token || !repoName) {
      return Response.json({ error: 'username, token, and repoName are required' }, { status: 400 });
    }

    const owner = username.trim();
    const repo  = repoName.trim();

    // 1. Repo metadata
    const repoData = await ghFetch(`https://api.github.com/repos/${owner}/${repo}`, token);

    // 2. All commits
    const commits = await getAllCommits(owner, repo, token);

    // 3. Languages breakdown
    const languages = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/languages`, token);

    // 4. Contributors
    let contributors = [];
    try {
      contributors = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/contributors?per_page=20`, token);
    } catch (_) {}

    // 5. README (base64 encoded)
    let readme = null;
    try {
      const r = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/readme`, token);
      readme = Buffer.from(r.content, 'base64').toString('utf-8').slice(0, 3000);
    } catch (_) {}

    // 6. All repos (for fork/clone detection)
    const allRepos = await ghFetch(
      `https://api.github.com/users/${owner}/repos?per_page=100&sort=updated`,
      token
    );

    // ── Compute stats ──────────────────────────────────────────
    // Unique days worked
    const daySet = new Set(
      commits.map(c => (c.commit?.author?.date || '').slice(0, 10)).filter(Boolean)
    );
    const uniqueDays = [...daySet].sort();

    // Weekly activity (commits per week)
    const weekMap = {};
    commits.forEach(c => {
      const date = c.commit?.author?.date;
      if (!date) return;
      const d  = new Date(date);
      // ISO week start (Monday)
      const day = d.getDay() || 7;
      d.setDate(d.getDate() - day + 1);
      const key = d.toISOString().slice(0, 10);
      weekMap[key] = (weekMap[key] || 0) + 1;
    });

    // Monthly activity
    const monthMap = {};
    commits.forEach(c => {
      const date = c.commit?.author?.date;
      if (!date) return;
      const key = date.slice(0, 7); // YYYY-MM
      monthMap[key] = (monthMap[key] || 0) + 1;
    });

    // Commit authors
    const authorMap = {};
    commits.forEach(c => {
      const name = c.commit?.author?.name || 'Unknown';
      authorMap[name] = (authorMap[name] || 0) + 1;
    });

    // Fork detection in all repos
    const forkedRepos = allRepos
      .filter(r => r.fork)
      .map(r => ({ name: r.name, url: r.html_url, description: r.description, updatedAt: r.updated_at }));

    const originalRepos = allRepos
      .filter(r => !r.fork)
      .map(r => ({ name: r.name, url: r.html_url, language: r.language, stars: r.stargazers_count }));

    return Response.json({
      repo: {
        name:          repoData.name,
        fullName:      repoData.full_name,
        description:   repoData.description,
        url:           repoData.html_url,
        isFork:        repoData.fork,
        parent:        repoData.parent?.full_name,
        parentUrl:     repoData.parent?.html_url,
        createdAt:     repoData.created_at,
        updatedAt:     repoData.updated_at,
        pushedAt:      repoData.pushed_at,
        defaultBranch: repoData.default_branch,
        stars:         repoData.stargazers_count,
        forks:         repoData.forks_count,
        watchers:      repoData.watchers_count,
        openIssues:    repoData.open_issues_count,
        size:          repoData.size,
        topics:        repoData.topics || [],
        license:       repoData.license?.name,
        visibility:    repoData.visibility,
      },
      commits: commits.map(c => ({
        sha:     c.sha?.slice(0, 7),
        message: c.commit?.message?.split('\n')[0], // first line only
        date:    c.commit?.author?.date,
        author:  c.commit?.author?.name,
        email:   c.commit?.author?.email,
        url:     c.html_url,
      })),
      stats: {
        totalCommits:    commits.length,
        uniqueDays:      uniqueDays.length,
        uniqueDaysList:  uniqueDays,
        firstCommit:     commits.length ? commits[commits.length - 1]?.commit?.author?.date : null,
        lastCommit:      commits.length ? commits[0]?.commit?.author?.date : null,
        weeklyActivity:  weekMap,
        monthlyActivity: monthMap,
        authorBreakdown: authorMap,
      },
      languages,
      contributors: contributors.map(c => ({
        login:       c.login,
        avatar:      c.avatar_url,
        commits:     c.contributions,
        url:         c.html_url,
      })),
      readme,
      forkAnalysis: {
        forkedRepos,
        originalRepos,
        totalRepos:    allRepos.length,
        forkCount:     forkedRepos.length,
        originalCount: originalRepos.length,
      },
    });
  } catch (err) {
    return Response.json({ error: err.message }, { status: 500 });
  }
}
