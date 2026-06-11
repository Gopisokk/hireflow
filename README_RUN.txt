═══════════════════════════════════════════════════════════════════
  HIREFLOW — GitHub API Explorer
  Run Guide + API Call Analysis
═══════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1 — Make sure Node.js is installed
  ----------------------------------------
  Open terminal and check:
    node --version     (need v18 or higher)
    npm --version

  If not installed: https://nodejs.org


  STEP 2 — Go to the project folder
  ------------------------------------
    cd C:\Users\radha\Desktop\finalyear_project\code_file\hireflow


  STEP 3 — Install dependencies (first time only)
  --------------------------------------------------
    npm install


  STEP 4 — Get a GitHub Personal Access Token
  ---------------------------------------------
    1. Go to: https://github.com/settings/tokens
    2. Click "Generate new token (classic)"
    3. Give it a name (e.g. "HireFlow")
    4. Select scopes:
         [x] read:user
         [x] public_repo
    5. Click "Generate token"
    6. COPY the token (starts with ghp_...)
       — you won't see it again!


  STEP 5 — Start the app
  ------------------------
    npm run dev

  The app will start at:
    Local:    http://localhost:3000
    Network:  http://192.168.0.105:3000


  STEP 6 — Use the app
  ----------------------
  Page 1 — Profile Explorer:   http://localhost:3000
    → Enter GitHub username + token
    → Click "Fetch Everything from GitHub API"
    → See: profile, repos, contribution calendar,
           languages, raw JSON

  Page 2 — Repo Deep Dive:     http://localhost:3000/repo
    → Enter username + repo name + token
    → Click "Analyze Repository"
    → See: all commits, unique days worked,
           activity calendar, fork/clone check


  STOP the app
  -------------
    Press Ctrl + C in the terminal


  RESTART after closing
  ----------------------
    Just run:  npm run dev   (no install needed again)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WHAT 1 GRAPHQL CALL GIVES YOU vs WHAT IT CANNOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ONE GraphQL call to https://api.github.com/graphql
  returns ALL of this in a single request:

  ✅ CAN GET in 1 GraphQL call
  ─────────────────────────────
  USER PROFILE
    • name, email, bio, company, location, website
    • avatar URL, join date, last updated
    • followers count, following count
    • total public repos count
    • starred repos count

  CONTRIBUTION DATA (last 12 months)
    • totalCommitContributions
    • totalPullRequestContributions
    • totalIssueContributions
    • totalRepositoryContributions
    • Full contribution calendar (52 weeks × 7 days)
      → every single day + count + color
    → This alone gives: active days, streak,
      peak day, avg commits/week

  REPOSITORIES (up to 30 most recent)
  For EACH repo you get:
    • name, description, URL
    • isFork (clone detection) ✓
    • parent repo name (if forked) ✓
    • stars, forks count
    • createdAt, updatedAt (fraud detection) ✓
    • primaryLanguage
    • All languages with byte sizes (first 10) ✓
    • topics/tags
    • diskUsage (repo size)
    • isPrivate, isArchived
    • LAST commit: message, date, author
    • TOTAL commit count for the repo ✓

  PINNED REPOS (up to 6)
    • name, description, stars, language

  ❌ CANNOT GET in GraphQL (needs separate REST calls)
  ──────────────────────────────────────────────────────
  • Full commit history (all messages + dates)
    → GraphQL only gives LAST 1 commit + total count
    → Need: GET /repos/{owner}/{repo}/commits  per repo

  • README file content
    → Need: GET /repos/{owner}/{repo}/readme  per repo

  • Contributors list (who committed how much)
    → Need: GET /repos/{owner}/{repo}/contributors  per repo

  • Recent public activity (pushes, PRs, forks, stars)
    → Need: GET /users/{username}/events

  • Extra profile fields (public_gists, hireable,
    twitter_username, site_admin)
    → Need: GET /users/{username}

  • Exact commit messages for fraud analysis
    → Need: GET /repos/{owner}/{repo}/commits  per repo


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  API CALLS NEEDED: 500 to 1000 DEVELOPERS
  (3 Projects Deep-Dive Per Person)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GitHub Rate Limit (with token): 5,000 requests / hour
  GraphQL Rate Limit:             5,000 points  / hour

  ──────────────────────────────────────────────────────
  STAGE 2 — Basic GitHub Verification (per developer)
  ──────────────────────────────────────────────────────

  Call #  | What                            | Count
  ─────────────────────────────────────────────────
    1      | GraphQL — full profile +        |
           | repos + contributions +         |   1
           | languages + calendar            |
  ─────────────────────────────────────────────────
  TOTAL Stage 2 (basic) per developer = 1 call

  For 500 devs  →   500 calls  →  ~6 minutes
  For 1000 devs →  1000 calls  →  ~12 minutes


  ──────────────────────────────────────────────────────
  STAGE 2 + STAGE 3 — Deep Dive on 3 Projects Each
  ──────────────────────────────────────────────────────

  Call #  | What                            | Count
  ─────────────────────────────────────────────────
    1      | GraphQL (profile + repos)       |   1
    2      | REST events (/events)           |   1
  ─────────────────────────────────────────────────
  Per project (×3 projects per person):
    3      | Commit history — page 1         |   1 per repo
           | (+ 1 more per 100 commits)      |
    4      | README content                  |   1 per repo
    5      | Contributors list               |   1 per repo
  ─────────────────────────────────────────────────
  TOTAL per developer (3 projects, <100 commits each):
    1 (GraphQL) + 1 (events) + 3×3 (deep repos) = 11 calls

  NOTE: If a repo has 200 commits → 2 commit pages
        If a repo has 300 commits → 3 commit pages
        (adds 1-2 extra calls per busy repo)

  ──────────────────────────────────────────────────────
  FULL CALCULATION TABLE
  ──────────────────────────────────────────────────────

  Developers | Calls Each | Total Calls | Hours Needed
  ───────────────────────────────────────────────────────
      500    |     11     |    5,500    | ~1.1 hours
      750    |     11     |    8,250    | ~1.65 hours
     1000    |     11     |   11,000    | ~2.2 hours

  With busy repos (avg 200 commits each, 2 pages):
      500    |     14     |    7,000    | ~1.4 hours
      750    |     14     |   10,500    | ~2.1 hours
     1000    |     14     |   14,000    | ~2.8 hours


  ──────────────────────────────────────────────────────
  HOW TO SPEED THIS UP (for real project scale)
  ──────────────────────────────────────────────────────

  Option 1 — Multiple GitHub Tokens (parallel processing)
    Use 3 different GitHub accounts → 3 tokens
    → 3 × 5,000 = 15,000 req/hour
    → 1000 devs processed in under 1 hour

  Option 2 — GraphQL-first, REST only when needed
    Run GraphQL for ALL 1000 devs first (1000 calls)
    Then only fetch deep data for SHORTLISTED candidates
    If you shortlist top 10% → 100 devs × 9 REST calls
    = 900 REST + 1000 GraphQL = 1900 total calls
    → Well within 1-hour rate limit

  Option 3 — Cache results
    Store GraphQL responses in database
    Only re-fetch if updatedAt changed
    → Saves 80% of calls on re-runs

  ──────────────────────────────────────────────────────
  RECOMMENDED APPROACH FOR HIREFLOW
  ──────────────────────────────────────────────────────

  Stage 1: Resume parsing (no API calls)
    → Shortlist top 10% from resumes
    → From 1000 applicants → 100 shortlisted

  Stage 2: GitHub verification for 100 shortlisted only
    → 100 × 1 GraphQL call = 100 calls
    → Fast, under rate limit in minutes

  Stage 3: Deep code analysis for final 20-30 candidates
    → 30 devs × 3 repos × 3 REST calls = 270 calls

  TOTAL API CALLS FOR FULL PIPELINE (1000 applicants):
    Stage 1: 0 calls
    Stage 2: 100 GraphQL calls
    Stage 3: 270 REST calls
    ─────────────────────────
    TOTAL: 370 API calls
    Time:  < 5 minutes (well within rate limit)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROJECT FILES STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  hireflow/
  ├── app/
  │   ├── page.js                  ← Main GitHub profile explorer
  │   ├── layout.js                ← App shell (fonts, metadata)
  │   ├── globals.css              ← All styles (dark theme)
  │   ├── repo/
  │   │   └── page.js             ← Repo deep dive analyzer
  │   └── api/
  │       ├── github/
  │       │   └── route.js        ← GraphQL + REST proxy (profile)
  │       └── repo/
  │           └── route.js        ← Commit history + fork analysis
  ├── components/
  │   ├── StatGrid.jsx             ← Number stat cards
  │   ├── RepoGrid.jsx             ← Repo cards with fork badges
  │   ├── ContribCalendar.jsx      ← Canvas heatmap calendar
  │   ├── LanguageChart.jsx        ← Language breakdown bars
  │   └── EventsList.jsx           ← Recent activity list
  ├── package.json
  └── README_RUN.txt               ← This file


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  QUICK REFERENCE — GITHUB RATE LIMITS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  With token (ghp_...):      5,000 REST requests/hour
                             5,000 GraphQL points/hour
  Without token:               60 requests/hour (useless at scale)
  Check your remaining:      GET https://api.github.com/rate_limit
  Header returned:           X-RateLimit-Remaining: ####


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
