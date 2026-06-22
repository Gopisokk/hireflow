"""
HireFlow Skill & Project Extractor
------------------------------------
Regex-based extraction of technical skills against a curated taxonomy,
plus heuristic extraction of project names from resume text.
"""

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Skill Taxonomy (~200 skills organized by category)
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_TAXONOMY: dict[str, list[str]] = {
    "languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
        "Go", "Golang", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
        "R", "MATLAB", "Perl", "Haskell", "Lua", "Dart", "Elixir",
        "Clojure", "Julia", "Objective-C", "Shell", "Bash", "PowerShell",
        "SQL", "HTML", "CSS", "SASS", "SCSS", "LESS", "Assembly",
        "Solidity", "VHDL", "Verilog",
    ],
    "frameworks_frontend": [
        "React", "React.js", "ReactJS", "Next.js", "NextJS", "Angular",
        "AngularJS", "Vue", "Vue.js", "VueJS", "Svelte", "SvelteKit",
        "Nuxt", "Nuxt.js", "Gatsby", "Remix", "Astro",
        "Bootstrap", "Tailwind CSS", "Tailwind", "Material UI", "MUI",
        "Chakra UI", "Ant Design", "jQuery",
    ],
    "frameworks_backend": [
        "Django", "Flask", "FastAPI", "Express", "Express.js",
        "Spring", "Spring Boot", "NestJS", "Nest.js",
        "Ruby on Rails", "Rails", "Laravel", "Gin", "Fiber",
        "ASP.NET", ".NET", "Actix", "Rocket",
        "Koa", "Hapi", "Phoenix", "Sinatra",
    ],
    "databases": [
        "MySQL", "PostgreSQL", "Postgres", "SQLite", "MongoDB",
        "Redis", "Cassandra", "DynamoDB", "CockroachDB",
        "MariaDB", "Oracle", "SQL Server", "MSSQL",
        "Neo4j", "InfluxDB", "Elasticsearch", "ElasticSearch",
        "Firebase", "Firestore", "Supabase", "PlanetScale",
        "Couchbase", "CouchDB", "Memcached",
    ],
    "cloud_platforms": [
        "AWS", "Amazon Web Services", "Azure", "Google Cloud", "GCP",
        "Heroku", "DigitalOcean", "Vercel", "Netlify",
        "Cloudflare", "IBM Cloud", "Oracle Cloud", "Linode",
        "Render", "Railway", "Fly.io",
    ],
    "cloud_services": [
        "S3", "EC2", "Lambda", "SQS", "SNS", "CloudFront",
        "API Gateway", "RDS", "ECS", "EKS", "Fargate",
        "CloudFormation", "Cloud Functions", "BigQuery",
        "App Engine", "Azure Functions", "Blob Storage",
    ],
    "devops_tools": [
        "Docker", "Kubernetes", "K8s", "Terraform", "Ansible",
        "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI",
        "Travis CI", "ArgoCD", "Helm", "Prometheus", "Grafana",
        "Nginx", "Apache", "Caddy", "HAProxy",
        "Vagrant", "Puppet", "Chef", "Pulumi",
    ],
    "data_ml": [
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Sklearn",
        "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
        "OpenCV", "NLTK", "SpaCy", "Hugging Face", "HuggingFace",
        "Transformers", "XGBoost", "LightGBM", "CatBoost",
        "Apache Spark", "Spark", "Hadoop", "Airflow", "Kafka",
        "MLflow", "Kubeflow", "SageMaker", "Weights & Biases",
        "Jupyter", "Colab", "Streamlit", "Gradio",
        "LangChain", "LlamaIndex", "OpenAI API", "GPT",
        "BERT", "YOLO", "GANs", "Stable Diffusion",
    ],
    "mobile": [
        "React Native", "Flutter", "SwiftUI", "Jetpack Compose",
        "Xamarin", "Ionic", "Cordova", "Expo",
        "Android SDK", "iOS SDK", "ARKit", "ARCore",
    ],
    "testing": [
        "Jest", "Mocha", "Chai", "Cypress", "Selenium",
        "Playwright", "Puppeteer", "PyTest", "Pytest", "unittest",
        "JUnit", "TestNG", "RSpec", "Postman",
    ],
    "version_control": [
        "Git", "GitHub", "GitLab", "Bitbucket", "SVN",
    ],
    "other_tools": [
        "GraphQL", "REST", "RESTful", "gRPC", "WebSocket",
        "OAuth", "JWT", "SAML", "SSO",
        "Figma", "Adobe XD", "Sketch",
        "Jira", "Trello", "Notion", "Confluence",
        "Linux", "Unix", "Windows Server",
        "Agile", "Scrum", "Kanban", "CI/CD",
        "Microservices", "Serverless", "Monorepo",
        "RabbitMQ", "ZeroMQ", "NATS", "Celery",
        "Webpack", "Vite", "Babel", "ESLint", "Prettier",
        "npm", "yarn", "pnpm", "pip", "Poetry", "Conda",
    ],
}

# Build a flat lookup: lowercase skill → canonical name
_SKILL_LOOKUP: dict[str, str] = {}
for _category, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        _SKILL_LOOKUP[_skill.lower()] = _skill

# Pre-compile patterns sorted longest-first to avoid partial matches
_SORTED_SKILLS = sorted(_SKILL_LOOKUP.keys(), key=len, reverse=True)
_SKILL_PATTERNS: list[tuple[re.Pattern[str], str]] = []
for _sk in _SORTED_SKILLS:
    # Word-boundary matching; escape special regex chars in skill name
    pattern = re.compile(r"(?<![a-zA-Z0-9\-_])" + re.escape(_sk) + r"(?![a-zA-Z0-9\-_])", re.IGNORECASE)
    _SKILL_PATTERNS.append((pattern, _SKILL_LOOKUP[_sk]))


# ═══════════════════════════════════════════════════════════════════════════════
#  Project section header patterns
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:projects?|personal projects?|academic projects?|"
    r"key projects?|notable projects?|selected projects?|side projects?)"
    r"\s*[:\-]?\s*\n",
    re.IGNORECASE,
)

_NEXT_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:experience|education|skills?|certifications?|"
    r"achievements?|awards?|publications?|interests?|hobbies|"
    r"references?|summary|objective|contact|languages|technologies|"
    r"open source|competitive)\s*[:\-]?\s*\n",
    re.IGNORECASE,
)

_PROJECT_BULLET_RE = re.compile(
    r"(?:^|\n)\s*(?:[•\-\*\u2022\u25CF\u25CB\u2023]|\d+[.)]\s)\s*(.+)",
)

_PROJECT_TITLE_RE = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Za-z0-9\s\-:&/]{3,60}?)(?:\s*[-–—|:]?\s*|(?=\s*\n))",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_skills(text: str) -> list[str]:
    """
    Extract technical skills from resume text using regex matching
    against the skill taxonomy.

    Parameters
    ----------
    text : str
        The full resume text.

    Returns
    -------
    list[str]
        Deduplicated list of canonical skill names found in the text.
    """
    found: dict[str, str] = {}  # lowercase -> canonical
    text_lower = text.lower()

    for pattern, canonical in _SKILL_PATTERNS:
        if canonical.lower() in found:
            continue
        if pattern.search(text):
            found[canonical.lower()] = canonical

    return list(found.values())


def extract_projects(text: str) -> list[str]:
    """
    Heuristically extract project names from resume text.

    Handles the common resume formats:
      - "Project Name – Tech Stack (Python, ...) [link]"   (plain title line)
      - "• Project Name – description"                      (bullet format)
      - "Project Name | Stack"                              (pipe separator)
    """
    projects: list[str] = []
    seen: set[str] = set()

    def _clean_title(raw: str) -> str:
        """Strip tech stack, links, em-dashes and trailing noise from a project title."""
        # Replace replacement character (bad encoding of em-dash)
        raw = raw.replace("\ufffd", " – ")
        # Split on em-dash / en-dash / pipe — everything after is tech stack or desc
        core = re.split(r"\s*[–—|]\s*", raw)[0]
        # Remove parenthetical tech stack "(Python, FastAPI, ...)"
        core = re.sub(r"\s*\([^)]{0,80}\)", "", core)
        # Remove trailing [link], [GitHub], etc.
        core = re.sub(r"\s*\[.*?\]", "", core)
        # Remove trailing punctuation
        core = core.strip().rstrip(":.,–—-")
        return core

    def _add(name: str) -> None:
        name = _clean_title(name).strip()
        key = name.lower()
        # Must be 2+ words, at least 4 chars, no pure-noise fragments
        words = name.split()
        if len(name) >= 4 and len(words) >= 2 and key not in seen:
            seen.add(key)
            projects.append(name)

    # ── Step 1: isolate the Projects section ──────────────────────────────────
    section_match = _PROJECT_SECTION_RE.search(text)
    if section_match:
        start = section_match.end()
        next_section = _NEXT_SECTION_RE.search(text[start:])
        end = start + next_section.start() if next_section else len(text)
        project_section = text[start:end]
    else:
        project_section = text

    lines = [ln.strip() for ln in project_section.splitlines()]

    # ── Step 2: pick out title lines ──────────────────────────────────────────
    # In the target resume, each project starts with a non-bullet line that:
    #   - Contains the project name (Title Case or mixed)
    #   - Often has tech in parens and [link] at the end
    #   - Is followed by bullet/description lines
    _NOISE_STARTS = {
        "built", "trained", "developed", "implemented", "created",
        "designed", "wrote", "used", "applied", "worked", "fixed",
        "added", "achieved", "engineered", "consolidated", "refactored",
    }

    for line in lines:
        if not line:
            continue

        # Skip pure bullet description lines
        if re.match(r"^[•\-\*\u2022\u25CF\u25CB\u2023\d]", line):
            continue

        # Skip lines that start with a lowercase action verb (description fragment)
        first_word = line.split()[0].lower().rstrip(".")
        if first_word in _NOISE_STARTS:
            continue

        # Must have at least 2 words and start with uppercase
        words = line.split()
        if len(words) < 2 or not line[0].isupper():
            continue

        # Skip lines that look like section headers (single word or all caps short)
        if len(words) == 1:
            continue

        # Skip lines that are clearly statistics/scores  e.g. "54.66 SMAPE, Rank..."
        if re.match(r"^\d", line):
            continue

        _add(line)

    # ── Step 3: deduplicate and filter fragments ──────────────────────────────
    result = []
    for p in projects:
        words = p.split()
        # Filter out lines where the first word is a noise word (post-clean)
        if words[0].lower() not in _NOISE_STARTS and len(words) >= 2:
            result.append(p)

    return result[:12]


def get_skill_category(skill: str) -> Optional[str]:
    """Return the category a skill belongs to, or None if not found."""
    skill_lower = skill.lower()
    for category, skills_list in SKILL_TAXONOMY.items():
        for s in skills_list:
            if s.lower() == skill_lower:
                return category
    return None
