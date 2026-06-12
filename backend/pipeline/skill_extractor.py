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
    r"references?|summary|objective|contact|languages)\s*[:\-]?\s*\n",
    re.IGNORECASE,
)

_PROJECT_BULLET_RE = re.compile(
    r"(?:^|\n)\s*(?:[•\-\*\u2022\u25CF\u25CB\u2023]|\d+[.)]\s)\s*(.+)",
)

_PROJECT_TITLE_RE = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Za-z0-9\s\-:&/]{3,60}?)(?:\s*[-–—|:]\s*|\s*\n)",
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

    Strategy:
    1. Find the "Projects" section by matching common headers.
    2. Extract until the next section header.
    3. Pull project names from bullet points or title-like lines.

    Parameters
    ----------
    text : str
        The full resume text.

    Returns
    -------
    list[str]
        List of project names (best-effort extraction).
    """
    projects: list[str] = []

    # Try to find the projects section
    section_match = _PROJECT_SECTION_RE.search(text)
    if section_match:
        start = section_match.end()
        # Find where the next section begins
        next_section = _NEXT_SECTION_RE.search(text[start:])
        end = start + next_section.start() if next_section else len(text)
        project_section = text[start:end]
    else:
        # Fallback: scan the entire text for project-like patterns
        project_section = text

    # Extract from bullet points (most common resume format)
    seen: set[str] = set()
    for match in _PROJECT_BULLET_RE.finditer(project_section):
        line = match.group(1).strip()
        # Take the first sentence / phrase (before a dash or colon as description)
        name_match = re.match(r"^([^:\-–—|]{3,60}?)(?:\s*[-–—|:]\s*|$)", line)
        if name_match:
            name = name_match.group(1).strip()
            # Skip lines that are just descriptions (too many lowercase words)
            words = name.split()
            if len(words) <= 8 and name.lower() not in seen:
                seen.add(name.lower())
                projects.append(name)

    # If we didn't find bullets in a dedicated section, try title patterns
    if not projects and section_match:
        for match in _PROJECT_TITLE_RE.finditer(project_section):
            name = match.group(1).strip()
            if len(name) > 3 and name.lower() not in seen:
                seen.add(name.lower())
                projects.append(name)

    return projects[:20]  # Cap at 20 to avoid noise


def get_skill_category(skill: str) -> Optional[str]:
    """Return the category a skill belongs to, or None if not found."""
    skill_lower = skill.lower()
    for category, skills_list in SKILL_TAXONOMY.items():
        for s in skills_list:
            if s.lower() == skill_lower:
                return category
    return None
