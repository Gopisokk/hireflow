export const runtime = 'nodejs';

const GQL_QUERY = `
query($login: String!) {
  user(login: $login) {
    name
    login
    email
    bio
    company
    location
    websiteUrl
    avatarUrl
    createdAt
    updatedAt
    followers { totalCount }
    following  { totalCount }
    starredRepositories { totalCount }

    contributionsCollection {
      totalCommitContributions
      totalPullRequestContributions
      totalIssueContributions
      totalRepositoryContributions
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
            color
          }
        }
      }
    }

    repositories(first: 30, orderBy: { field: UPDATED_AT, direction: DESC }) {
      totalCount
      nodes {
        name
        description
        url
        isFork
        isPrivate
        isArchived
        parent { nameWithOwner url }
        stargazerCount
        forkCount
        createdAt
        updatedAt
        diskUsage
        primaryLanguage { name }
        languages(first: 10) {
          edges {
            size
            node { name }
          }
        }
        repositoryTopics(first: 10) {
          nodes { topic { name } }
        }
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 1) {
                totalCount
                nodes {
                  message
                  committedDate
                  author { name email }
                }
              }
            }
          }
        }
      }
    }

    pinnedItems(first: 6) {
      nodes {
        ... on Repository {
          name
          description
          url
          stargazerCount
          primaryLanguage { name }
        }
      }
    }
  }
}`;

export async function POST(req) {
  try {
    const { username, token } = await req.json();

    if (!username || !token) {
      return Response.json({ error: 'username and token are required' }, { status: 400 });
    }

    const headers = {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    };

    // 1. GraphQL — profile + repos + contributions (single request)
    const gqlRes = await fetch('https://api.github.com/graphql', {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: GQL_QUERY, variables: { login: username } }),
    });

    if (!gqlRes.ok) {
      const txt = await gqlRes.text();
      return Response.json({ error: `GraphQL ${gqlRes.status}: ${txt}` }, { status: gqlRes.status });
    }

    const gqlJson = await gqlRes.json();
    if (gqlJson.errors) {
      return Response.json({ error: gqlJson.errors.map((e) => e.message).join(' | ') }, { status: 400 });
    }

    // 2. REST — user profile (extra fields not in GraphQL)
    const restRes = await fetch(`https://api.github.com/users/${username}`, { headers });
    if (!restRes.ok) {
      return Response.json({ error: `REST ${restRes.status}` }, { status: restRes.status });
    }
    const restUser = await restRes.json();

    // 3. REST — recent events
    const eventsRes = await fetch(
      `https://api.github.com/users/${username}/events?per_page=20`,
      { headers }
    );
    const events = eventsRes.ok ? await eventsRes.json() : [];

    return Response.json({
      gql: gqlJson.data,
      rest: restUser,
      events,
    });
  } catch (err) {
    return Response.json({ error: err.message }, { status: 500 });
  }
}
