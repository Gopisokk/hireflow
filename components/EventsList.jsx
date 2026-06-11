'use client';

const EVENT_ICONS = {
  PushEvent:            '⬆️',
  PullRequestEvent:     '🔀',
  IssuesEvent:          '🐛',
  CreateEvent:          '✨',
  WatchEvent:           '⭐',
  ForkEvent:            '⑂',
  DeleteEvent:          '🗑️',
  IssueCommentEvent:    '💬',
  ReleaseEvent:         '🚀',
  PullRequestReviewEvent:'👁️',
  MemberEvent:          '👤',
  PublicEvent:          '🌐',
};

function fmtDate(d) {
  const dt = new Date(d);
  const now = new Date();
  const diff = Math.floor((now - dt) / 1000);
  if (diff < 60)        return `${diff}s ago`;
  if (diff < 3600)      return `${Math.floor(diff/60)}m ago`;
  if (diff < 86400)     return `${Math.floor(diff/3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.floor(diff/86400)}d ago`;
  return dt.toLocaleDateString('en-GB', { day:'numeric', month:'short' });
}

function eventLabel(e) {
  switch (e.type) {
    case 'PushEvent':       return `Pushed ${e.payload?.commits?.length ?? 0} commit(s)`;
    case 'CreateEvent':     return `Created ${e.payload?.ref_type} ${e.payload?.ref || ''}`.trim();
    case 'ForkEvent':       return `Forked to ${e.payload?.forkee?.full_name || ''}`;
    case 'WatchEvent':      return 'Starred repo';
    case 'IssuesEvent':     return `${e.payload?.action} issue #${e.payload?.issue?.number}`;
    case 'PullRequestEvent':return `${e.payload?.action} PR #${e.payload?.pull_request?.number}`;
    case 'ReleaseEvent':    return `Released ${e.payload?.release?.tag_name}`;
    case 'DeleteEvent':     return `Deleted ${e.payload?.ref_type} ${e.payload?.ref}`;
    default:                return e.type.replace('Event', '');
  }
}

export default function EventsList({ events }) {
  if (!events?.length) {
    return <p style={{ color: 'var(--muted)', fontSize: 13 }}>No recent public events.</p>;
  }

  return (
    <div>
      {events.slice(0, 20).map((e, i) => (
        <div className="event-row" key={i}>
          <div className="event-icon">{EVENT_ICONS[e.type] || '📌'}</div>
          <div className="event-type">{e.type.replace('Event', '')}</div>
          <div className="event-repo">
            <strong style={{ color: 'var(--text)' }}>{e.repo?.name}</strong>
            <span style={{ color: 'var(--muted)', marginLeft: 8, fontSize: 12 }}>
              {eventLabel(e)}
            </span>
          </div>
          <div className="event-date">{fmtDate(e.created_at)}</div>
        </div>
      ))}
    </div>
  );
}
