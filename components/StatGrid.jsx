'use client';
export default function StatGrid({ items }) {
  return (
    <div className="stat-grid">
      {items.map(({ label, value }) => (
        <div className="stat-card" key={label}>
          <div className="stat-value">{typeof value === 'number' ? value.toLocaleString() : (value ?? '—')}</div>
          <div className="stat-label">{label}</div>
        </div>
      ))}
    </div>
  );
}
