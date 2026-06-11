'use client';
import { useEffect, useRef } from 'react';

export default function ContribCalendar({ weeks, stats }) {
  const canvasRef = useRef(null);

  const days    = weeks.flatMap(w => w.contributionDays);
  const active  = days.filter(d => d.contributionCount > 0).length;
  const max     = Math.max(...days.map(d => d.contributionCount), 1);
  const peakDay = days.reduce((a, b) => b.contributionCount > a.contributionCount ? b : a, days[0]);
  const avgPerWeek = weeks.length > 0
    ? (days.reduce((s, d) => s + d.contributionCount, 0) / weeks.length).toFixed(1)
    : 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !weeks.length) return;

    const CELL = 12;
    const GAP  = 2;
    const STEP = CELL + GAP;
    const LEFT = 28;
    const TOP  = 20;
    const W    = LEFT + weeks.length * STEP + 10;
    const H    = TOP + 7 * STEP + 20;

    canvas.width  = W;
    canvas.height = H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // Day labels
    ['S','M','T','W','T','F','S'].forEach((d, i) => {
      if (i % 2 === 1) {
        ctx.fillStyle = '#475569';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText(d, 8, TOP + i * STEP + CELL - 1);
      }
    });

    // Month labels
    let lastMonth = -1;
    weeks.forEach((week, wi) => {
      if (week.contributionDays.length > 0) {
        const m = new Date(week.contributionDays[0].date).getMonth();
        if (m !== lastMonth) {
          const names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
          ctx.fillStyle = '#64748b';
          ctx.font = '9px Inter, sans-serif';
          ctx.fillText(names[m], LEFT + wi * STEP, 11);
          lastMonth = m;
        }
      }
    });

    // Cells
    weeks.forEach((week, wi) => {
      week.contributionDays.forEach(day => {
        const dow = new Date(day.date).getDay();
        const x   = LEFT + wi * STEP;
        const y   = TOP + dow * STEP;

        const count = day.contributionCount;
        let fillColor;
        if (count === 0) {
          fillColor = '#161f30';
        } else {
          // Use GitHub color if provided, else derive intensity
          fillColor = day.color || (count > 0 ? interpolateColor(count, max) : '#161f30');
        }

        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.roundRect(x, y, CELL, CELL, 2);
        ctx.fill();
      });
    });
  }, [weeks]);

  return (
    <div>
      <div className="cal-canvas-wrap">
        <canvas ref={canvasRef} />
      </div>

      <div className="cal-legend" style={{ marginTop: 10 }}>
        <span>Less</span>
        <div className="cal-legend-cells">
          {['#161f30','#0e4429','#006d32','#26a641','#39d353'].map((c, i) => (
            <div key={i} className="cal-legend-cell" style={{ background: c }} />
          ))}
        </div>
        <span>More</span>
      </div>

      <div className="stat-grid" style={{ marginTop: 20 }}>
        <div className="stat-card">
          <div className="stat-value">{stats.totalContributions.toLocaleString()}</div>
          <div className="stat-label">Total Contributions</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{active}</div>
          <div className="stat-label">Active Days / Year</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{avgPerWeek}</div>
          <div className="stat-label">Avg Contributions / Week</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{max}</div>
          <div className="stat-label">Peak Single Day</div>
        </div>
      </div>

      {peakDay && (
        <p style={{ fontSize: 12, color: 'var(--text2)', marginTop: 12 }}>
          🔥 Busiest day: <strong style={{ color: 'var(--text)' }}>{peakDay.date}</strong>
          {' '}with <strong style={{ color: 'var(--text)' }}>{peakDay.contributionCount}</strong> contributions
        </p>
      )}
    </div>
  );
}

function interpolateColor(count, max) {
  const shades = ['#0e4429','#006d32','#26a641','#39d353'];
  const idx = Math.min(Math.floor((count / max) * shades.length), shades.length - 1);
  return shades[idx];
}
