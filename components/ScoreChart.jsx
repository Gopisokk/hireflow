'use client';

import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  ArcElement,
} from 'chart.js';
import { Bar, Radar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  ArcElement
);

const CHART_COLORS = {
  accent: '#6366f1',
  accent2: '#8b5cf6',
  cyan: '#06b6d4',
  green: '#10b981',
  yellow: '#f59e0b',
  red: '#ef4444',
  text: '#e2e8f0',
  text2: '#94a3b8',
  muted: '#475569',
  surface: '#0f1623',
  surface2: '#161f30',
  border: 'rgba(255,255,255,0.07)',
};

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: CHART_COLORS.text2,
        font: { family: 'Inter, sans-serif', size: 11 },
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: CHART_COLORS.surface,
      borderColor: CHART_COLORS.border,
      borderWidth: 1,
      titleColor: CHART_COLORS.text,
      bodyColor: CHART_COLORS.text2,
      titleFont: { family: 'Inter, sans-serif', weight: '700' },
      bodyFont: { family: 'Inter, sans-serif' },
      padding: 12,
      cornerRadius: 8,
    },
  },
};

/* ─── Score Distribution Bar Chart ─────────────────────── */
function ScoreDistribution({ candidates }) {
  const data = useMemo(() => {
    const ranges = ['0-20', '20-40', '40-60', '60-80', '80-100'];
    const atsCounts = [0, 0, 0, 0, 0];
    const githubCounts = [0, 0, 0, 0, 0];
    const finalCounts = [0, 0, 0, 0, 0];

    candidates.forEach(c => {
      const ats = c.ats_score;
      const gh = c.github_score;
      const fin = c.final_score;

      const bucket = (v) => {
        if (v === null || v === undefined) return -1;
        if (v < 20) return 0;
        if (v < 40) return 1;
        if (v < 60) return 2;
        if (v < 80) return 3;
        return 4;
      };

      const ai = bucket(ats);
      const gi = bucket(gh);
      const fi = bucket(fin);
      if (ai >= 0) atsCounts[ai]++;
      if (gi >= 0) githubCounts[gi]++;
      if (fi >= 0) finalCounts[fi]++;
    });

    return {
      labels: ranges,
      datasets: [
        {
          label: 'ATS Score',
          data: atsCounts,
          backgroundColor: 'rgba(99,102,241,0.7)',
          borderColor: '#6366f1',
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: 'GitHub Score',
          data: githubCounts,
          backgroundColor: 'rgba(6,182,212,0.7)',
          borderColor: '#06b6d4',
          borderWidth: 1,
          borderRadius: 6,
        },
        {
          label: 'Final Score',
          data: finalCounts,
          backgroundColor: 'rgba(16,185,129,0.7)',
          borderColor: '#10b981',
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    };
  }, [candidates]);

  const options = {
    ...commonOptions,
    scales: {
      x: {
        grid: { color: CHART_COLORS.border },
        ticks: { color: CHART_COLORS.text2, font: { family: 'Inter, sans-serif', size: 11 } },
      },
      y: {
        grid: { color: CHART_COLORS.border },
        ticks: { color: CHART_COLORS.text2, font: { family: 'Inter, sans-serif', size: 11 }, stepSize: 1 },
        beginAtZero: true,
      },
    },
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Score Distribution',
        color: CHART_COLORS.text,
        font: { family: 'Inter, sans-serif', size: 14, weight: '700' },
        padding: { bottom: 16 },
      },
    },
  };

  return (
    <div className="chart-container">
      <Bar data={data} options={options} />
    </div>
  );
}

/* ─── Score Radar Chart ────────────────────────────────── */
function ScoreRadar({ candidates }) {
  const data = useMemo(() => {
    const labels = ['ATS Score', 'GitHub Score', 'Keyword Match', 'Active Days', 'Commit Score', 'Project Match'];

    // Use top 5 candidates
    const top5 = [...candidates]
      .sort((a, b) => (b.final_score || 0) - (a.final_score || 0))
      .slice(0, 5);

    const colors = [
      { bg: 'rgba(99,102,241,0.2)', border: '#6366f1' },
      { bg: 'rgba(6,182,212,0.2)', border: '#06b6d4' },
      { bg: 'rgba(16,185,129,0.2)', border: '#10b981' },
      { bg: 'rgba(245,158,11,0.2)', border: '#f59e0b' },
      { bg: 'rgba(139,92,246,0.2)', border: '#8b5cf6' },
    ];

    const datasets = top5.map((c, i) => ({
      label: c.name || `Candidate ${i + 1}`,
      data: [
        c.ats_score || 0,
        c.github_score || 0,
        c.ats_breakdown?.keyword_match || 0,
        Math.min((c.github_breakdown?.active_days || 0) / 3.65, 100),
        Math.min((c.github_breakdown?.total_commits || 0) / 10, 100),
        c.ats_breakdown?.project_match || 0,
      ],
      backgroundColor: colors[i % colors.length].bg,
      borderColor: colors[i % colors.length].border,
      borderWidth: 2,
      pointBackgroundColor: colors[i % colors.length].border,
      pointBorderColor: '#0f1623',
      pointBorderWidth: 2,
      pointRadius: 3,
    }));

    return { labels, datasets };
  }, [candidates]);

  const options = {
    ...commonOptions,
    scales: {
      r: {
        angleLines: { color: CHART_COLORS.border },
        grid: { color: CHART_COLORS.border },
        pointLabels: { color: CHART_COLORS.text2, font: { family: 'Inter, sans-serif', size: 11 } },
        ticks: { display: false },
        suggestedMin: 0,
        suggestedMax: 100,
      },
    },
    plugins: {
      ...commonOptions.plugins,
      title: {
        display: true,
        text: 'Top 5 Candidates Comparison',
        color: CHART_COLORS.text,
        font: { family: 'Inter, sans-serif', size: 14, weight: '700' },
        padding: { bottom: 16 },
      },
    },
  };

  return (
    <div className="chart-container chart-container-radar">
      <Radar data={data} options={options} />
    </div>
  );
}

/* ─── Algorithm Comparison ─────────────────────────────── */
function AlgorithmComparison({ candidates }) {
  const data = useMemo(() => {
    const avgAts = candidates.length
      ? candidates.reduce((s, c) => s + (c.ats_score || 0), 0) / candidates.length
      : 0;
    const avgGithub = candidates.length
      ? candidates.reduce((s, c) => s + (c.github_score || 0), 0) / candidates.length
      : 0;
    const avgFinal = candidates.length
      ? candidates.reduce((s, c) => s + (c.final_score || 0), 0) / candidates.length
      : 0;
    const maxAts = candidates.length
      ? Math.max(...candidates.map(c => c.ats_score || 0))
      : 0;
    const maxGithub = candidates.length
      ? Math.max(...candidates.map(c => c.github_score || 0))
      : 0;

    return {
      labels: ['Avg ATS', 'Avg GitHub', 'Avg Final', 'Max ATS', 'Max GitHub'],
      datasets: [
        {
          label: 'Scores',
          data: [avgAts, avgGithub, avgFinal, maxAts, maxGithub],
          backgroundColor: [
            'rgba(99,102,241,0.75)',
            'rgba(6,182,212,0.75)',
            'rgba(16,185,129,0.75)',
            'rgba(139,92,246,0.75)',
            'rgba(245,158,11,0.75)',
          ],
          borderColor: [
            '#6366f1', '#06b6d4', '#10b981', '#8b5cf6', '#f59e0b',
          ],
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    };
  }, [candidates]);

  const options = {
    ...commonOptions,
    indexAxis: 'y',
    scales: {
      x: {
        grid: { color: CHART_COLORS.border },
        ticks: { color: CHART_COLORS.text2, font: { family: 'Inter, sans-serif', size: 11 } },
        beginAtZero: true,
        max: 100,
      },
      y: {
        grid: { color: CHART_COLORS.border },
        ticks: { color: CHART_COLORS.text2, font: { family: 'Inter, sans-serif', size: 11 } },
      },
    },
    plugins: {
      ...commonOptions.plugins,
      legend: { display: false },
      title: {
        display: true,
        text: 'Score Overview',
        color: CHART_COLORS.text,
        font: { family: 'Inter, sans-serif', size: 14, weight: '700' },
        padding: { bottom: 16 },
      },
    },
  };

  return (
    <div className="chart-container">
      <Bar data={data} options={options} />
    </div>
  );
}

/* ─── Exported Wrapper ─────────────────────────────────── */
export default function ScoreChart({ candidates = [], type = 'distribution' }) {
  if (!candidates.length) {
    return (
      <div className="chart-container chart-empty">
        <span className="chart-empty-icon">📊</span>
        <span>No data available for charts</span>
      </div>
    );
  }

  switch (type) {
    case 'distribution':
      return <ScoreDistribution candidates={candidates} />;
    case 'radar':
      return <ScoreRadar candidates={candidates} />;
    case 'comparison':
      return <AlgorithmComparison candidates={candidates} />;
    default:
      return <ScoreDistribution candidates={candidates} />;
  }
}
