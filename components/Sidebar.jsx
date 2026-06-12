'use client';

import { useState, useEffect } from 'react';

const NAV_ITEMS = [
  { path: '/',          icon: '🏠', label: 'Home' },
  { path: '/dashboard', icon: '📊', label: 'Dashboard' },
  { path: '/explorer',  icon: '🔍', label: 'Explorer' },
  { path: '/repo',      icon: '📦', label: 'Repo Analyzer' },
];

export default function Sidebar({ currentPath }) {
  const [collapsed, setCollapsed] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle | running | error

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/backend/pipeline/status', { method: 'GET', signal: AbortSignal.timeout(2000) });
        if (res.ok) {
          const contentType = res.headers.get('content-type');
          if (contentType && contentType.includes('text/event-stream')) {
            setPipelineStatus('running');
          } else {
            const data = await res.json().catch(() => null);
            if (data && data.stage && data.stage !== 'idle' && data.stage !== 'completed') {
              setPipelineStatus('running');
            } else {
              setPipelineStatus('idle');
            }
          }
        }
      } catch {
        setPipelineStatus('idle');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const activePath = currentPath || (typeof window !== 'undefined' ? window.location.pathname : '/');

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="sidebar-header">
        <a href="/" className="sidebar-logo-link">
          <div className="sidebar-logo">
            <span className="sidebar-logo-icon">⚡</span>
            {!collapsed && <span className="sidebar-logo-text">HireFlow</span>}
          </div>
        </a>
        <button
          className="sidebar-toggle"
          onClick={() => setCollapsed(c => !c)}
          title={collapsed ? 'Expand' : 'Collapse'}
        >
          {collapsed ? '→' : '←'}
        </button>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(item => {
          const isActive = activePath === item.path || 
            (item.path !== '/' && activePath.startsWith(item.path));
          return (
            <a
              key={item.path}
              href={item.path}
              className={`sidebar-link ${isActive ? 'sidebar-active' : ''}`}
              title={collapsed ? item.label : undefined}
            >
              <span className="sidebar-link-icon">{item.icon}</span>
              {!collapsed && <span className="sidebar-link-label">{item.label}</span>}
              {isActive && <span className="sidebar-active-indicator" />}
            </a>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status">
          <span className={`sidebar-status-dot sidebar-status-${pipelineStatus}`} />
          {!collapsed && (
            <span className="sidebar-status-text">
              {pipelineStatus === 'running' ? 'Pipeline Running' :
               pipelineStatus === 'error' ? 'Pipeline Error' : 'Pipeline Idle'}
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
