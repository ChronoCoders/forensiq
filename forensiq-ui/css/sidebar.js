// Core auth constants — defined synchronously so all pages can use them
// immediately without waiting for auth.js to load.
const AUTH_URL = 'http://localhost:3000';
const API_URL  = 'http://localhost:3000/api';

function getToken() {
  return sessionStorage.getItem('forensiq_token');
}

// Load the rest of auth.js (checkAuth, logout, injectUserInfo, etc.)
(function () {
  const s = document.createElement('script');
  s.src = 'css/auth.js';
  document.head.appendChild(s);
})();

const NAV = [
  {
    label: 'Analysis',
    items: [
      { key: 'index',         label: 'Case Overview',   href: 'index.html' },
      { key: 'evidence',      label: 'Evidence',        href: 'evidence.html' },
      { key: 'contradictions',label: 'Contradictions',  href: 'contradictions.html' },
      { key: 'timeline',      label: 'Timeline',        href: 'timeline.html' },
      { key: 'relationships', label: 'Relationships',   href: 'relationships.html' },
    ],
  },
  {
    label: 'System',
    items: [
      { key: 'audit-log', label: 'Audit Log', href: 'audit-log.html' },
      { key: 'ingest',    label: 'Ingest',    href: 'ingest.html' },
      { key: 'reports',   label: 'Reports',   href: 'reports.html' },
    ],
  },
  {
    label: 'Cases',
    items: [
      { key: 'cases-active',   label: 'Active',     href: 'cases-active.html' },
      { key: 'cases-archived', label: 'Archived',   href: 'cases-archived.html' },
    ],
  },
];

function renderSidebar(activeKey) {
  const sections = NAV.map(section => {
    const items = section.items.map(item => {
      const active = item.key === activeKey ? ' active' : '';
      const badge = item.badge
        ? `<span class="nav-badge">${item.badge}</span>`
        : '';
      return `<a class="nav-item${active}" href="${item.href}">
        <div class="nav-dot"></div>
        ${item.label}${badge}
      </a>`;
    }).join('');

    return `<div class="nav-section">
      <div class="nav-label">${section.label}</div>
      ${items}
    </div>`;
  }).join('');

  const html = `<aside class="sidebar">
    <div class="logo">
      <div class="logo-mark">FOREN<span>SIQ</span></div>
      <div class="logo-sub">Evidence Analysis</div>
    </div>
    <nav class="nav">${sections}</nav>
    <div class="sidebar-footer">
      <div class="analyst-info">
        <div class="analyst-avatar">--</div>
        <div>
          <div class="analyst-name analyst-name">Analyst</div>
          <div class="analyst-role analyst-role">--</div>
        </div>
      </div>
    </div>
  </aside>`;

  const mount = document.getElementById('sidebar-mount');
  if (mount) mount.outerHTML = html;
}
