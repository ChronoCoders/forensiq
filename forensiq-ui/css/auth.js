function getUser() {
  const u = sessionStorage.getItem('forensiq_user');
  return u ? JSON.parse(u) : null;
}

function isTokenExpired() {
  const exp = sessionStorage.getItem('forensiq_expires');
  if (!exp) return true;
  return new Date(exp) < new Date();
}

function checkAuth() {
  const token = getToken();
  if (!token || isTokenExpired()) {
    sessionStorage.clear();
    window.location.href = 'login.html';
    return false;
  }
  return true;
}

async function logout() {
  const token = getToken();
  if (token) {
    try {
      await fetch(`${AUTH_URL}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
    } catch (_) {}
  }
  sessionStorage.clear();
  window.location.href = 'login.html';
}

function injectUserInfo() {
  const user = getUser();
  if (!user) return;

  const nameEl = document.querySelector('.analyst-name');
  const roleEl = document.querySelector('.analyst-role');
  const avatarEl = document.querySelector('.analyst-avatar');

  if (nameEl) nameEl.textContent = user.username;
  if (roleEl) roleEl.textContent = user.role.replace('_', ' ');
  if (avatarEl) {
    const parts = user.username.split('.');
    avatarEl.textContent = parts.map(p => p[0]?.toUpperCase() || '').join('').substring(0, 2);
  }
}

function addLogoutButton() {
  const footer = document.querySelector('.sidebar-footer');
  if (!footer) return;

  const btn = document.createElement('button');
  btn.textContent = 'Sign Out';
  btn.style.cssText = `
    margin-top: 10px;
    width: 100%;
    background: transparent;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 7px;
    color: #4a5568;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.3px;
  `;
  btn.onmouseover = () => { btn.style.color = '#e2e6f0'; btn.style.borderColor = 'rgba(255,255,255,0.12)'; };
  btn.onmouseout = () => { btn.style.color = '#4a5568'; btn.style.borderColor = 'rgba(255,255,255,0.07)'; };
  btn.onclick = logout;
  footer.appendChild(btn);
}

document.addEventListener('DOMContentLoaded', () => {
  if (window.location.pathname.endsWith('login.html')) return;
  if (!checkAuth()) return;
  setTimeout(() => {
    injectUserInfo();
    addLogoutButton();
  }, 50);
});

// When auth.js is loaded dynamically (after DOMContentLoaded has already fired),
// the listener above never runs — trigger auth immediately instead.
if (document.readyState !== 'loading') {
  (function () {
    if (window.location.pathname.endsWith('login.html')) return;
    if (!checkAuth()) return;
    setTimeout(() => {
      injectUserInfo();
      addLogoutButton();
    }, 50);
  })();
}
