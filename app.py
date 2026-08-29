"""
SiteSense 5G - GeoAI decision-support dashboard
Team Neural Shield (KH-002) - ASEAN GeoAI Fusion 2026

A Streamlit + Leafmap dashboard that fuses tower locations, population and
settlement data to:
  (1) map estimated coverage and quantify people / kampungs in estimated
      underserved areas (coverage is a range-based OpenCelliD proxy, not
      verified operator coverage);
  (2) flag overloaded or underperforming towers;
  (3) recommend preliminary locations for a new 5G tower.

Run locally:
    py -m streamlit run app.py

Layout inspiration: "ANDROMEDA" network-analytics dashboard
  (left filter sidebar - central map - right status panel).
"""

import hmac
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --- Map backend: Leafmap (folium backend) with a plain-folium fallback ----
# leafmap.foliumap.Map subclasses folium.Map, so st_folium can render either.
try:
    import leafmap.foliumap as leafmap
    HAVE_LEAFMAP = True
except Exception:  # pragma: no cover - fallback if leafmap unavailable
    import folium
    HAVE_LEAFMAP = False

from streamlit_folium import st_folium
import streamlit.components.v1 as components

import coverage as cov
import recommend as rec
import data_access as da

# --- Paths + scopes ---------------------------------------------------------
# The data layer (towers/population/villages/coverage-gap/site-recommendation
# loading, PostGIS-with-file-fallback) lives in data_access.py so the FastAPI
# service (api/main.py) can reuse the exact same logic instead of a second
# copy. app.py only adds Streamlit's @st.cache_data around it below.
APP_DIR = da.APP_DIR
CSV = da.CSV
SCOPES = da.SCOPES
RADIO_META = da.RADIO_META

# --- FastAPI service URL (api/main.py's /predict-rsrp demo) ----------------
# Same override pattern as db.py's DATABASE_URL: env var, then st.secrets,
# then a localhost default for local dev.
import os as _os
LSTM_API_URL = _os.environ.get("LSTM_API_URL")
if not LSTM_API_URL:
    try:
        LSTM_API_URL = st.secrets["LSTM_API_URL"]
    except Exception:
        LSTM_API_URL = "http://127.0.0.1:8001"

# --- Access gate (boss asked to restrict the public dashboard) -------------
# Same override pattern as above: env var, then st.secrets. If no password is
# configured (local dev with no .env), the gate is skipped so it never locks
# out a fresh checkout.
def _env_or_secret(key: str, default: str | None = None) -> str | None:
    val = _os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

APP_USERNAME = _env_or_secret("APP_USERNAME", "admin")
APP_PASSWORD = _env_or_secret("APP_PASSWORD")

# --- On-map legend (matches team-reviewed design: 7-item network legend) ---
LEGEND_HTML = """
<div style="
    position: fixed; bottom: 26px; right: 26px; z-index: 9999;
    background: rgba(17,21,28,0.92); border: 1px solid #232a36; border-radius: 12px;
    padding: 12px 15px; font-family: 'Source Sans Pro', sans-serif;
    color: #f1f4f8; font-size: 12.5px; line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,.45);
">
  <div style="font-weight:700; margin-bottom:6px; color:#cfd6e0;">Network Legend</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#e53935;margin-right:8px;"></span>5G NR</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#1e88e5;margin-right:8px;"></span>4G LTE</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#9c27b0;margin-right:8px;"></span>3G UMTS</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#9e9e9e;margin-right:8px;"></span>2G GSM</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:transparent;border:2px solid #ff1744;margin-right:8px;"></span>Overloaded Tower</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#ff9800;margin-right:8px;"></span>Est. Underserved Village</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;
       background:#00c853;margin-right:8px;"></span>Preliminary 5G Site</div>
</div>
"""


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
# Thin @st.cache_data wrappers around data_access.py's framework-agnostic
# functions — the actual loading/PostGIS-fallback/algorithm logic lives
# there so api/main.py (FastAPI) can call the same code, not a copy.
load_towers = st.cache_data(show_spinner=False)(da.load_towers)
_using_db = st.cache_data(show_spinner=False)(da.using_db)
load_population = st.cache_data(show_spinner=False)(da.load_population)
load_places = st.cache_data(show_spinner=False)(da.load_places)
compute_gap = st.cache_data(show_spinner="Computing coverage gap…")(da.compute_gap)
compute_sites = st.cache_data(show_spinner="Scoring candidate sites…")(da.compute_sites)


# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
_authenticated = st.session_state.get("authenticated", False)

# Browser-tab favicon — the SiteSense cell-tower mark as an inline SVG data URI,
# on a dark rounded tile so it reads on both light and dark browser tabs. Renders
# identically everywhere (no OS/emoji dependence).
import base64 as _b64
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="15 26 90 90">'
    '<defs><linearGradient id="fav" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#2dd4bf"/>'
    '</linearGradient></defs>'
    '<g stroke="url(#fav)" stroke-width="7" stroke-linecap="round" fill="none">'
    '<path d="M45 40 a 24 24 0 0 0 0 36" opacity="0.92"/>'
    '<path d="M75 40 a 24 24 0 0 1 0 36" opacity="0.92"/></g>'
    '<circle cx="60" cy="52" r="9" fill="url(#fav)"/>'
    '<g stroke="url(#fav)" stroke-width="7" stroke-linecap="round" stroke-linejoin="round" fill="none">'
    '<path d="M60 60 L60 98"/><path d="M46 100 L60 64 L74 100"/></g>'
    '</svg>'
)
_FAVICON = "data:image/svg+xml;base64," + _b64.b64encode(_FAVICON_SVG.encode()).decode()

st.set_page_config(
    page_title="SiteSense 5G - Penang",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded" if _authenticated else "collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.6rem; padding-bottom: 1rem;
        padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }
      .app-title { font-size: 2rem; font-weight: 700; line-height: 1.35;
        color: #f1f4f8; margin: 0 0 2px 0; padding-top: 4px; }
      .app-sub { color: #8b96a5; font-size: 0.85rem; margin: 0 0 10px 0; }
      .kpi-card {
        background: #11151c; border: 1px solid #232a36; border-radius: 12px;
        padding: 14px 16px; height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
      }
      .kpi-label { color: #8b96a5; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: .04em; margin-bottom: 4px; }
      .kpi-value { color: #f1f4f8; font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
      .kpi-sub { color: #6f7b8a; font-size: 0.72rem; margin-top: 2px; }
      .panel {
        background: #11151c; border: 1px solid #232a36; border-radius: 12px;
        padding: 14px 16px; margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
      }
      .panel h4 { margin: 0 0 8px 0; font-size: 0.9rem; color: #cfd6e0; }
      .ai-insight {
        background: #0e1a17; border: 1px solid #1c3b32; border-radius: 10px;
        padding: 12px 14px; color: #bfe8d8; font-size: 0.85rem;
      }
      .todo { color: #d9a441; }

      /* --- Network-health donut (ANDROMEDA reference) --- */
      .donut-row { display: flex; align-items: center; gap: 16px; margin-top: 10px; }
      .donut { width: 78px; height: 78px; border-radius: 50%; flex-shrink: 0; position: relative; }
      .donut::after {
        content: ""; position: absolute; inset: 11px; border-radius: 50%; background: #11151c;
      }
      .donut-legend { font-size: .74rem; color: #b7c0cc; line-height: 1.9; }
      .donut-legend b { color: #f1f4f8; }
      .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }

      /* --- Site-score gradient bar (Stanfield Land reference) --- */
      .score-bar {
        position: relative; height: 6px; border-radius: 3px; margin: 5px 0 3px 0;
        background: linear-gradient(90deg, #ff1744 0%, #ff9800 50%, #00e676 100%);
      }
      .score-marker {
        position: absolute; top: -3px; width: 3px; height: 12px; border-radius: 1px;
        background: #ffffff; box-shadow: 0 0 4px rgba(0,0,0,.7);
      }

      /* --- Login gate --- */
      /* .st-key-login_card targets the st.container(key="login_card") below
         (Streamlit >=1.31 stamps a matching class on the container's div),
         so the header text and the form fields render inside one card. */
      .st-key-login_card {
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        z-index: 3; width: min(94vw, 540px);
        background: linear-gradient(160deg, rgba(18,23,32,0.97), rgba(8,10,15,0.985));
        backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
        border: 1px solid rgba(34,211,238,0.42); border-radius: 20px;
        padding: 30px 48px 46px 48px;
        box-shadow: 0 24px 70px rgba(0,0,0,.7);
      }
      .login-icon { font-size: 6rem; line-height: 1; text-align: center; margin-bottom: 4px; }
      .login-title { font-size: 2.5rem; font-weight: 800; color: #f1f4f8;
        text-align: center; margin: 0 0 4px 0; letter-spacing: .3px; }
      .login-sub { color: #9aa6b4; font-size: 0.95rem; text-align: center;
        margin: 0 0 14px 0; line-height: 1.5; }
      .st-key-login_card div[data-testid="stForm"] { border: none; padding: 0; }
      .st-key-login_card div[data-testid="stTextInputRootElement"] {
        background: rgba(23,28,38,0.85); border: 1px solid #2a3242; border-radius: 9px;
      }
      .st-key-login_card div[data-testid="stTextInputRootElement"]:focus-within {
        border-color: #22d3ee; box-shadow: 0 0 0 2px rgba(34,211,238,0.18);
      }
      .st-key-login_card input[data-testid="stTextInputField"] { color: #f1f4f8; font-size: 1rem; }
      /* hide Streamlit's "Press Enter to submit form" hint inside the login form */
      .st-key-login_card [data-testid="InputInstructions"] { display: none; }
      .st-key-login_card button[data-testid^="stBaseButton"] {
        background: linear-gradient(135deg, #0891b2, #0e7490);
        border: none; color: #ffffff; font-weight: 700; font-size: 1.05rem;
        padding: 0.72rem 1rem; border-radius: 10px; margin-top: 6px;
        box-shadow: 0 4px 16px rgba(8,145,178,0.35); transition: all .15s ease;
      }
      .st-key-login_card button[data-testid^="stBaseButton"]:hover {
        background: linear-gradient(135deg, #0aa6cc, #0f8296);
        box-shadow: 0 6px 22px rgba(8,145,178,0.5); transform: translateY(-1px);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# Interactive telecom-network canvas (login page only). The component runs in a
# same-origin iframe and injects a full-viewport <canvas> into the parent
# document, so it renders as a live, mouse-reactive background behind the card.
_LOGIN_CANVAS = """
<script>
(function(){
  const win = window.parent, doc = win.document;
  if (win.__ssNetRaf) { win.cancelAnimationFrame(win.__ssNetRaf); win.__ssNetRaf = null; }
  const prev = doc.getElementById('ss-net-canvas'); if (prev) prev.remove();
  const c = doc.createElement('canvas');
  c.id = 'ss-net-canvas';
  c.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;';
  doc.body.insertBefore(c, doc.body.firstChild);
  const ctx = c.getContext('2d');
  let W=0,H=0,DPR=1, pts=[], mouse={x:-9999,y:-9999};
  function size(){ DPR=Math.min(win.devicePixelRatio||1,2); W=win.innerWidth; H=win.innerHeight;
    c.width=W*DPR; c.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0); }
  function init(){ const n=Math.max(45, Math.min(95, Math.floor(W*H/16000))); pts=[];
    for(let i=0;i<n;i++){ pts.push({x:Math.random()*W,y:Math.random()*H,
      vx:(Math.random()-0.5)*0.35, vy:(Math.random()-0.5)*0.35}); } }
  size(); init();
  win.addEventListener('resize', function(){ size(); init(); });
  win.addEventListener('mousemove', function(e){ mouse.x=e.clientX; mouse.y=e.clientY; });
  win.addEventListener('mouseout', function(){ mouse.x=-9999; mouse.y=-9999; });
  const D=145, MD=210;
  function frame(){
    ctx.fillStyle='#0a0e15'; ctx.fillRect(0,0,W,H);
    for(const p of pts){ p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>W) p.vx*=-1; if(p.y<0||p.y>H) p.vy*=-1;
      const dx=mouse.x-p.x, dy=mouse.y-p.y, d=Math.hypot(dx,dy);
      if(d<MD && d>0.1){ p.x+=dx/d*0.5; p.y+=dy/d*0.5; } }
    for(let i=0;i<pts.length;i++){ for(let j=i+1;j<pts.length;j++){
      const a=pts[i], b=pts[j], dx=a.x-b.x, dy=a.y-b.y, d=Math.hypot(dx,dy);
      if(d<D){ ctx.strokeStyle='rgba(34,211,238,'+(0.16*(1-d/D))+')'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); } } }
    if(mouse.x>-9000){ for(const p of pts){ const dx=mouse.x-p.x, dy=mouse.y-p.y, d=Math.hypot(dx,dy);
      if(d<MD){ ctx.strokeStyle='rgba(45,212,191,'+(0.38*(1-d/MD))+')'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(mouse.x,mouse.y); ctx.lineTo(p.x,p.y); ctx.stroke(); } } }
    for(const p of pts){ ctx.fillStyle='rgba(130,205,225,0.8)';
      ctx.beginPath(); ctx.arc(p.x,p.y,1.7,0,6.2832); ctx.fill(); }
    if(mouse.x>-9000){ ctx.fillStyle='rgba(34,211,238,0.95)';
      ctx.beginPath(); ctx.arc(mouse.x,mouse.y,3,0,6.2832); ctx.fill(); }
    win.__ssNetRaf = win.requestAnimationFrame(frame);
  }
  frame();
})();
</script>
"""


# Custom SiteSense 5G mark — a cell tower broadcasting signal, in the project's
# cyan/teal gradient. Inline SVG so it renders identically everywhere (unlike the
# platform-dependent 📡 emoji) and stays crisp at any size.
_LOGO_SVG = """
<svg width="104" height="104" viewBox="0 0 120 120" fill="none"
     xmlns="http://www.w3.org/2000/svg"
     style="display:block;margin:0 auto;filter:drop-shadow(0 3px 10px rgba(34,211,238,0.30))">
  <defs>
    <linearGradient id="ssgLogo" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22d3ee"/>
      <stop offset="1" stop-color="#2dd4bf"/>
    </linearGradient>
  </defs>
  <g stroke="url(#ssgLogo)" stroke-width="6" stroke-linecap="round" fill="none">
    <path d="M44 36 a 26 26 0 0 0 0 40" opacity="0.9"/>
    <path d="M31 27 a 42 42 0 0 0 0 58" opacity="0.42"/>
    <path d="M76 36 a 26 26 0 0 1 0 40" opacity="0.9"/>
    <path d="M89 27 a 42 42 0 0 1 0 58" opacity="0.42"/>
  </g>
  <circle cx="60" cy="50" r="9" fill="url(#ssgLogo)"/>
  <g stroke="url(#ssgLogo)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M60 59 L60 100"/>
    <path d="M45 102 L60 62 L75 102"/>
  </g>
</svg>
"""


def require_login() -> None:
    """Gate the whole dashboard behind a branded login form. Skipped when no
    APP_PASSWORD is configured (fresh local checkout with no .env)."""
    if _authenticated or not APP_PASSWORD:
        return

    # Transparent app background so the interactive network canvas (injected by
    # the component below) shows through behind the glassy login card.
    st.markdown(
        "<style>.stApp,[data-testid='stAppViewContainer'],[data-testid='stMain'],"
        "section.main,[data-testid='stHeader']{background:transparent !important;}</style>",
        unsafe_allow_html=True,
    )
    components.html(_LOGIN_CANVAS, height=0)

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        with st.container(key="login_card"):
            st.markdown(
                '<div class="login-icon">' + _LOGO_SVG + '</div>'
                '<div class="login-title">SiteSense 5G</div>'
                '<div class="login-sub">Enter your credentials to access the '
                'SiteSense 5G dashboard</div>',
                unsafe_allow_html=True,
            )
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="Username")
                password = st.text_input("Password", type="password",
                                         placeholder="Enter your password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                user_ok = hmac.compare_digest(username, APP_USERNAME)
                pass_ok = hmac.compare_digest(password, APP_PASSWORD)
                if user_ok and pass_ok:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

    st.stop()


require_login()


# --- Dashboard icon set: consistent inline-SVG line icons (cyan accent) that
# replace the platform-dependent emojis, matching the login logo's style. ---
_CYAN = "#22d3ee"


def _svg(body: str, color: str = _CYAN, size: str = "1.15em", mr: str = "0.45em") -> str:
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
            f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:-0.22em;margin-right:{mr}">'
            f'{body}</svg>')


ICON = {
    "tower": _svg('<circle cx="12" cy="6" r="1"/><path d="M12 7v13"/>'
                  '<path d="M8 20l4-9 4 9"/><path d="M8.3 4.3a5 5 0 0 0 0 5.4"/>'
                  '<path d="M15.7 4.3a5 5 0 0 1 0 5.4"/>'),
    "people": _svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
                   '<circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
                   '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>'),
    "village": _svg('<path d="M3 10.5L12 3l9 7.5"/><path d="M5 9.8V21h14V9.8"/>'
                    '<path d="M10 21v-5h4v5"/>'),
    "flame": _svg('<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.07-2.14-.22-4.05 '
                  '2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.15.43-2.29 '
                  '1-3a2.5 2.5 0 0 0 2.5 2.5z"/>', color="#ff7a45"),
    "chart": _svg('<path d="M6 20v-6"/><path d="M12 20V4"/><path d="M18 20v-9"/>'),
    "bulb": _svg('<path d="M9 18h6"/><path d="M10 22h4"/>'
                 '<path d="M8 15a6 6 0 1 1 8 0c-.8.7-1.3 1.5-1.5 2.5H9.5c-.2-1-.7-1.8-1.5-2.5z"/>'),
    "trend": _svg('<path d="M22 7l-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>'),
}

# Small gradient tower mark for the sidebar header (reuses the login logo).
_SIDEBAR_LOGO = _LOGO_SVG.replace(
    'width="104" height="104"', 'width="30" height="30"'
).replace('display:block;margin:0 auto;', 'display:inline-block;vertical-align:middle;')


def kpi_card(col, label: str, value: str, sub: str = "", icon: str = "") -> None:
    prefix = f'{icon}&nbsp;' if icon else ""
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{prefix}{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def donut_widget(good: int, warn: int, err: int) -> str:
    """CSS-only conic-gradient donut (no charting library), health-breakdown
    style borrowed from the ANDROMEDA dashboard reference."""
    total = max(1, good + warn + err)
    p_good = good / total * 100
    p_warn = warn / total * 100
    gradient = (
        f"conic-gradient(#00e676 0% {p_good:.1f}%, "
        f"#ff9800 {p_good:.1f}% {p_good + p_warn:.1f}%, "
        f"#ff1744 {p_good + p_warn:.1f}% 100%)"
    )
    return (
        f'<div class="donut-row">'
        f'<div class="donut" style="background:{gradient}"></div>'
        f'<div class="donut-legend">'
        f'<span class="dot" style="background:#00e676"></span>good <b>{good:,}</b><br>'
        f'<span class="dot" style="background:#ff9800"></span>warning <b>{warn:,}</b><br>'
        f'<span class="dot" style="background:#ff1744"></span>error <b>{err:,}</b>'
        f'</div></div>'
    )


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
if not CSV.exists():
    st.error(f"Cannot find 502.csv at:\n{CSV}")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar - scope + filters (Function inputs)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:9px;margin:0 0 6px 0">'
        f'{_SIDEBAR_LOGO}'
        f'<span style="font-size:1.5rem;font-weight:700;color:#f1f4f8">SiteSense 5G</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("GeoAI tower-siting & 5G coverage-gap planner")
    st.divider()

    scope = st.radio(
        "Study area", list(SCOPES.keys()), index=0,
        help="Switch between the full Penang State (island + mainland "
             "Seberang Perai) and Penang Island only.",
    )
    st.caption(f"Scope: {SCOPES[scope]['note']}")
    st.caption(f"Data source: {'🟢 PostGIS' if _using_db() else '📁 local files'}")
    df = load_towers(scope)
    st.divider()

    st.markdown("### Filters")
    radios_present = [r for r in RADIO_META if r in set(df["radio"])]
    radio_labels = {RADIO_META[r][0]: r for r in radios_present}
    picked_labels = st.multiselect(
        "Network generation",
        options=list(radio_labels.keys()),
        default=[RADIO_META[r][0] for r in radios_present if r in ("NR", "LTE")],
    )
    picked = [radio_labels[l] for l in picked_labels] or radios_present

    rng_max = int(df["range"].max())
    max_range = st.slider(
        "Max coverage range (m)", 100, rng_max, rng_max, step=100,
        help="Reported cell range from OpenCelliD (coverage footprint proxy). "
             "Defaults to include all towers; lower it to exclude cells with "
             "implausibly large reported range.",
    )

    load_p = st.slider(
        "Overloaded threshold (load percentile)", 50, 99, 90, step=1,
        help="Cells above this 'samples' percentile are flagged as overloaded.",
    )

    show_coverage = st.checkbox("Show coverage footprints", value=False,
                                help="Draw each cell's range as a circle (slower).")
    show_gaps = st.checkbox("Show estimated underserved villages", value=True,
                            help="Marks OSM places outside every tower's estimated coverage "
                                 "footprint (OpenCelliD reported range) — not verified operator "
                                 "coverage data.")

    st.markdown("### Preliminary new-site recommendations")
    weight_profile = st.selectbox(
        "Site-scoring profile", list(rec.WEIGHT_PROFILES.keys()), index=0,
        help="Weighted multi-criteria scoring (population reached, backhaul "
             "proximity to existing towers, overloaded-tower relief). "
             "'Coverage-first' reproduces the original population-only ranking. "
             "Weights are team-set/expert-judgement, not learned from data.",
    )
    with st.expander("Profile weights"):
        w = rec.WEIGHT_PROFILES[weight_profile]
        st.write(f"Population reached: **{w['population']:.0%}** · "
                f"Backhaul proximity: **{w['backhaul']:.0%}** · "
                f"Overload relief: **{w['overload']:.0%}** · "
                f"Slope penalty: **{w['slope']:.0%}**")
        st.caption("Backhaul proximity to an existing tower is used as a rough cost "
                  "proxy above the ranking; each site's estimated capex (below) uses "
                  "a real cited industry benchmark instead. Slope uses real SRTM "
                  "elevation data. Weights are team-set/expert-judgement, not learned.")
    new_range = st.slider("New 5G tower range (m)", 300, 3000, 1700, step=100,
                          help="Assumed coverage radius of a new 5G tower.")
    n_sites = st.slider("Sites to recommend", 1, 15, 5)
    show_sites = st.checkbox("Show preliminary recommended sites", value=True)

    st.divider()
    st.caption(
        "Data: OpenCelliD Malaysia (MCC 502). "
        f"Map: {'Leafmap' if HAVE_LEAFMAP else 'Folium (leafmap not installed)'}"
    )

# --------------------------------------------------------------------------
# Apply filters + derive metrics
# --------------------------------------------------------------------------
fdf = df[df["radio"].isin(picked) & (df["range"] <= max_range)].copy()

load_threshold = fdf["samples"].quantile(load_p / 100.0)
fdf["overloaded"] = fdf["samples"] >= load_threshold
overloaded = fdf[fdf["overloaded"]]

n_towers = len(fdf)
n_5g = int((fdf["radio"] == "NR").sum())
n_4g = int((fdf["radio"] == "LTE").sum())
n_overloaded = len(overloaded)

# --- Function 1: estimated coverage gap (people + villages underserved) --
gap = compute_gap(scope, tuple(sorted(picked)), int(max_range))
places = load_places(scope)

# --- Function 3: recommended new-5G-tower sites ---------------------------
sites = compute_sites(scope, tuple(sorted(picked)), int(max_range), int(load_p),
                      int(new_range), int(n_sites), weight_profile)

# --------------------------------------------------------------------------
# Header + KPI row
# --------------------------------------------------------------------------
st.markdown(
    f'<div class="app-title">SiteSense 5G — Coverage &amp; Site-Selection Overview</div>'
    f'<div class="app-sub">{scope} pilot ({SCOPES[scope]["note"]}) · fuse towers + '
    f'population + settlements → an estimated coverage gap and preliminary new-site picks '
    f'(not verified operator coverage)</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Cells in view", f"{n_towers:,}", f"{n_4g:,} × 4G · {n_5g:,} × 5G", icon=ICON["tower"])
kpi_card(k2, "Est. people underserved", f"{gap['uncovered_pop']:,.0f}",
         f"{gap['pct_covered']:.1f}% of {gap['total_pop']:,.0f} est. covered", icon=ICON["people"])
kpi_card(k3, "Kampungs — est. underserved", f"{gap['n_uncovered_places']:,}",
         f"of {gap['n_places']:,} OSM places in view", icon=ICON["village"])
overloaded_sub = (
    f"≥ {load_p}th pct load ({int(load_threshold)} samples)"
    if n_towers else "no cells match the current filter"
)
kpi_card(k4, "Overloaded towers", f"{n_overloaded:,}", overloaded_sub, icon=ICON["flame"])

st.write("")

# --------------------------------------------------------------------------
# Main: map (left) + status panel (right)
# --------------------------------------------------------------------------
map_col, panel_col = st.columns([3, 1], gap="small")

with map_col:
    _center = SCOPES[scope]["center"]
    _zoom = SCOPES[scope]["zoom"]
    if HAVE_LEAFMAP:
        m = leafmap.Map(center=_center, zoom=_zoom,
                        draw_control=False, measure_control=False,
                        fullscreen_control=True)
        m.add_basemap("CartoDB.Positron")
    else:
        m = folium.Map(location=_center, zoom_start=_zoom, tiles="cartodbpositron")

    import folium as _folium  # available via either path
    m.get_root().html.add_child(_folium.Element(LEGEND_HTML))
    for radio in ["GSM", "UMTS", "LTE", "NR"]:  # draw important ones last (on top)
        sub = fdf[fdf["radio"] == radio]
        if sub.empty:
            continue
        label, color = RADIO_META[radio]
        for _, r in sub.iterrows():
            is_hot = bool(r.get("overloaded", False))
            if show_coverage and pd.notna(r["range"]):
                _folium.Circle(
                    location=[r["lat"], r["lon"]], radius=float(r["range"]),
                    color=color, weight=1, fill=True, fill_opacity=0.05,
                ).add_to(m)
            _folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=3, color=color, weight=1,
                fill=True, fill_opacity=0.9,
                popup=_folium.Popup(
                    f"<b>{label}</b>"
                    f"{' · <span style=color:#ff1744>OVERLOADED</span>' if is_hot else ''}"
                    f"<br>range: {r['range']:.0f} m"
                    f"<br>load (samples): {r['samples']:.0f}"
                    f"<br>cell: {r['cell']}",
                    max_width=240,
                ),
            ).add_to(m)
            if is_hot:
                # Hollow red ring on top, matching the "Overloaded Tower" legend
                # swatch exactly — keeps the base dot's generation colour visible
                # instead of overwriting it with solid red.
                _folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=8, color="#ff1744", weight=2,
                    fill=False, opacity=0.95,
                ).add_to(m)

    # Function 1 — mark the villages/places that fall outside all coverage
    if show_gaps:
        uncov = ~gap["villages_covered_mask"]
        for lon, lat, nm, pl in zip(
            places["lon"][uncov], places["lat"][uncov],
            np.asarray(places["name"])[uncov], np.asarray(places["place"])[uncov],
        ):
            _folium.CircleMarker(
                location=[lat, lon], radius=6,
                color="#ffffff", weight=2,
                fill=True, fill_color="#ff9800", fill_opacity=0.95,
                popup=_folium.Popup(
                    f"<b>{nm or '(unnamed)'}</b><br>{pl}<br>"
                    f"<span style='color:#e65100'>ESTIMATED UNDERSERVED (4G/5G)</span>",
                    max_width=220,
                ),
            ).add_to(m)

    # Function 3 — recommended new-5G-tower sites (numbered, green)
    if show_sites:
        for s in sites:
            _folium.Circle(
                location=[s["lat"], s["lon"]], radius=new_range,
                color="#00e676", weight=1, fill=True, fill_opacity=0.10,
            ).add_to(m)
            _folium.Marker(
                location=[s["lat"], s["lon"]],
                icon=_folium.DivIcon(html=(
                    f'<div style="background:#00c853;color:#04160b;font-weight:800;'
                    f'width:24px;height:24px;line-height:24px;text-align:center;'
                    f'border-radius:50%;border:2px solid #fff;'
                    f'box-shadow:0 0 4px rgba(0,0,0,.5)">{s["rank"]}</div>'
                )),
                popup=_folium.Popup(
                    f"<b>Preliminary 5G site #{s['rank']}</b><br>"
                    f"+{s['people_gained']:,.0f} people in estimated new coverage<br>"
                    f"relieves {s['overloaded_relieved']} overloaded tower(s)<br>"
                    f"<span style='color:#666'>{s['lat']:.4f}, {s['lon']:.4f}</span>",
                    max_width=250,
                ),
            ).add_to(m)

    st_folium(m, use_container_width=True, height=960, returned_objects=[])

with panel_col:
    # --- Status (network-health donut, ANDROMEDA-style) ---
    # "Good" = under the overload threshold; "Warning" = overloaded but under
    # 2x threshold; "Error" = severely overloaded (>= 2x threshold) — reuses
    # the same load_threshold Function 2 already computes, just bucketed.
    if n_towers:
        n_error = int((fdf["samples"] >= load_threshold * 2).sum())
        n_warning = n_overloaded - n_error
        n_good = n_towers - n_overloaded
    else:
        n_good = n_warning = n_error = 0
    st.markdown(
        f'<div class="panel"><h4>Status</h4>'
        f'<div style="color:#8b96a5;font-size:.8rem">Cells in view</div>'
        f'<div style="color:#f1f4f8;font-size:1.3rem;font-weight:700">{n_towers:,}</div>'
        f'<div style="margin-top:6px">'
        f'<span style="color:#1e88e5">● 4G {n_4g:,}</span>&nbsp;&nbsp;'
        f'<span style="color:#e53935">● 5G {n_5g:,}</span>&nbsp;&nbsp;'
        f'<span style="color:#ff1744">● hot {n_overloaded:,}</span>'
        f'</div>'
        f'{donut_widget(n_good, n_warning, n_error)}'
        f'<div style="margin-top:6px;color:#5b6472;font-size:.68rem;font-style:italic">'
        f'Health = load vs. the {load_p}th-percentile overload threshold '
        f'(warning &lt; 2× · error ≥ 2×).</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # NOTE: a separate "Flagged towers" detail table used to live here
    # (Function 2). Removed as redundant for the live pitch — the overloaded
    # count is already shown above (Status panel "hot" count) and in the
    # "Overloaded towers" KPI card, and per-site relief is called out in the
    # AI Insights panel below ("relieves N overloaded tower(s)").

    # --- Coverage gap (Function 1) ---
    st.markdown(
        f'<div class="panel"><h4>{ICON["chart"]}Estimated coverage gap (4G/5G)</h4>'
        f'<div style="color:#f1f4f8;font-size:1.3rem;font-weight:700">'
        f'{gap["uncovered_pop"]:,.0f}</div>'
        f'<div style="color:#8b96a5;font-size:.78rem">people in estimated underserved areas · '
        f'{100 - gap["pct_covered"]:.1f}% of {scope}</div>'
        f'<div style="margin-top:8px;color:#ff9800">● {gap["n_uncovered_places"]} '
        f'villages/kampungs — estimated underserved</div>'
        f'<div style="margin-top:6px;color:#5b6472;font-size:.68rem;font-style:italic">'
        f'Estimated from each cell\'s reported OpenCelliD range, not verified operator '
        f'coverage data.</div></div>',
        unsafe_allow_html=True,
    )
    if gap["n_uncovered_places"]:
        uncov = ~gap["villages_covered_mask"]
        names = [n for n in np.asarray(places["name"])[uncov] if n]
        with st.expander(f"List {gap['n_uncovered_places']} estimated underserved places"):
            st.write(", ".join(sorted(names)) or "(unnamed places only)")

    # --- AI insight (Function 3: recommended sites) ---
    if sites:
        covered_total = sites[-1]["cumulative_gained"]
        pct_gap = covered_total / gap["uncovered_pop"] * 100 if gap["uncovered_pop"] else 0
        PHASE_COLOR = {1: "#00e676", 2: "#ff9800", 3: "#7f8a99"}
        cards = [
            f'<div style="margin:6px 0;padding-bottom:6px;'
            f'border-bottom:1px solid #1c3b32">'
            f'<b style="color:#00e676">#{s["rank"]}</b> '
            f'<span style="color:{PHASE_COLOR.get(s.get("phase"), "#7f8a99")};font-size:.72rem;'
            f'border:1px solid currentColor;border-radius:4px;padding:0 4px;margin-left:4px">'
            f'Phase {s.get("phase", "?")}</span>'
            f'<div class="score-bar"><div class="score-marker" '
            f'style="left:{max(0.0, min(1.0, s.get("score", 0.0))) * 100:.1f}%"></div></div>'
            f'+{s["people_gained"]:,.0f} people'
            f'{f" · relieves {s['overloaded_relieved']} overloaded" if s["overloaded_relieved"] else ""}'
            f'{f" · {s['backhaul_distance_m']:,.0f} m to nearest existing tower" if s.get("backhaul_distance_m") is not None else ""}'
            f'{f" · {s['slope_deg']:.1f}° slope" if s.get("slope_deg") is not None else ""}'
            f'<br><span style="color:#8b96a5;font-size:.72rem">'
            f'est. ${s["est_capex_usd"]:,.0f} · {s["people_per_1000usd"]:.1f} people per $1,000</span>'
            f'<br><span style="color:#7f8a99;font-size:.75rem">'
            f'{s["lat"]:.4f}, {s["lon"]:.4f}</span></div>'
            for s in sites
        ]
        rows_first = "".join(cards[:3])
        rows_rest = "".join(cards[3:])
        more_html = ""
        if len(cards) > 3:
            more_html = (
                '<details style="margin-top:2px">'
                '<summary style="cursor:pointer;color:#22d3ee;font-size:.78rem;'
                'padding:6px 0 2px 0">'
                f'Show {len(cards) - 3} more sites</summary>'
                f'<div style="margin-top:4px">{rows_rest}</div></details>'
            )
        total_cost = sum(s["est_capex_usd"] for s in sites)
        st.markdown(
            f'<div class="panel"><h4>{ICON["bulb"]}AI Insights — preliminary 5G site recommendations</h4>'
            f'<div class="ai-insight">Building these <b>{len(sites)}</b> towers '
            f'(range {new_range} m, <b>{weight_profile}</b> profile, est. total capex '
            f'<b>${total_cost:,.0f}</b>) would reach an estimated '
            f'<b>{covered_total:,.0f}</b> of the {gap["uncovered_pop"]:,.0f} people in estimated '
            f'underserved areas (<b>{pct_gap:.0f}%</b> of the gap).'
            f'<div style="margin-top:8px">{rows_first}</div>{more_html}'
            f'<div style="margin-top:8px;color:#5b6472;font-size:.68rem;font-style:italic">'
            f'Weighted multi-criteria score (population reached · backhaul proximity to existing '
            f'towers · overload relief · SRTM slope penalty) — weights are team-set/expert-judgement, '
            f'not learned from data. Capex = $150k base + $62.5k/km backhaul fiber (industry '
            f'benchmark, PatentPC 2026 — order-of-magnitude estimate, not a site-specific quote). '
            f'Phase = rollout order by cost-efficiency (people reached per dollar), not selection '
            f'rank. Slope is a buildability proxy, not a full RF propagation model.'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="panel"><h4>{ICON["bulb"]}AI Insights</h4>'
            '<div class="ai-insight">No estimated coverage gap to close in the current '
            'filter.</div></div>',
            unsafe_allow_html=True,
        )

    # --- Bonus: LSTM network-trend demo, real Penang data ------------------
    # Calls the separate FastAPI service over HTTP (not in-process) — the
    # Streamlit app stays free of the heavy TensorFlow dependency, matching
    # the finale's Streamlit-frontend / FastAPI-service split. Trained on
    # real Penang tile history (Ookla Open Data, quarterly) — no
    # KL-transfer-learning caveat needed, this model has actually seen
    # Penang. See ml/train_lstm_penang.py for the KL-vs-Penang mapping.
    st.markdown(f'<div class="panel"><h4>{ICON["trend"]}LSTM Network-Trend Demo (Penang, real data)</h4>',
               unsafe_allow_html=True)
    st.caption("Deployable ML pipeline demo — predicts next quarter's average mobile "
              "download throughput for a real Penang map tile from its last 4 quarters "
              "of real measurements (Ookla Open Data). Separate FastAPI service, not "
              "part of the coverage/site-recommendation logic above.")
    if st.button("▶ Run Penang prediction", key="lstm_penang_btn"):
        try:
            import requests
            base = LSTM_API_URL
            sample = requests.get(f"{base}/sample-request-penang", timeout=5).json()
            resp = requests.post(f"{base}/predict-penang-network",
                                 json={"observations": sample["observations"]}, timeout=15)
            resp.raise_for_status()
            r = resp.json()
            actual = sample.get("actual_next_avg_d_kbps")
            actual_line = (f'<div style="color:#8b96a5;font-size:.78rem;margin-top:2px">'
                          f'actual next quarter: {actual/1000:.1f} Mbps</div>' if actual else "")
            st.markdown(
                f'<div style="margin-top:4px">'
                f'<span style="color:#f1f4f8;font-size:1.3rem;font-weight:700">'
                f'{r["predicted_mbps"]} Mbps</span>&nbsp;&nbsp;'
                f'<span style="color:#00e676">● predicted download</span>'
                f'{actual_line}'
                f'<div style="color:#8b96a5;font-size:.78rem;margin-top:4px">'
                f'tile {sample["quadkey"]} · {r["response_time_ms"]:.0f} ms · {r["model_version"]}</div>'
                f'<div style="margin-top:6px;color:#5b6472;font-size:.68rem;font-style:italic">'
                f'{r["note"]}</div></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.warning(
                f"LSTM API not reachable at {LSTM_API_URL} ({e}). Start it with:\n\n"
                f"`.venv-ml/Scripts/python.exe -m uvicorn api.main:app --port 8001`"
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # NOTE: an earlier KL-trained signal-quality LSTM demo panel used to live
    # here. Removed from the live dashboard (2026-08-20, user request) as
    # unnecessary for the finale demo now that the Penang model above exists —
    # the KL model/training script/notebook are kept in the repo (ml/) and
    # api/main.py's /predict-rsrp endpoint still works if referenced in Q&A,
    # just not shown in the UI.

st.caption(
    "Function 1 (estimated underserved population/kampungs) ✓ · Function 2 (overloaded flags) ✓ "
    "· Function 3 (greedy max-coverage new-5G-site recommendations) ✓ — all live. "
    "MVP complete."
)
