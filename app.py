"""Streamlit frontend for the fake news detector.

Run:  streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src import detector, database, news_fetcher

st.set_page_config(
    page_title="Veritas — truth, instantly.",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Chart palette (CSS palette must match)
C_FG = "#ededed"
C_MUTED = "#737373"
C_BORDER = "rgba(255,255,255,0.08)"
C_FAKE = "#f87171"
C_REAL = "#4ade80"
C_UNCERTAIN = "#fbbf24"
C_ACCENT = "#a78bfa"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, sans-serif", color=C_FG, size=12),
    margin=dict(l=20, r=20, t=30, b=20),
)


# ═════════════════════════ STYLES ═════════════════════════
# Helper: inject raw HTML/CSS using st.html when available (Streamlit ≥1.33),
# falling back to st.markdown for older versions.
def _raw_html(html: str) -> None:
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


_raw_html("""
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700;800&family=Geist+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #050505;
    --bg-2: #0a0a0a;
    --panel: #0d0d0d;
    --fg: #ededed;
    --muted: #737373;
    --dim: #a3a3a3;
    --border: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.16);
    --accent: #a78bfa;
    --accent-2: #60a5fa;
    --accent-3: #f472b6;
    --fake: #f87171;
    --real: #4ade80;
    --uncertain: #fbbf24;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Geist', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.011em;
}
.stApp { background: var(--bg); overflow-x: hidden; }

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }

/* ───── Top Nav ───── */
.nav {
    position: sticky; top: 0; z-index: 100;
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 48px;
    background: rgba(5,5,5,0.7);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-bottom: 1px solid var(--border);
}
.nav-brand { display: flex; align-items: center; gap: 10px; font-weight: 500; }
.nav-mark {
    width: 22px; height: 22px; background: var(--fg);
    transform: rotate(45deg); border-radius: 3px;
    box-shadow: 0 0 20px rgba(167,139,250,0.4);
}
.nav-name {
    font-family: 'Instrument Serif', serif !important;
    font-size: 1.35rem; font-style: italic;
}
.nav-links { display: flex; gap: 32px; }
.nav-links a {
    color: var(--muted); text-decoration: none; font-size: 0.88rem;
    font-weight: 500; transition: color 0.15s ease;
}
.nav-links a:hover { color: var(--fg); }
.nav-cta {
    background: var(--fg); color: #050505 !important;
    padding: 8px 16px; border-radius: 100px; font-size: 0.85rem;
    font-weight: 600; text-decoration: none;
    transition: transform 0.15s ease;
    display: inline-block;
}
.nav-cta:hover { transform: scale(1.04); }

/* ───── Section container ───── */
.section {
    max-width: 1200px;
    margin: 0 auto;
    padding: 120px 48px;
    position: relative;
}
.section.tight { padding: 60px 48px; }

/* ───── Hero ───── */
.hero {
    position: relative;
    padding: 140px 48px 100px 48px;
    max-width: 1200px;
    margin: 0 auto;
    text-align: center;
    z-index: 2;
}
.hero-bg {
    position: absolute; inset: 0; z-index: -1;
    overflow: hidden; pointer-events: none;
}
.hero-bg::before {
    content: ""; position: absolute; inset: -20%;
    background:
        radial-gradient(ellipse 60% 45% at 20% 20%, rgba(167,139,250,0.25), transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 30%, rgba(96,165,250,0.18), transparent 60%),
        radial-gradient(ellipse 55% 45% at 50% 80%, rgba(244,114,182,0.14), transparent 60%);
    animation: breathe 16s ease-in-out infinite;
    filter: blur(40px);
}
.hero-grid {
    position: absolute; inset: 0; z-index: -1; pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 60% 70% at 50% 40%, #000 40%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse 60% 70% at 50% 40%, #000 40%, transparent 80%);
}
@keyframes breathe {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.9; }
    50% { transform: translate(2%, -1%) scale(1.05); opacity: 1; }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-pill {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 6px 14px 6px 6px;
    border: 1px solid var(--border);
    border-radius: 100px;
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
    font-size: 0.78rem; color: var(--dim);
    margin-bottom: 32px;
    animation: fadeUp 0.7s ease-out;
}
.hero-pill-tag {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff; padding: 3px 9px; border-radius: 100px;
    font-size: 0.7rem; font-weight: 600;
}
.hero-pill-dot {
    width: 7px; height: 7px; background: var(--real);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--real);
    animation: float 2s ease-in-out infinite;
}

.hero-h1 {
    font-family: 'Instrument Serif', serif !important;
    font-size: clamp(3rem, 7vw, 6.5rem);
    font-weight: 400; line-height: 0.98;
    letter-spacing: -0.035em;
    margin: 0 auto 24px auto;
    max-width: 14ch;
    animation: fadeUp 0.8s ease-out 0.1s backwards;
}
.hero-h1 em {
    font-style: italic;
    background: linear-gradient(120deg, var(--accent-3), var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-lede {
    font-size: 1.15rem; color: var(--dim);
    max-width: 560px; margin: 0 auto 40px auto;
    line-height: 1.55;
    animation: fadeUp 0.8s ease-out 0.25s backwards;
}
.hero-ctas {
    display: flex; gap: 12px; justify-content: center;
    animation: fadeUp 0.8s ease-out 0.4s backwards;
}
.btn-primary, .btn-secondary {
    padding: 12px 22px; border-radius: 100px;
    font-size: 0.92rem; font-weight: 600;
    text-decoration: none;
    transition: all 0.2s ease;
    display: inline-flex; align-items: center; gap: 8px;
}
.btn-primary {
    background: var(--fg); color: #050505;
    box-shadow: 0 4px 24px rgba(255,255,255,0.15);
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(255,255,255,0.22); }
.btn-secondary {
    background: rgba(255,255,255,0.04); color: var(--fg);
    border: 1px solid var(--border-strong);
}
.btn-secondary:hover { background: rgba(255,255,255,0.08); }

/* Hero stats strip */
.hero-stats {
    display: flex; gap: 60px; justify-content: center;
    margin-top: 80px; padding-top: 40px;
    border-top: 1px solid var(--border);
    animation: fadeUp 0.8s ease-out 0.55s backwards;
}
.hero-stat { text-align: center; }
.hero-stat-num {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2.6rem; line-height: 1; color: var(--fg);
    background: linear-gradient(135deg, #fff, #888);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-stat-label {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-top: 10px;
}

/* ───── Marquee / news ticker ───── */
.marquee-wrap {
    padding: 20px 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    background: var(--bg-2);
    overflow: hidden;
    position: relative;
}
.marquee-wrap::before, .marquee-wrap::after {
    content: ""; position: absolute; top: 0; bottom: 0;
    width: 120px; z-index: 2; pointer-events: none;
}
.marquee-wrap::before { left: 0; background: linear-gradient(90deg, var(--bg-2), transparent); }
.marquee-wrap::after { right: 0; background: linear-gradient(-90deg, var(--bg-2), transparent); }
.marquee-track {
    display: flex; gap: 48px;
    animation: scroll-x 50s linear infinite;
    width: max-content;
}
.marquee-item {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.82rem; color: var(--dim);
    white-space: nowrap;
}
.marquee-item .tag {
    font-size: 0.68rem; padding: 2px 8px;
    border: 1px solid var(--border); border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.marquee-item .tag.real { color: var(--real); border-color: rgba(74,222,128,0.35); }
.marquee-item .tag.fake { color: var(--fake); border-color: rgba(248,113,113,0.35); }
.marquee-item .tag.uncertain { color: var(--uncertain); border-color: rgba(251,191,36,0.35); }
@keyframes scroll-x {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* ───── Section header ───── */
.section-eyebrow {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.2em;
    margin-bottom: 20px;
    display: inline-flex; align-items: center; gap: 10px;
}
.section-eyebrow::before {
    content: ""; width: 24px; height: 1px; background: var(--muted);
}
.section-h2 {
    font-family: 'Instrument Serif', serif !important;
    font-size: clamp(2.2rem, 4vw, 3.8rem);
    font-weight: 400; line-height: 1.05;
    letter-spacing: -0.025em;
    margin: 0 0 20px 0;
    max-width: 18ch;
}
.section-h2 em { font-style: italic; color: var(--muted); }
.section-lede {
    color: var(--dim); font-size: 1.05rem; line-height: 1.6;
    max-width: 560px;
}

/* ───── Feature grid ───── */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin-top: 60px;
}
@media (max-width: 900px) { .feat-grid { grid-template-columns: 1fr; } }
.feat-card {
    padding: 32px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005));
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
    min-height: 240px;
}
.feat-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    opacity: 0; transition: opacity 0.3s ease;
}
.feat-card:hover {
    border-color: var(--border-strong);
    transform: translateY(-3px);
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
}
.feat-card:hover::before { opacity: 1; }
.feat-num {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.18em;
    margin-bottom: 48px;
}
.feat-icon {
    width: 44px; height: 44px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 1.5rem;
    background: linear-gradient(135deg, rgba(167,139,250,0.2), rgba(96,165,250,0.2));
    border: 1px solid rgba(167,139,250,0.3);
    margin-bottom: 20px;
    color: var(--fg);
}
.feat-title {
    font-size: 1.15rem; font-weight: 500;
    margin: 0 0 8px 0; color: var(--fg);
    letter-spacing: -0.015em;
}
.feat-desc { color: var(--dim); font-size: 0.92rem; line-height: 1.55; }

/* ───── Pipeline ───── */
.pipe-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0; margin-top: 60px;
    position: relative;
}
@media (max-width: 1000px) { .pipe-grid { grid-template-columns: 1fr; gap: 12px; } }
.pipe-step {
    padding: 28px 24px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.015);
    position: relative;
    transition: all 0.2s ease;
}
.pipe-step:first-child { border-radius: 14px 0 0 14px; }
.pipe-step:last-child { border-radius: 0 14px 14px 0; }
.pipe-step + .pipe-step { border-left: none; }
@media (max-width: 1000px) {
    .pipe-step { border-radius: 14px !important; border-left: 1px solid var(--border) !important; }
}
.pipe-step:hover {
    background: rgba(255,255,255,0.04);
    border-color: var(--border-strong);
    z-index: 2;
}
.pipe-num {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 2.4rem; line-height: 1;
    color: var(--accent); margin-bottom: 16px;
}
.pipe-title { font-size: 1rem; font-weight: 500; margin-bottom: 8px; }
.pipe-desc { color: var(--dim); font-size: 0.86rem; line-height: 1.5; }
.pipe-tag {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.65rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-top: 14px; display: block;
}

/* ───── Detector card ───── */
.detector-wrap {
    max-width: 860px; margin: 0 auto;
    padding: 40px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.005)),
        var(--panel);
    position: relative;
    box-shadow: 0 30px 100px rgba(0,0,0,0.5);
}
.detector-wrap::before {
    content: ""; position: absolute; inset: -1px; z-index: -1;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(167,139,250,0.35), transparent 30%, transparent 70%, rgba(96,165,250,0.25));
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    padding: 1px;
    opacity: 0.8;
}
.detector-title {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.18em;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 10px;
}
.detector-title::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%;
    background: var(--real); box-shadow: 0 0 8px var(--real);
    animation: float 2s ease-in-out infinite;
}

/* ───── Verdict display ───── */
.verdict-wrap {
    padding: 36px 0 28px 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin: 28px 0;
}
.verdict-label {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.2em; color: var(--muted);
    margin-bottom: 14px;
}
.verdict-value {
    font-family: 'Instrument Serif', serif !important;
    font-size: 4.5rem; font-weight: 400;
    line-height: 1; letter-spacing: -0.03em;
}
.verdict-value.fake { color: var(--fake); }
.verdict-value.real { color: var(--real); }
.verdict-value.uncertain { color: var(--uncertain); }

.dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:8px; vertical-align:middle;}
.dot.fake { background: var(--fake); box-shadow: 0 0 8px var(--fake); }
.dot.real { background: var(--real); box-shadow: 0 0 8px var(--real); }
.dot.uncertain { background: var(--uncertain); box-shadow: 0 0 8px var(--uncertain); }

.probbar {
    width: 100%; height: 2px; background: rgba(255,255,255,0.08);
    border-radius: 2px; margin: 24px 0 8px 0;
}
.probbar-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; background: var(--fg); }
.probbar-fill.fake { background: linear-gradient(90deg, var(--fake), #fca5a5); box-shadow: 0 0 12px var(--fake); }
.probbar-fill.real { background: linear-gradient(90deg, var(--real), #86efac); box-shadow: 0 0 12px var(--real); }
.probbar-fill.uncertain { background: linear-gradient(90deg, var(--uncertain), #fde68a); box-shadow: 0 0 12px var(--uncertain); }
.probbar-meta {
    display: flex; justify-content: space-between;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* ───── Signal rows ───── */
.signal-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.signal-row:last-child { border-bottom: none; }
.signal-key {
    color: var(--muted); font-family: 'Geist Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.72rem;
}
.signal-val {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.95rem; color: var(--fg);
}
.signal-val.muted { color: var(--muted); }

.reason-line {
    padding: 10px 0; color: #c6c6c6; font-size: 0.92rem;
    border-bottom: 1px dashed var(--border); display: flex; gap: 14px;
}
.reason-line:last-child { border-bottom: none; }
.reason-line::before { content: "—"; color: var(--muted); flex-shrink: 0; }

.fc { padding: 18px 0; border-bottom: 1px solid var(--border); }
.fc:last-child { border-bottom: none; }
.fc-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.fc-pub {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.12em;
}
.fc-rating {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; padding: 2px 8px;
    border: 1px solid var(--border-strong); border-radius: 3px;
}
.fc-claim { color: var(--fg); font-size: 0.95rem; line-height: 1.5; margin: 8px 0; }
.fc-open {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.74rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
}
.fc-open a { color: var(--fg) !important; text-decoration: none; border-bottom: 1px solid var(--border-strong); }

/* ───── Inputs ───── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--fg) !important;
    font-size: 0.95rem !important;
    padding: 14px 16px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.12) !important;
}
.stTextArea label, .stTextInput label, .stRadio label, .stSelectbox label {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.7rem !important; color: var(--muted) !important;
    text-transform: uppercase; letter-spacing: 0.15em;
}

/* Buttons */
.stButton > button {
    border-radius: 100px !important; font-weight: 600 !important;
    padding: 0.65rem 1.4rem !important; font-size: 0.9rem !important;
    border: 1px solid var(--border-strong) !important;
    background: transparent !important; color: var(--fg) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: var(--fg) !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: var(--fg) !important; color: #050505 !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(255,255,255,0.12) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,255,255,0.2) !important;
}

/* Tabs = section switch */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent;
    padding: 4px; border: 1px solid var(--border);
    border-radius: 100px;
    max-width: max-content; margin: 0 auto 40px auto;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 100px !important;
    color: var(--muted) !important; font-weight: 500 !important;
    font-size: 0.86rem !important; padding: 8px 18px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #050505 !important;
    background: var(--fg) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

/* Radio segmented */
.stRadio > div {
    gap: 0 !important; border: 1px solid var(--border);
    border-radius: 100px; padding: 3px; display: inline-flex;
}
.stRadio label {
    background: transparent !important; border: none !important;
    padding: 6px 16px !important; margin: 0 !important;
    border-radius: 100px !important; color: var(--muted) !important;
    font-size: 0.85rem !important; text-transform: none !important;
    letter-spacing: 0 !important; cursor: pointer;
}
.stRadio [data-baseweb="radio"] { display: none !important; }
.stRadio label:has(input:checked) { background: rgba(255,255,255,0.08) !important; color: var(--fg) !important; }

[data-testid="stExpander"] summary, .streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted) !important;
}

.stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }

/* KPI */
.kpi-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin: 20px 0;
}
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
.kpi {
    padding: 24px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
    transition: border-color 0.2s ease;
}
.kpi:hover { border-color: var(--border-strong); }
.kpi-label {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.68rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 14px;
}
.kpi-value {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2.6rem; font-weight: 400; line-height: 1;
}

.section-label {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.18em;
    margin: 40px 0 18px 0;
}

/* Source board */
.board-row {
    display: grid;
    grid-template-columns: 28px 1fr 120px 110px;
    gap: 14px; align-items: center;
    padding: 14px 0; border-bottom: 1px solid var(--border);
}
.board-row:last-child { border-bottom: none; }
.board-rank { font-family: 'Geist Mono', monospace !important; color: var(--muted); font-size: 0.85rem; }
.board-name { color: var(--fg); font-size: 0.92rem; }
.board-count { font-family: 'Geist Mono', monospace !important; color: var(--muted); font-size: 0.82rem; text-align: right; }
.board-bar { height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.board-bar-fill { height: 100%; border-radius: 2px; }

/* Empty state */
.empty {
    padding: 80px 20px; text-align: center;
    color: var(--muted); border: 1px dashed var(--border-strong);
    border-radius: 14px;
}
.empty-mark {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 2rem; color: var(--fg); margin-bottom: 10px;
}
.empty-sub {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.15em;
}

/* CTA band */
.cta-band {
    max-width: 1200px; margin: 0 auto 0 auto;
    padding: 80px 48px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(167,139,250,0.18), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.005));
    text-align: center;
    position: relative; overflow: hidden;
}
.cta-band::before {
    content: ""; position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    mask-image: radial-gradient(ellipse 60% 80% at 50% 50%, #000 30%, transparent 80%);
    pointer-events: none;
}
.cta-h {
    font-family: 'Instrument Serif', serif !important;
    font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.05;
    margin: 0 0 14px 0; position: relative;
}
.cta-h em {
    font-style: italic;
    background: linear-gradient(120deg, var(--accent-3), var(--accent), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.cta-p { color: var(--dim); margin: 0 auto 28px auto; max-width: 480px; position: relative; }

/* Footer */
.footer {
    padding: 60px 48px 40px 48px;
    border-top: 1px solid var(--border);
    margin-top: 120px;
}
.footer-inner {
    max-width: 1200px; margin: 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 20px;
}
.footer-brand { display: flex; align-items: center; gap: 10px; }
.footer-mark {
    width: 18px; height: 18px; background: var(--fg);
    transform: rotate(45deg); border-radius: 2px;
}
.footer-name {
    font-family: 'Instrument Serif', serif !important;
    font-style: italic; font-size: 1.05rem;
}
.footer-meta {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.15em;
}
.footer-stack { display: flex; gap: 18px; color: var(--muted); font-size: 0.82rem; }

.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""")


# ═════════════════════════ NAV ═════════════════════════
_raw_html("""
<div class="nav">
    <div class="nav-brand">
        <div class="nav-mark"></div>
        <span class="nav-name">Veritas</span>
    </div>
    <div class="nav-links">
        <a href="#detector">Detector</a>
        <a href="#features">Features</a>
        <a href="#how">How it works</a>
        <a href="#insights">Insights</a>
    </div>
    <a class="nav-cta" href="#detector">Try it now</a>
</div>
""")


# ═════════════════════════ HERO ═════════════════════════
s_stats = database.stats()
total = s_stats["total"]
fake_count = s_stats["by_label"].get("FAKE", 0)

_raw_html(f"""
<div class="hero">
    <div class="hero-bg"></div>
    <div class="hero-grid"></div>
    <div class="hero-pill">
        <span class="hero-pill-tag">NEW</span>
        <span>Three-layer detection · Updated daily</span>
        <span class="hero-pill-dot"></span>
    </div>
    <h1 class="hero-h1">Spot fake news,<br><em>instantly.</em></h1>
    <p class="hero-lede">A transformer classifier, a source credibility tier list, and live fact-check lookup — combined into one verdict. Paste a URL and know in seconds.</p>
    <div class="hero-ctas">
        <a href="#detector" class="btn-primary">Analyze a story →</a>
        <a href="#how" class="btn-secondary">How it works</a>
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-num">3</div>
            <div class="hero-stat-label">Signals</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">{total if total else '—'}</div>
            <div class="hero-stat-label">Articles analyzed</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">{fake_count if fake_count else '—'}</div>
            <div class="hero-stat-label">Flagged fake</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">24h</div>
            <div class="hero-stat-label">Refresh cadence</div>
        </div>
    </div>
</div>
""")


# ═════════════════════════ MARQUEE ═════════════════════════
recent_rows = database.recent(limit=18)
if recent_rows:
    items_html = ""
    for r in recent_rows * 2:  # duplicate for seamless loop
        lbl = (r.get("label") or "UNCERTAIN").lower()
        title = (r.get("title") or "—")[:80]
        items_html += f'<div class="marquee-item"><span class="tag {lbl}">{lbl}</span><span>{title}</span></div>'
else:
    demo_items = [
        ("real", "BBC · UK inflation falls for third consecutive month"),
        ("fake", "InfoWars · Chemtrails contain mind-control agents"),
        ("real", "Reuters · Apple reports Q1 record revenue"),
        ("uncertain", "Daily Mail · New study links coffee to longevity"),
        ("fake", "Before It's News · 5G towers causing mass die-off"),
        ("real", "NPR · Senate passes infrastructure bill"),
    ]
    items_html = ""
    for lbl, title in demo_items * 2:
        items_html += f'<div class="marquee-item"><span class="tag {lbl}">{lbl}</span><span>{title}</span></div>'

_raw_html(f"""
<div class="marquee-wrap">
    <div class="marquee-track">{items_html}</div>
</div>
""")


# ═════════════════════════ DETECTOR SECTION ═════════════════════════
_raw_html('<div id="detector"></div>')
_raw_html("""
<div class="section">
    <div style="text-align:center;">
        <div class="section-eyebrow">Live detector</div>
        <h2 class="section-h2" style="margin-left:auto;margin-right:auto;">Paste a story. <em>Get a verdict.</em></h2>
    </div>
</div>
""")

# Tabs for the actual functional app
tab_check, tab_feed, tab_stats = st.tabs(["Analyze", "Live feed", "Insights"])

with tab_check:
    _, center, _ = st.columns([1, 5, 1])
    with center:
        _raw_html('<div class="detector-wrap">')
        _raw_html('<div class="detector-title">Detector · Online</div>')

        mode = st.radio("input", ["URL", "Text"], horizontal=True, label_visibility="collapsed")

        text_input = ""
        url_input = None

        if mode == "Text":
            text_input = st.text_area("Article text or claim", height=160,
                                      placeholder="Paste an article or a single claim…")
            url_input = st.text_input("Source URL (optional)", placeholder="https://example.com/article")
        else:
            url_input = st.text_input("Article URL", placeholder="https://bbc.com/news/…")
            if url_input:
                with st.spinner("Fetching…"):
                    meta = news_fetcher.extract_full_article(url_input)
                if meta.get("content"):
                    text_input = f"{meta.get('title') or ''}\n\n{meta['content']}"
                    with st.expander(f"Fetched — {meta.get('title') or 'article'}"):
                        st.caption(meta["content"][:1500] + ("…" if len(meta["content"]) > 1500 else ""))
                else:
                    st.warning(f"Could not extract article text. {meta.get('error', '')}")

        analyze_btn = st.button("Analyze →", type="primary", disabled=not text_input, use_container_width=True)

        if analyze_btn and text_input:
            with st.spinner("Running analysis…"):
                verdict = detector.analyze(text_input, url=url_input or None)

            klass = verdict.label.lower()
            pct = int(round(verdict.fake_prob * 100))
            conf = int(round(verdict.confidence * 100))
            _raw_html(f"""
            <div class="verdict-wrap">
                <div class="verdict-label">Verdict</div>
                <div class="verdict-value {klass}"><span class="dot {klass}"></span>{verdict.label.title()}</div>
                <div class="probbar"><div class="probbar-fill {klass}" style="width:{pct}%;"></div></div>
                <div class="probbar-meta">
                    <span>Fake probability · {pct}%</span>
                    <span>Confidence · {conf}%</span>
                </div>
            </div>
            """)

            # Radar
            _raw_html('<div class="section-label">Signal breakdown</div>')
            src_fake = 1.0 - verdict.source.get("score", 0.5)
            fc_val = verdict.fact_check_prob if verdict.fact_check_prob is not None else 0.5
            vals = [verdict.ml["fake_prob"], src_fake, fc_val, verdict.fake_prob]
            cats = ["ML classifier", "Source tier", "Fact-check", "Combined"]
            color = {"FAKE": C_FAKE, "REAL": C_REAL, "UNCERTAIN": C_UNCERTAIN}[verdict.label]
            h = color.lstrip("#"); rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.18)"

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[v * 100 for v in vals] + [vals[0] * 100],
                theta=cats + [cats[0]],
                fill="toself", fillcolor=rgba,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate="<b>%{theta}</b><br>%{r:.0f}% fake<extra></extra>",
            ))
            radar.update_layout(
                **PLOTLY_BASE, height=320, showlegend=False,
                polar=dict(bgcolor="rgba(0,0,0,0)",
                          radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                                          ticksuffix="%", gridcolor=C_BORDER,
                                          tickfont=dict(size=9, color=C_MUTED), linecolor=C_BORDER, angle=90),
                          angularaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER,
                                          tickfont=dict(size=11, color=C_FG))),
            )

            rc1, rc2 = st.columns([1, 1], gap="large")
            with rc1:
                st.plotly_chart(radar, use_container_width=True, config={"displayModeBar": False})
            with rc2:
                src_domain = verdict.source.get("domain") or "—"
                src_tier = verdict.source.get("tier", "unknown").upper()
                fc_count = len(verdict.fact_checks)
                fc_str = f"{verdict.fact_check_prob*100:.0f}% fake" if verdict.fact_check_prob is not None else "no matches"

                def row(k, v, muted=False):
                    cls = "signal-val muted" if muted else "signal-val"
                    return f'<div class="signal-row"><span class="signal-key">{k}</span><span class="{cls}">{v}</span></div>'

                rows_html = (
                    row("ML classifier", f"{verdict.ml['fake_prob']*100:.0f}%")
                    + row("Source", src_domain, src_domain == "—")
                    + row("Tier", src_tier, src_tier == "UNKNOWN")
                    + row("Fact-checks", fc_str, fc_count == 0)
                    + row("Combined", f"{verdict.fake_prob*100:.0f}%")
                )
                _raw_html(rows_html)

            _raw_html('<div class="section-label">Reasoning</div>')
            _raw_html("".join(f'<div class="reason-line">{r}</div>' for r in verdict.reasons))

            if verdict.fact_checks:
                _raw_html('<div class="section-label">Related fact-checks</div>')
                for fc in verdict.fact_checks:
                    _raw_html(f"""
                    <div class="fc">
                        <div class="fc-top">
                            <span class="fc-pub">{fc.get('publisher') or 'Unknown'}</span>
                            <span class="fc-rating">{fc.get('rating') or 'n/a'}</span>
                        </div>
                        <div class="fc-claim">{fc.get('claim') or ''}</div>
                        <div class="fc-open"><a href="{fc.get('url') or '#'}" target="_blank">Read full fact-check →</a></div>
                    </div>
                    """)

            with st.expander("Raw signal data"):
                st.json(verdict.to_dict())

        _raw_html('</div>')  # detector-wrap


# ───── LIVE FEED TAB ─────
with tab_feed:
    _, center, _ = st.columns([1, 5, 1])
    with center:
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Refresh →"):
                with st.spinner("Fetching & analyzing…"):
                    from scripts.run_daily_update import main as run_update
                    run_update()
                st.rerun()
        with c2:
            label_filter = st.selectbox("filter", ["All", "FAKE", "UNCERTAIN", "REAL"], label_visibility="collapsed")

        rows = database.recent(limit=100, label=None if label_filter == "All" else label_filter)

        if not rows:
            _raw_html("""
            <div class="empty">
                <div class="empty-mark"><em>Nothing yet.</em></div>
                <div class="empty-sub">— Hit refresh to pull today's news</div>
            </div>
            """)
        else:
            today_counts = Counter(r["label"] for r in rows if r.get("label"))
            kpi_html = '<div class="kpi-row">'
            for key, klass in [("REAL", "real"), ("UNCERTAIN", "uncertain"), ("FAKE", "fake"), ("TOTAL", None)]:
                val = today_counts.get(key, 0) if key != "TOTAL" else sum(today_counts.values())
                dot = f'<span class="dot {klass}"></span>' if klass else ""
                label = key.title() if key != "TOTAL" else "Total"
                kpi_html += f'<div class="kpi"><div class="kpi-label">{dot}{label}</div><div class="kpi-value">{val}</div></div>'
            kpi_html += "</div>"
            _raw_html(kpi_html)

            df = pd.DataFrame(rows)
            df = df[["label", "fake_prob", "confidence", "source", "title", "url"]].rename(columns={
                "label": "Verdict", "fake_prob": "Fake",
                "confidence": "Conf.", "source": "Source",
                "title": "Title", "url": "URL",
            })
            if "Fake" in df.columns:
                df["Fake"] = (df["Fake"] * 100).round(0)
                df["Conf."] = (df["Conf."] * 100).round(0)
            st.dataframe(
                df, hide_index=True, use_container_width=True, height=520,
                column_config={
                    "URL": st.column_config.LinkColumn(display_text="open ↗"),
                    "Fake": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
                    "Conf.": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
                },
            )


# ───── INSIGHTS TAB ─────
with tab_stats:
    _, center, _ = st.columns([1, 5, 1])
    with center:
        all_rows = database.recent(limit=10000)
        counts = s_stats["by_label"]

        if s_stats["total"] == 0:
            _raw_html("""
            <div class="empty">
                <div class="empty-mark"><em>No data yet.</em></div>
                <div class="empty-sub">— Run the live feed to populate</div>
            </div>
            """)
        else:
            kpis = [
                ("Total analyzed", s_stats["total"], None),
                ("Flagged fake", counts.get("FAKE", 0), "fake"),
                ("Uncertain", counts.get("UNCERTAIN", 0), "uncertain"),
                ("Verified real", counts.get("REAL", 0), "real"),
            ]
            html = '<div class="kpi-row">'
            for label, val, klass in kpis:
                dot = f'<span class="dot {klass}"></span>' if klass else ""
                html += f'<div class="kpi"><div class="kpi-label">{dot}{label}</div><div class="kpi-value">{val}</div></div>'
            html += "</div>"
            _raw_html(html)

            # Donut
            _raw_html('<div class="section-label">Verdict distribution</div>')
            donut = go.Figure(go.Pie(
                labels=["Real", "Uncertain", "Fake"],
                values=[counts.get("REAL", 0), counts.get("UNCERTAIN", 0), counts.get("FAKE", 0)],
                hole=0.72,
                marker=dict(colors=[C_REAL, C_UNCERTAIN, C_FAKE], line=dict(color="#050505", width=2)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value} articles<br>%{percent}<extra></extra>",
            ))
            donut.update_layout(
                **PLOTLY_BASE, height=300, showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                            font=dict(size=12, color=C_FG), bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    text=f"<b>{s_stats['total']}</b><br><span style='font-size:11px;color:{C_MUTED}'>TOTAL</span>",
                    x=0.5, y=0.5, font=dict(size=26, color=C_FG, family="Instrument Serif"),
                    showarrow=False)],
            )
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

            # Timeline
            df = pd.DataFrame(all_rows)
            if not df.empty:
                df["date"] = pd.to_datetime(df["fetched_at"], errors="coerce").dt.date.astype(str)
                pivot = df.groupby(["date", "label"]).size().unstack(fill_value=0)
                for col in ["REAL", "UNCERTAIN", "FAKE"]:
                    if col not in pivot.columns:
                        pivot[col] = 0

                _raw_html('<div class="section-label">Analyzed over time</div>')
                tl = go.Figure()
                for label, c in [("REAL", C_REAL), ("UNCERTAIN", C_UNCERTAIN), ("FAKE", C_FAKE)]:
                    tl.add_trace(go.Bar(
                        x=pivot.index, y=pivot[label],
                        name=label.title(), marker_color=c, marker_line_width=0,
                        hovertemplate=f"<b>{label.title()}</b><br>%{{y}} · %{{x}}<extra></extra>",
                    ))
                tl.update_layout(
                    **PLOTLY_BASE, barmode="stack", height=280,
                    xaxis=dict(showgrid=False, color=C_MUTED, linecolor=C_BORDER),
                    yaxis=dict(gridcolor=C_BORDER, color=C_MUTED, zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
                                font=dict(size=11, color=C_MUTED), bgcolor="rgba(0,0,0,0)"),
                    bargap=0.35,
                )
                st.plotly_chart(tl, use_container_width=True, config={"displayModeBar": False})

                # Distribution
                probs = (df[df["fake_prob"].notna()]["fake_prob"] * 100).tolist()
                if probs:
                    _raw_html('<div class="section-label">Fake probability distribution</div>')
                    st.caption("Dashed lines at 35% and 65% mark the Real / Uncertain / Fake thresholds.")
                    hist = go.Figure(go.Histogram(
                        x=probs, nbinsx=20,
                        marker=dict(color=C_ACCENT, line=dict(width=0)), opacity=0.85,
                        hovertemplate="%{x}% range<br>%{y} articles<extra></extra>",
                    ))
                    hist.add_vline(x=35, line_dash="dash", line_color=C_REAL, line_width=1, opacity=0.6)
                    hist.add_vline(x=65, line_dash="dash", line_color=C_FAKE, line_width=1, opacity=0.6)
                    hist.update_layout(
                        **PLOTLY_BASE, height=260, bargap=0.08,
                        xaxis=dict(title="Fake probability (%)", gridcolor=C_BORDER, color=C_MUTED, zeroline=False),
                        yaxis=dict(title="Articles", gridcolor=C_BORDER, color=C_MUTED, zeroline=False),
                    )
                    st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})

            # Leaderboard
            sources = [r.get("source") for r in all_rows if r.get("source")]
            top = Counter(sources).most_common(8)
            if top:
                _raw_html('<div class="section-label">Top sources</div>')
                max_c = top[0][1]
                board_html = ""
                for i, (src, c) in enumerate(top, 1):
                    srs = [r for r in all_rows if r.get("source") == src and r.get("fake_prob") is not None]
                    avg = sum(r["fake_prob"] for r in srs) / len(srs) if srs else 0
                    width = int((c / max_c) * 100)
                    bc = C_FAKE if avg >= 0.5 else (C_UNCERTAIN if avg >= 0.35 else C_REAL)
                    board_html += f"""
                    <div class="board-row">
                        <span class="board-rank">{i:02d}</span>
                        <span class="board-name">{src}</span>
                        <div class="board-bar"><div class="board-bar-fill" style="width:{width}%;background:{bc};"></div></div>
                        <span class="board-count">{c} · {avg*100:.0f}%</span>
                    </div>
                    """
                _raw_html(board_html)


# ═════════════════════════ FEATURES ═════════════════════════
_raw_html('<div id="features"></div>')
_raw_html("""
<div class="section">
    <div class="section-eyebrow">What's inside</div>
    <h2 class="section-h2">Built for <em>speed</em> and <em>rigor</em>.</h2>
    <p class="section-lede">Every component was picked for a specific weakness of the others. No single layer can be fooled on its own.</p>

    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-num">01 / ML</div>
            <div class="feat-icon">α</div>
            <div class="feat-title">Transformer classifier</div>
            <div class="feat-desc">RoBERTa fine-tuned on fake-news corpora. Reads style and tone in ~1 second on CPU.</div>
        </div>
        <div class="feat-card">
            <div class="feat-num">02 / TIER</div>
            <div class="feat-icon">β</div>
            <div class="feat-title">Source credibility</div>
            <div class="feat-desc">Curated domain tier list. Reuters, BBC in the top tier. Infowars, Before It's News in the bottom.</div>
        </div>
        <div class="feat-card">
            <div class="feat-num">03 / API</div>
            <div class="feat-icon">γ</div>
            <div class="feat-title">Fact-check lookup</div>
            <div class="feat-desc">Google Fact Check Tools API. Anchors scores to real verdicts from PolitiFact, Snopes, AFP.</div>
        </div>
        <div class="feat-card">
            <div class="feat-num">04 / AUTO</div>
            <div class="feat-icon">δ</div>
            <div class="feat-title">Daily ingestion</div>
            <div class="feat-desc">APScheduler or Windows Task Scheduler pulls NewsAPI top headlines every 24 hours.</div>
        </div>
        <div class="feat-card">
            <div class="feat-num">05 / UI</div>
            <div class="feat-icon">ε</div>
            <div class="feat-title">Explain-first UI</div>
            <div class="feat-desc">Every verdict ships with signal breakdown, reasoning, and links to related fact-checks.</div>
        </div>
        <div class="feat-card">
            <div class="feat-num">06 / DB</div>
            <div class="feat-icon">ζ</div>
            <div class="feat-title">Zero infra</div>
            <div class="feat-desc">SQLite + local model cache. Runs on any laptop. No cloud, no billing, no setup tax.</div>
        </div>
    </div>
</div>
""")


# ═════════════════════════ PIPELINE ═════════════════════════
_raw_html('<div id="how"></div>')
_raw_html("""
<div class="section">
    <div class="section-eyebrow">How it works</div>
    <h2 class="section-h2">Five steps, <em>one verdict.</em></h2>
    <p class="section-lede">The article flows left to right. Each station can independently flag a story. The final combiner weights them.</p>

    <div class="pipe-grid">
        <div class="pipe-step">
            <div class="pipe-num">01</div>
            <div class="pipe-title">Ingest</div>
            <div class="pipe-desc">Pull the article — full text via newspaper3k, or paste manually.</div>
            <span class="pipe-tag">NEWSPAPER3K</span>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">02</div>
            <div class="pipe-title">Classify</div>
            <div class="pipe-desc">Transformer reads text, returns fake probability.</div>
            <span class="pipe-tag">ROBERTA</span>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">03</div>
            <div class="pipe-title">Score source</div>
            <div class="pipe-desc">Domain looked up in the credibility tier list.</div>
            <span class="pipe-tag">TIER LIST</span>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">04</div>
            <div class="pipe-title">Fact-check</div>
            <div class="pipe-desc">Google API returns human verdicts on matching claims.</div>
            <span class="pipe-tag">GOOGLE API</span>
        </div>
        <div class="pipe-step">
            <div class="pipe-num">05</div>
            <div class="pipe-title">Combine</div>
            <div class="pipe-desc">Weighted score · thresholds · final verdict.</div>
            <span class="pipe-tag">DETECTOR.PY</span>
        </div>
    </div>
</div>
""")


# ═════════════════════════ CTA ═════════════════════════
_raw_html(f"""
<div style="padding: 0 48px;">
    <div class="cta-band">
        <h2 class="cta-h">Try it <em>yourself.</em></h2>
        <p class="cta-p">Paste any URL, any claim, any headline. Get a verdict backed by three signals in under two seconds.</p>
        <a class="btn-primary" href="#detector">Open detector →</a>
    </div>
</div>
""")


# ═════════════════════════ FOOTER ═════════════════════════
_raw_html("""
<div class="footer">
    <div class="footer-inner">
        <div class="footer-brand">
            <div class="footer-mark"></div>
            <span class="footer-name">Veritas</span>
            <span class="footer-meta" style="margin-left:16px;">AIML · 2nd year project</span>
        </div>
        <div class="footer-stack">
            <span>Python</span>·<span>Streamlit</span>·<span>Transformers</span>·<span>SQLite</span>·<span>Plotly</span>
        </div>
    </div>
</div>
""")
