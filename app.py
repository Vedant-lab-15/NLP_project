"""
app.py
------
MedExtract-Lite — Professional Clinical NER Interface
Light clinical theme · CPU-only · Pure Python
Run: streamlit run app.py

Features:
  🔬 Extractor      — paste / upload a note, extract + highlight entities
  📝 Compare Notes  — side-by-side entity diff between two notes
  📊 Batch View     — browse 5 built-in notes, charts, frequency table
  🔍 Entity Search  — search a term across all batch notes
  ℹ️  About         — architecture, stack, BIO tag viewer
"""

from __future__ import annotations
import os, sys, io, re, collections
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.nlp_engine import MedExtractEngine, Entity

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DICT_PATH = os.path.join(ROOT, "data", "medical_dict.json")

C = {
    "DISEASE": "#DC2626", "SYMPTOM": "#D97706", "DRUG":    "#059669",
    "DIS_BG":  "#FEF2F2", "DIS_BD":  "#FECACA",
    "SYM_BG":  "#FFFBEB", "SYM_BD":  "#FDE68A",
    "DRG_BG":  "#ECFDF5", "DRG_BD":  "#A7F3D0",
    "bg":      "#F8FAFC", "white":   "#FFFFFF",
    "surface": "#FFFFFF", "surface2":"#F1F5F9",
    "border":  "#E2E8F0", "border2": "#CBD5E1",
    "muted":   "#64748B", "muted2":  "#94A3B8",
    "text":    "#0F172A", "text2":   "#334155",
    "accent":  "#2563EB", "accent2": "#EFF6FF", "accent3": "#BFDBFE",
    "header":  "#1E3A5F",
    "green":   "#16A34A", "red":     "#DC2626",
}

ICONS = {"DISEASE": "🦠", "SYMPTOM": "🔍", "DRUG": "💊"}

DEMO_NOTE = """\
ADMITTING DIAGNOSIS:
Patient admitted with acute myocardial infarction.
Presented with chest pain and diaphoresis.

PAST MEDICAL HISTORY:
Known hypertension and type 2 diabetes mellitus.
No history of atrial fibrillation.

HOSPITAL COURSE:
No evidence of pneumonia on chest X-ray.
Congestive heart failure developed as a complication.

MEDICATIONS ON DISCHARGE:
aspirin 75mg, atorvastatin 40mg, metoprolol 25mg, metformin 500mg.\
"""

DEMO_NOTE_B = """\
ADMITTING DIAGNOSIS:
Patient presents with congestive heart failure exacerbation.
Reports shortness of breath and pedal edema.

PAST MEDICAL HISTORY:
Known COPD and hypertension.
No history of stroke.

HOSPITAL COURSE:
No evidence of sepsis.
DVT ruled out on Doppler ultrasound.

MEDICATIONS ON DISCHARGE:
furosemide 40mg, carvedilol 12.5mg, lisinopril 5mg.\
"""

_BATCH_NOTES = [
    ("001", "Patient has hypertension and type 2 diabetes mellitus. "
            "No evidence of pneumonia. Prescribed metformin and lisinopril."),
    ("002", "Admitted with acute myocardial infarction. Chest pain and diaphoresis. "
            "No atrial fibrillation. Started on aspirin and atorvastatin."),
    ("003", "COPD exacerbation with dyspnea and wheezing. "
            "Background of HTN. Denies chest pain. Albuterol and prednisone given."),
    ("004", "UTI diagnosed. Symptoms: dysuria and frequency. "
            "No fever detected. Ciprofloxacin 500mg prescribed."),
    ("005", "DVT on Doppler. Leg swelling and pain. "
            "No pulmonary embolism. Anticoagulation with rivaroxaban started."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedExtract-Lite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {C['text']};
}}
.stApp {{ background-color: {C['bg']}; }}
#MainMenu, header, footer {{ visibility: hidden; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C['white']} !important;
    border-right: 1px solid {C['border']} !important;
    box-shadow: 2px 0 8px rgba(0,0,0,.04);
}}
[data-testid="stSidebar"] section {{ padding-top:0 !important; }}

/* Top bar */
.med-topbar {{
    background: {C['header']};
    padding: 0 28px; height: 58px;
    display: flex; align-items: center; gap: 14px;
    margin: -1rem -1rem 0 -1rem;
}}
.med-topbar-logo {{ font-size:1.22rem; font-weight:700; color:#fff; letter-spacing:-.02em; }}
.med-topbar-logo span {{ color:#93C5FD; }}
.med-topbar-badge {{
    font-size:.65rem; font-weight:600; color:#93C5FD;
    background:rgba(147,197,253,.15); border:1px solid rgba(147,197,253,.3);
    padding:2px 9px; border-radius:20px; letter-spacing:.06em; text-transform:uppercase;
}}
.med-topbar-right {{ margin-left:auto; font-size:.72rem; color:#93C5FD; }}

/* Page header */
.page-title {{ font-size:1.25rem; font-weight:700; color:{C['text']}; letter-spacing:-.02em; margin:0 0 3px 0; }}
.page-subtitle {{ font-size:.81rem; color:{C['muted']}; margin:0; }}

/* KPI cards */
.kpi-card {{
    background:{C['white']}; border:1px solid {C['border']}; border-radius:10px;
    padding:16px 18px; display:flex; flex-direction:column; gap:4px;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
}}
.kpi-label {{ font-size:.68rem; font-weight:600; color:{C['muted']}; text-transform:uppercase; letter-spacing:.09em; }}
.kpi-value {{ font-size:1.9rem; font-weight:700; color:{C['text']}; line-height:1.1; }}
.kpi-sub   {{ font-size:.72rem; color:{C['muted2']}; }}

/* Entity pills */
.pill {{ display:inline-flex; align-items:center; gap:5px; padding:4px 11px; border-radius:20px; font-size:.75rem; font-weight:600; }}
.pill-DISEASE {{ background:{C['DIS_BG']}; color:{C['DISEASE']}; border:1px solid {C['DIS_BD']}; }}
.pill-SYMPTOM {{ background:{C['SYM_BG']}; color:{C['SYMPTOM']}; border:1px solid {C['SYM_BD']}; }}
.pill-DRUG    {{ background:{C['DRG_BG']}; color:{C['DRUG']};    border:1px solid {C['DRG_BD']}; }}

/* Highlights */
.hl-DISEASE {{ background:{C['DIS_BG']}; border-bottom:2px solid {C['DISEASE']}; border-radius:3px; padding:1px 2px; color:{C['text']}; }}
.hl-SYMPTOM {{ background:{C['SYM_BG']}; border-bottom:2px solid {C['SYMPTOM']}; border-radius:3px; padding:1px 2px; color:{C['text']}; }}
.hl-DRUG    {{ background:{C['DRG_BG']}; border-bottom:2px solid {C['DRUG']};    border-radius:3px; padding:1px 2px; color:{C['text']}; }}
.hl-neg     {{ text-decoration:line-through; opacity:.45; }}

/* Note viewer */
.note-viewer {{
    background:{C['white']}; border:1px solid {C['border']}; border-radius:10px;
    padding:18px 20px;
    font-family:'JetBrains Mono','Courier New',monospace;
    font-size:.8rem; line-height:1.9; white-space:pre-wrap; color:{C['text2']};
    max-height:400px; overflow-y:auto;
    box-shadow:inset 0 1px 3px rgba(0,0,0,.04);
}}

/* Sidebar section label */
.sec-label {{
    font-size:.64rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;
    color:{C['muted2']}; padding:8px 0 4px 0; border-top:1px solid {C['border']}; margin-top:8px;
}}

/* Architecture rows */
.arch-row {{
    display:flex; align-items:flex-start; gap:14px;
    background:{C['white']}; border:1px solid {C['border']}; border-radius:10px;
    padding:13px 16px; margin-bottom:8px; box-shadow:0 1px 3px rgba(0,0,0,.04);
}}
.arch-num {{
    min-width:28px; height:28px; background:{C['accent']}; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:.72rem; font-weight:700; color:white; margin-top:1px;
}}
.arch-title {{ font-weight:600; font-size:.85rem; color:{C['text']}; margin-bottom:2px; }}
.arch-desc  {{ font-size:.76rem; color:{C['muted']}; line-height:1.5; }}

/* Stack rows */
.stack-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 14px; background:{C['surface2']}; border-radius:8px;
    margin-bottom:5px; border:1px solid {C['border']};
}}
.stack-name {{ font-size:.82rem; font-weight:600; color:{C['text']}; }}
.stack-role {{ font-size:.74rem; color:{C['muted']}; }}

/* Summary card */
.summary-card {{
    background:{C['accent2']}; border:1px solid {C['accent3']};
    border-radius:10px; padding:16px 18px; margin-bottom:8px;
}}
.summary-card h4 {{
    font-size:.82rem; font-weight:700; color:{C['accent']}; margin:0 0 8px 0;
}}
.summary-line {{ font-size:.82rem; color:{C['text2']}; line-height:1.8; }}

/* BIO tag table */
.bio-table {{ border-collapse:collapse; width:100%; font-size:.78rem; font-family:'JetBrains Mono',monospace; }}
.bio-table th {{
    background:{C['surface2']}; color:{C['muted']}; font-weight:600;
    padding:6px 12px; text-align:left; border-bottom:1px solid {C['border']};
    font-size:.7rem; text-transform:uppercase; letter-spacing:.06em;
}}
.bio-table td {{ padding:5px 12px; border-bottom:1px solid {C['border']}; }}
.bio-b  {{ color:{C['accent']}; font-weight:700; }}
.bio-i  {{ color:#7C3AED; }}
.bio-o  {{ color:{C['muted2']}; }}

/* Diff badges */
.diff-added   {{ background:#DCFCE7; color:#166534; border:1px solid #BBF7D0; padding:2px 8px; border-radius:12px; font-size:.74rem; font-weight:600; margin:2px; display:inline-block; }}
.diff-removed {{ background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; padding:2px 8px; border-radius:12px; font-size:.74rem; font-weight:600; margin:2px; display:inline-block; text-decoration:line-through; }}
.diff-common  {{ background:{C['surface2']}; color:{C['text2']}; border:1px solid {C['border']}; padding:2px 8px; border-radius:12px; font-size:.74rem; margin:2px; display:inline-block; }}

/* Search highlight */
.search-hit {{ background:#FEF9C3; border-bottom:2px solid #CA8A04; border-radius:3px; padding:1px 2px; }}

/* Streamlit overrides */
.stButton > button {{
    background:{C['accent']} !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:.84rem !important;
    padding:10px 20px !important;
    box-shadow:0 1px 4px rgba(37,99,235,.25) !important;
    transition:all .15s ease !important;
}}
.stButton > button:hover {{
    background:#1D4ED8 !important;
    box-shadow:0 3px 8px rgba(37,99,235,.35) !important;
}}
[data-testid="stDownloadButton"] > button {{
    background:{C['white']} !important; color:{C['accent']} !important;
    border:1px solid {C['accent3']} !important; border-radius:8px !important;
    font-weight:600 !important; font-size:.82rem !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background:{C['surface2']}; border-radius:10px; padding:4px; gap:3px; border:1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent; color:{C['muted']} !important;
    border-radius:7px; font-size:.81rem; font-weight:500; padding:6px 16px; border:none !important;
}}
.stTabs [aria-selected="true"] {{
    background:{C['white']} !important; color:{C['accent']} !important;
    box-shadow:0 1px 3px rgba(0,0,0,.08); font-weight:600 !important;
}}
[data-testid="stDataFrame"] {{
    border:1px solid {C['border']}; border-radius:10px; overflow:hidden;
    box-shadow:0 1px 3px rgba(0,0,0,.04);
}}
textarea {{
    background:{C['white']} !important; color:{C['text2']} !important;
    border:1px solid {C['border2']} !important; border-radius:10px !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:.81rem !important; line-height:1.7 !important;
}}
textarea:focus {{
    border-color:{C['accent']} !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.1) !important;
}}
[data-baseweb="select"] {{
    background:{C['white']} !important; border-color:{C['border2']} !important; border-radius:8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    border-radius:8px; padding:8px 10px; transition:background .1s; font-size:.85rem;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
    background:{C['surface2']};
}}
[data-testid="metric-container"] {{
    background:{C['white']}; border:1px solid {C['border']};
    border-radius:10px; padding:14px 16px; box-shadow:0 1px 4px rgba(0,0,0,.04);
}}
hr {{ border-color:{C['border']} !important; margin:14px 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading clinical NLP engine…")
def get_engine() -> MedExtractEngine:
    if os.path.exists(DICT_PATH):
        return MedExtractEngine(dict_path=DICT_PATH)
    return MedExtractEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _lbl_hdr(text: str) -> str:
    return f"<div style='font-size:.82rem;font-weight:600;color:{C['text2']};margin-bottom:6px;'>{text}</div>"

def annotate_html(text: str, entities) -> str:
    if not entities:
        return f'<div class="note-viewer">{_esc(text)}</div>'
    parts, cursor = [], 0
    for ent in sorted(entities, key=lambda e: e.start):
        if ent.start < cursor:
            continue
        parts.append(_esc(text[cursor:ent.start]))
        neg_cls = " hl-neg" if ent.negated else ""
        tip  = f"{ent.label}{' · NEGATED' if ent.negated else ''}"
        col  = C.get(ent.label, "#888")
        parts.append(
            f'<span class="hl-{ent.label}{neg_cls}" title="{tip}">'
            f'{_esc(ent.text)}'
            f'<sup style="font-size:.6em;color:{col};margin-left:1px;">{ICONS.get(ent.label,"")}</sup>'
            f'</span>'
        )
        cursor = ent.end
    parts.append(_esc(text[cursor:]))
    return f'<div class="note-viewer">{"".join(parts)}</div>'

def kpi(label: str, value, sub: str = "", color: str = "") -> str:
    vs = f"color:{color};" if color else f"color:{C['text']};"
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="{vs}">{value}</div>'
        + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
        + '</div>'
    )

def pill(label: str) -> str:
    return f'<span class="pill pill-{label}">{ICONS[label]} {label}</span>'

def to_df(entities) -> pd.DataFrame:
    if not entities:
        return pd.DataFrame(columns=["Text","Label","Negated","Start","End"])
    return pd.DataFrame([{
        "Text": e.text, "Label": e.label,
        "Negated": "✗" if e.negated else "✓",
        "Start": e.start, "End": e.end,
    } for e in entities])


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_clinical_summary(entities: list[Entity]) -> str:
    """Generate a plain-English clinical summary from extracted entities."""
    confirmed = [e for e in entities if not e.negated]
    negated   = [e for e in entities if e.negated]

    diseases = list({e.text.lower() for e in confirmed if e.label == "DISEASE"})
    symptoms = list({e.text.lower() for e in confirmed if e.label == "SYMPTOM"})
    drugs    = list({e.text.lower() for e in confirmed if e.label == "DRUG"})
    neg_list = list({e.text.lower() for e in negated})

    lines = []
    if diseases:
        lines.append(f"🦠 <b>Active conditions:</b> {', '.join(diseases)}.")
    if symptoms:
        lines.append(f"🔍 <b>Presenting symptoms:</b> {', '.join(symptoms)}.")
    if drugs:
        lines.append(f"💊 <b>Current medications:</b> {', '.join(drugs)}.")
    if neg_list:
        lines.append(f"🚫 <b>Ruled out / denied:</b> {', '.join(neg_list)}.")
    if not lines:
        return "<em>No confirmed entities found.</em>"
    return "<br>".join(lines)


def build_bio_tags(text: str, entities: list[Entity]) -> list[tuple[str, str]]:
    """Return BIO token-tag pairs from the entity list."""
    # Build char-level label array
    char_tags = ["O"] * len(text)
    for ent in entities:
        if ent.negated:
            continue
        char_tags[ent.start] = f"B-{ent.label}"
        for i in range(ent.start + 1, ent.end):
            char_tags[i] = f"I-{ent.label}"

    bio: list[tuple[str, str]] = []
    # Tokenise on word boundaries
    for m in re.finditer(r"\b\w+(?:['-]\w+)*\b", text):
        token = m.group(0)
        tag   = char_tags[m.start()]
        bio.append((token, tag))
    return bio


def entity_diff(ents_a: list[Entity], ents_b: list[Entity]):
    """Return (only_in_a, only_in_b, in_both) sets of (text.lower, label)."""
    set_a = {(e.text.lower(), e.label) for e in ents_a if not e.negated}
    set_b = {(e.text.lower(), e.label) for e in ents_b if not e.negated}
    return set_a - set_b, set_b - set_a, set_a & set_b


def search_in_notes(term: str, engine: MedExtractEngine) -> list[dict]:
    """Return notes + entities where term matches entity text."""
    term_lower = term.strip().lower()
    if not term_lower:
        return []
    results = []
    for nid, text in _BATCH_NOTES:
        ents = engine.extract(text)
        hits = [e for e in ents if term_lower in e.text.lower()]
        if hits:
            results.append({"note_id": nid, "text": text, "hits": hits})
    return results


def highlight_search(text: str, term: str) -> str:
    """Wrap every occurrence of term in the note viewer with a yellow highlight."""
    if not term.strip():
        return f'<div class="note-viewer">{_esc(text)}</div>'
    pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
    parts, cursor = [], 0
    for m in pattern.finditer(text):
        parts.append(_esc(text[cursor:m.start()]))
        parts.append(f'<span class="search-hit">{_esc(m.group(0))}</span>')
        cursor = m.end()
    parts.append(_esc(text[cursor:]))
    return f'<div class="note-viewer">{"".join(parts)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────

def topbar():
    st.markdown("""
<div class="med-topbar">
  <span style="font-size:1.4rem;line-height:1;">🏥</span>
  <div class="med-topbar-logo">Med<span>Extract</span>-Lite</div>
  <div class="med-topbar-badge">Clinical NER</div>
  <div class="med-topbar-right">CPU-only &nbsp;·&nbsp; Pure Python &nbsp;·&nbsp; v1.1</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.markdown(f"""
<div style="padding:20px 4px 14px 4px;border-bottom:1px solid {C['border']};margin-bottom:4px;">
  <div style="font-size:1.1rem;font-weight:700;color:{C['text']};letter-spacing:-.02em;">
    🏥 MedExtract-Lite
  </div>
  <div style="font-size:.71rem;color:{C['muted']};margin-top:3px;">
    Rule &amp; Dictionary Clinical NER
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "nav",
            ["🔬  Extractor", "📝  Compare Notes", "📊  Batch View",
             "🔍  Entity Search", "ℹ️  About"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="sec-label" style="margin-top:14px;">Entity Legend</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "".join(f'<div style="margin:5px 0;">{pill(lbl)}</div>' for lbl in ICONS),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sec-label" style="margin-top:14px;">Options</div>',
                    unsafe_allow_html=True)
        show_neg = st.checkbox("Show negated entities", value=True)

        st.markdown('<div class="sec-label" style="margin-top:14px;">System</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-size:.71rem;color:{C['muted']};line-height:1.8;">
  ✅ No PyTorch<br>✅ No GPU / CUDA<br>✅ Python 3.8+<br>✅ 31 tests passing
</div>""", unsafe_allow_html=True)

    return page, show_neg


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: Extractor
# ─────────────────────────────────────────────────────────────────────────────

def page_extractor(engine: MedExtractEngine, show_neg: bool):
    st.markdown(f"""
<div style="margin-bottom:18px;">
  <div class="page-title">🔬 Clinical NER Extractor</div>
  <div class="page-subtitle">
    Paste or upload a note. Entities are colour-coded:
    <span style="color:{C['DISEASE']};font-weight:600;">Disease</span> ·
    <span style="color:{C['SYMPTOM']};font-weight:600;">Symptom</span> ·
    <span style="color:{C['DRUG']};font-weight:600;">Drug</span>.
    Negated entities are struck through.
  </div>
</div>
""", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown(_lbl_hdr("✏️ Input Note"), unsafe_allow_html=True)

        # ── NEW: file uploader ──────────────────────────────────────────────
        uploaded = st.file_uploader(
            "Upload a .txt clinical note (optional)",
            type=["txt"],
            label_visibility="visible",
        )
        upload_text = ""
        if uploaded is not None:
            upload_text = uploaded.read().decode("utf-8", errors="ignore")

        note = st.text_area(
            "note",
            value=upload_text if upload_text else DEMO_NOTE,
            height=330,
            label_visibility="collapsed",
            placeholder="Paste a clinical discharge summary here…",
        )
        run = st.button("⚡  Run Extraction", use_container_width=True)

    entities = engine.extract(note) if note else []
    display  = entities if show_neg else [e for e in entities if not e.negated]

    with col_out:
        st.markdown(_lbl_hdr("📋 Annotated Output"), unsafe_allow_html=True)
        if note:
            ka, kb, kc, kd = st.columns(4)
            ka.markdown(kpi("Total",    len(display),  "entities"), unsafe_allow_html=True)
            kb.markdown(kpi("Diseases", sum(1 for e in display if e.label=="DISEASE"), color=C["DISEASE"]), unsafe_allow_html=True)
            kc.markdown(kpi("Symptoms", sum(1 for e in display if e.label=="SYMPTOM"), color=C["SYMPTOM"]), unsafe_allow_html=True)
            kd.markdown(kpi("Drugs",    sum(1 for e in display if e.label=="DRUG"),    color=C["DRUG"]),    unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(annotate_html(note, display), unsafe_allow_html=True)

    if note and entities:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Tabs: Entity table | Clinical Summary | Negation | BIO tags ────
        t_tbl, t_sum, t_neg, t_bio = st.tabs(
            ["🗂️ Entity Table", "📋 Clinical Summary", "🚫 Negation Report", "🏷️ BIO Tags"]
        )

        df = to_df(display)

        # Entity table tab
        with t_tbl:
            st_all, st_dis, st_sym, st_drg = st.tabs(
                ["All", "🦠 Diseases", "🔍 Symptoms", "💊 Drugs"]
            )
            with st_all: st.dataframe(df, use_container_width=True, height=200)
            with st_dis: st.dataframe(df[df["Label"]=="DISEASE"].reset_index(drop=True), use_container_width=True, height=200)
            with st_sym: st.dataframe(df[df["Label"]=="SYMPTOM"].reset_index(drop=True), use_container_width=True, height=200)
            with st_drg: st.dataframe(df[df["Label"]=="DRUG"].reset_index(drop=True),    use_container_width=True, height=200)
            st.download_button("⬇️ Export CSV", df.to_csv(index=False),
                               file_name="entities.csv", mime="text/csv")

        # ── NEW: Clinical Summary tab ───────────────────────────────────────
        with t_sum:
            summary_html = build_clinical_summary(entities)
            st.markdown(f"""
<div class="summary-card">
  <h4>📋 Auto-Generated Clinical Summary</h4>
  <div class="summary-line">{summary_html}</div>
</div>""", unsafe_allow_html=True)

            # Confirmed vs negated donut-style metrics
            confirmed = [e for e in entities if not e.negated]
            negated   = [e for e in entities if e.negated]
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Entities",    len(entities))
            s2.metric("✅ Confirmed",       len(confirmed))
            s3.metric("🚫 Negated",         len(negated))

            # Confirmed entity list per category
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            for lbl in ["DISEASE", "SYMPTOM", "DRUG"]:
                items = sorted({e.text for e in confirmed if e.label == lbl},
                               key=lambda x: x.lower())
                if items:
                    bg = {"DISEASE": C["DIS_BG"], "SYMPTOM": C["SYM_BG"], "DRUG": C["DRG_BG"]}[lbl]
                    bd = {"DISEASE": C["DIS_BD"], "SYMPTOM": C["SYM_BD"], "DRUG": C["DRG_BD"]}[lbl]
                    items_html = " ".join(
                        f'<span style="background:{bg};border:1px solid {bd};'
                        f'color:{C[lbl]};padding:3px 10px;border-radius:12px;'
                        f'font-size:.78rem;font-weight:600;margin:2px;display:inline-block;">'
                        f'{_esc(it)}</span>'
                        for it in items
                    )
                    st.markdown(
                        f'<div style="margin-bottom:10px;">{pill(lbl)}'
                        f'<div style="margin-top:6px;">{items_html}</div></div>',
                        unsafe_allow_html=True,
                    )

        # ── NEW: Negation Report tab ────────────────────────────────────────
        with t_neg:
            neg_ents  = [e for e in entities if e.negated]
            conf_ents = [e for e in entities if not e.negated]

            n1, n2, n3 = st.columns(3)
            n1.metric("Total Entities",  len(entities))
            n2.metric("✅ Confirmed",     len(conf_ents),
                      delta=f"{100*len(conf_ents)//max(len(entities),1)}% of total")
            n3.metric("🚫 Negated",       len(neg_ents),
                      delta=f"{100*len(neg_ents)//max(len(entities),1)}% of total",
                      delta_color="inverse")

            if neg_ents:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown(_lbl_hdr("Negated Entities"), unsafe_allow_html=True)
                neg_df = pd.DataFrame([{
                    "Entity": e.text, "Label": e.label,
                    "Position": f"{e.start}–{e.end}"
                } for e in neg_ents])
                st.dataframe(neg_df, use_container_width=True, height=180)

                # Per-label negation breakdown
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown(_lbl_hdr("Negation by Entity Type"), unsafe_allow_html=True)
                neg_counts = pd.DataFrame([{"Label": e.label} for e in neg_ents])
                nc = neg_counts["Label"].value_counts().reset_index()
                nc.columns = ["Label", "Negated Count"]
                st.bar_chart(nc.set_index("Label"), height=160, color=["#DC2626"])
            else:
                st.info("✅ No negated entities found in this note.")

        # ── NEW: BIO Tags tab ───────────────────────────────────────────────
        with t_bio:
            st.markdown(f"""
<div style="font-size:.78rem;color:{C['muted']};margin-bottom:10px;">
  BIO (Beginning-Inside-Outside) tagging labels each token.
  <span class="bio-b">B-</span> = entity start &nbsp;
  <span class="bio-i">I-</span> = entity continuation &nbsp;
  <span class="bio-o">O</span> = non-entity token.
  Negated entities are excluded.
</div>""", unsafe_allow_html=True)

            bio_pairs = build_bio_tags(note, entities)
            # Render first 80 tokens in a styled table
            rows_html = ""
            for token, tag in bio_pairs[:80]:
                if tag.startswith("B-"):
                    tc = "bio-b"
                elif tag.startswith("I-"):
                    tc = "bio-i"
                else:
                    tc = "bio-o"
                rows_html += (
                    f"<tr><td>{_esc(token)}</td>"
                    f'<td class="{tc}">{tag}</td></tr>'
                )

            st.markdown(f"""
<div style="max-height:300px;overflow-y:auto;border:1px solid {C['border']};
     border-radius:10px;padding:4px;">
<table class="bio-table">
  <thead><tr><th>Token</th><th>BIO Tag</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>""", unsafe_allow_html=True)

            if len(bio_pairs) > 80:
                st.caption(f"Showing first 80 of {len(bio_pairs)} tokens.")

            # Download BIO as CSV
            bio_df = pd.DataFrame(bio_pairs, columns=["Token", "BIO_Tag"])
            st.download_button("⬇️ Download BIO Tags CSV",
                               bio_df.to_csv(index=False),
                               file_name="bio_tags.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2: Compare Notes  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def page_compare(engine: MedExtractEngine, show_neg: bool):
    st.markdown(f"""
<div style="margin-bottom:18px;">
  <div class="page-title">📝 Compare Notes</div>
  <div class="page-subtitle">
    Extract entities from two notes side-by-side and see what's new, removed, or shared.
  </div>
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(_lbl_hdr("📄 Note A"), unsafe_allow_html=True)
        note_a = st.text_area("note_a", value=DEMO_NOTE,   height=300,
                              label_visibility="collapsed")
    with col_b:
        st.markdown(_lbl_hdr("📄 Note B"), unsafe_allow_html=True)
        note_b = st.text_area("note_b", value=DEMO_NOTE_B, height=300,
                              label_visibility="collapsed")

    compare_btn = st.button("🔄  Compare", use_container_width=False)

    if note_a and note_b:
        ents_a = engine.extract(note_a)
        ents_b = engine.extract(note_b)
        if not show_neg:
            ents_a = [e for e in ents_a if not e.negated]
            ents_b = [e for e in ents_b if not e.negated]

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Side-by-side annotated views
        v1, v2 = st.columns(2, gap="large")
        with v1:
            st.markdown(_lbl_hdr("Note A — Annotated"), unsafe_allow_html=True)
            ka1, ka2, ka3 = st.columns(3)
            ka1.metric("🦠", sum(1 for e in ents_a if e.label=="DISEASE"))
            ka2.metric("🔍", sum(1 for e in ents_a if e.label=="SYMPTOM"))
            ka3.metric("💊", sum(1 for e in ents_a if e.label=="DRUG"))
            st.markdown(annotate_html(note_a, ents_a), unsafe_allow_html=True)
        with v2:
            st.markdown(_lbl_hdr("Note B — Annotated"), unsafe_allow_html=True)
            kb1, kb2, kb3 = st.columns(3)
            kb1.metric("🦠", sum(1 for e in ents_b if e.label=="DISEASE"))
            kb2.metric("🔍", sum(1 for e in ents_b if e.label=="SYMPTOM"))
            kb3.metric("💊", sum(1 for e in ents_b if e.label=="DRUG"))
            st.markdown(annotate_html(note_b, ents_b), unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # Diff section
        only_a, only_b, common = entity_diff(ents_a, ents_b)

        st.markdown(_lbl_hdr("🔄 Entity Diff"), unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)

        with d1:
            st.markdown(f"""
<div style="background:{C['DIS_BG']};border:1px solid {C['DIS_BD']};
     border-radius:10px;padding:14px 16px;">
  <div style="font-size:.72rem;font-weight:700;color:{C['red']};
       text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
    ✖ Only in Note A ({len(only_a)})
  </div>
  {''.join(f'<span class="diff-removed">{_esc(t)} <em style="font-size:.7em;">[{l}]</em></span>' for t,l in sorted(only_a)) or "<em style='font-size:.8rem;color:#94A3B8;'>None</em>"}
</div>""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
<div style="background:{C['DRG_BG']};border:1px solid {C['DRG_BD']};
     border-radius:10px;padding:14px 16px;">
  <div style="font-size:.72rem;font-weight:700;color:{C['green']};
       text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
    ✚ Only in Note B ({len(only_b)})
  </div>
  {''.join(f'<span class="diff-added">{_esc(t)} <em style="font-size:.7em;">[{l}]</em></span>' for t,l in sorted(only_b)) or "<em style='font-size:.8rem;color:#94A3B8;'>None</em>"}
</div>""", unsafe_allow_html=True)

        with d3:
            st.markdown(f"""
<div style="background:{C['surface2']};border:1px solid {C['border']};
     border-radius:10px;padding:14px 16px;">
  <div style="font-size:.72rem;font-weight:700;color:{C['muted']};
       text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
    ↔ Shared ({len(common)})
  </div>
  {''.join(f'<span class="diff-common">{_esc(t)} <em style="font-size:.7em;">[{l}]</em></span>' for t,l in sorted(common)) or "<em style='font-size:.8rem;color:#94A3B8;'>None</em>"}
</div>""", unsafe_allow_html=True)

        # Combined comparison DataFrame
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        rows = []
        all_keys = {(e.text.lower(), e.label) for e in ents_a + ents_b}
        set_a_keys = {(e.text.lower(), e.label) for e in ents_a}
        set_b_keys = {(e.text.lower(), e.label) for e in ents_b}
        for t, l in sorted(all_keys):
            rows.append({
                "Entity": t, "Label": l,
                "In Note A": "✓" if (t,l) in set_a_keys else "—",
                "In Note B": "✓" if (t,l) in set_b_keys else "—",
            })
        cmp_df = pd.DataFrame(rows)
        st.dataframe(cmp_df, use_container_width=True, height=240)
        st.download_button("⬇️ Download Comparison CSV",
                           cmp_df.to_csv(index=False),
                           file_name="note_comparison.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3: Batch View
# ─────────────────────────────────────────────────────────────────────────────

def page_batch(engine: MedExtractEngine, show_neg: bool):
    st.markdown(f"""
<div style="margin-bottom:18px;">
  <div class="page-title">📊 Batch View</div>
  <div class="page-subtitle">Five built-in clinical notes — browse annotations and corpus statistics.</div>
</div>
""", unsafe_allow_html=True)

    rows = []
    for nid, text in _BATCH_NOTES:
        for e in engine.extract(text):
            rows.append({"note_id": nid, "text": e.text,
                         "label": e.label, "negated": e.negated})
    df  = pd.DataFrame(rows, columns=["note_id","text","label","negated"])
    ddf = df if show_neg else df[~df["negated"]]

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(kpi("Notes",    len(_BATCH_NOTES), "processed"),  unsafe_allow_html=True)
    k2.markdown(kpi("Total",    len(ddf),          "entities"),   unsafe_allow_html=True)
    k3.markdown(kpi("Diseases", int((ddf["label"]=="DISEASE").sum()), color=C["DISEASE"]), unsafe_allow_html=True)
    k4.markdown(kpi("Symptoms", int((ddf["label"]=="SYMPTOM").sum()), color=C["SYMPTOM"]), unsafe_allow_html=True)
    k5.markdown(kpi("Drugs",    int((ddf["label"]=="DRUG").sum()),    color=C["DRUG"]),    unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown(_lbl_hdr("🗒️ Note Browser"), unsafe_allow_html=True)
        note_id = st.selectbox("note_select", [n for n,_ in _BATCH_NOTES],
                               format_func=lambda x: f"Clinical Note #{x}",
                               label_visibility="collapsed")
        ntext = next(t for nid,t in _BATCH_NOTES if nid==note_id)
        nents = engine.extract(ntext)
        if not show_neg:
            nents = [e for e in nents if not e.negated]
        nm1,nm2,nm3,nm4 = st.columns(4)
        nm1.metric("Entities", len(nents))
        nm2.metric("🦠", sum(1 for e in nents if e.label=="DISEASE"))
        nm3.metric("🔍", sum(1 for e in nents if e.label=="SYMPTOM"))
        nm4.metric("💊", sum(1 for e in nents if e.label=="DRUG"))
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(annotate_html(ntext, nents), unsafe_allow_html=True)

        # ── NEW: per-note clinical summary ──────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(_lbl_hdr("📋 Note Summary"), unsafe_allow_html=True)
        all_note_ents = engine.extract(ntext)
        st.markdown(f"""
<div class="summary-card">
  <div class="summary-line">{build_clinical_summary(all_note_ents)}</div>
</div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(_lbl_hdr("📈 Entity Distribution"), unsafe_allow_html=True)
        lc = ddf["label"].value_counts().reset_index()
        lc.columns = ["Label","Count"]
        st.bar_chart(lc.set_index("Label"), height=190, color=["#2563EB"])

        st.markdown(_lbl_hdr("📊 Negated vs Confirmed"), unsafe_allow_html=True)
        neg_counts = df.groupby("negated").size().reset_index(name="Count")
        neg_counts["Status"] = neg_counts["negated"].map({True:"Negated", False:"Confirmed"})
        st.bar_chart(neg_counts.set_index("Status")["Count"], height=140,
                     color=["#DC2626"])

        st.markdown(_lbl_hdr("🏷️ Most Frequent Entities"), unsafe_allow_html=True)
        top = (ddf[~ddf["negated"]]["text"].str.lower()
               .value_counts().head(8).reset_index())
        top.columns = ["Entity","Count"]
        st.dataframe(top, use_container_width=True, height=170)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(_lbl_hdr("📋 All Extracted Entities"), unsafe_allow_html=True)
    lbl_filter = st.multiselect("Filter label", ["DISEASE","SYMPTOM","DRUG"],
                                default=["DISEASE","SYMPTOM","DRUG"])
    filtered = ddf[ddf["label"].isin(lbl_filter)].reset_index(drop=True)
    st.dataframe(filtered, use_container_width=True, height=240)
    st.download_button("⬇️ Export CSV", filtered.to_csv(index=False),
                       file_name="batch_results.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4: Entity Search  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def page_search(engine: MedExtractEngine):
    st.markdown(f"""
<div style="margin-bottom:18px;">
  <div class="page-title">🔍 Entity Search</div>
  <div class="page-subtitle">
    Search for any medical term across all five batch notes.
    Matching notes are shown with the term highlighted.
  </div>
</div>
""", unsafe_allow_html=True)

    search_col, _ = st.columns([2, 3])
    with search_col:
        term = st.text_input(
            "Search term",
            placeholder="e.g.  hypertension  /  aspirin  /  chest pain",
            label_visibility="visible",
        )

    if not term.strip():
        st.markdown(f"""
<div style="background:{C['accent2']};border:1px solid {C['accent3']};
     border-radius:10px;padding:18px 20px;margin-top:10px;">
  <div style="font-size:.82rem;color:{C['accent']};font-weight:600;margin-bottom:6px;">
    💡 How to use
  </div>
  <div style="font-size:.78rem;color:{C['text2']};line-height:1.7;">
    Type a disease, symptom, drug, or any keyword above.<br>
    The app searches all 5 built-in clinical notes and shows<br>
    every note that contains a matching entity, with the term highlighted.
  </div>
</div>""", unsafe_allow_html=True)
        return

    results = search_in_notes(term, engine)

    # Stats row
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Notes searched", len(_BATCH_NOTES))
    r2.metric("Notes with match", len(results))
    r3.metric("Total hits", sum(len(r["hits"]) for r in results))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if not results:
        st.warning(f'No notes contain an entity matching **"{term}"**.')
        return

    _BG = {"DISEASE": C["DIS_BG"], "SYMPTOM": C["SYM_BG"], "DRUG": C["DRG_BG"]}
    _BD = {"DISEASE": C["DIS_BD"], "SYMPTOM": C["SYM_BD"], "DRUG": C["DRG_BD"]}

    for res in results:
        nid   = res["note_id"]
        text  = res["text"]
        hits  = res["hits"]

        # Hit pills
        pills_html = " ".join(
            f'<span style="background:{_BG[e.label]};border:1px solid {_BD[e.label]};'
            f'color:{C[e.label]};padding:2px 9px;border-radius:12px;'
            f'font-size:.75rem;font-weight:600;margin:2px;display:inline-block;">'
            f'{ICONS[e.label]} {_esc(e.text)}</span>'
            for e in hits
        )
        st.markdown(f"""
<div style="background:{C['white']};border:1px solid {C['border']};
     border-radius:10px;padding:14px 18px;margin-bottom:12px;
     box-shadow:0 1px 4px rgba(0,0,0,.05);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div style="font-size:.85rem;font-weight:700;color:{C['text']};">
      Note #{nid}
    </div>
    <div style="font-size:.72rem;color:{C['muted']};">{len(hits)} hit(s)</div>
  </div>
  <div style="margin-bottom:8px;">{pills_html}</div>
</div>""", unsafe_allow_html=True)

        with st.expander(f"View Note #{nid} with highlights"):
            st.markdown(highlight_search(text, term), unsafe_allow_html=True)

    # Summary table
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(_lbl_hdr("📋 Search Results Table"), unsafe_allow_html=True)
    search_rows = []
    for res in results:
        for e in res["hits"]:
            search_rows.append({
                "Note": res["note_id"], "Entity": e.text,
                "Label": e.label, "Negated": "✗" if e.negated else "✓",
            })
    st.dataframe(pd.DataFrame(search_rows), use_container_width=True, height=220)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5: About
# ─────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown(f"""
<div style="margin-bottom:18px;">
  <div class="page-title">ℹ️ About MedExtract-Lite</div>
  <div class="page-subtitle">Architecture, technology stack, and usage reference.</div>
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown(_lbl_hdr("🏗️ Extraction Pipeline"), unsafe_allow_html=True)
        for num, title, desc in [
            ("1","Lexicon Matching",
             "Longest-match lookup from medical_dict.json — 90 terms across DISEASE / SYMPTOM / DRUG"),
            ("2","Regex Abbreviations",
             "COPD · HTN · DVT · UTI · CAD · CHF · AKI · TIA · COVID-19 · type 2 DM …"),
            ("3","Span Deduplication",
             "Longest span wins on overlap — zero redundant or overlapping entities"),
            ("4","Negation Detection",
             "Sentence-scoped 60-char look-back: no · denied · without · no evidence of …"),
        ]:
            st.markdown(f"""
<div class="arch-row">
  <div class="arch-num">{num}</div>
  <div><div class="arch-title">{title}</div><div class="arch-desc">{desc}</div></div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(_lbl_hdr("📁 Project Structure"), unsafe_allow_html=True)
        st.code("""\
clinical_ner_simple/
├── data/
│   ├── generate_data.py     # Generates 5 synthetic notes
│   ├── clinical_notes.txt   # Auto-generated synthetic data
│   └── medical_dict.json    # 90-term clinical lexicon
├── src/
│   ├── __init__.py
│   ├── nlp_engine.py        # Core NER engine
│   └── pipeline.py          # Note loader & DataFrame pipeline
├── tests/
│   └── test_ner.py          # 31-test PyTest suite
├── app.py                   # This Streamlit app
├── run.bat                  # Windows one-click launcher
├── guide.md                 # Code navigation guide
└── requirements.txt""", language="text")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(_lbl_hdr("▶️ Quick Commands"), unsafe_allow_html=True)
        st.code("run.bat                          # Windows one-click\n\n"
                "pip install -r requirements.txt\n"
                "python data/generate_data.py\n"
                "pytest tests/test_ner.py -v\n"
                "streamlit run app.py", language="bash")

    with col_r:
        st.markdown(_lbl_hdr("🏷️ Entity Types"), unsafe_allow_html=True)
        for lbl, desc in [
            ("DISEASE","Diagnoses, conditions, disorders and syndromes"),
            ("SYMPTOM","Clinical complaints and presenting signs"),
            ("DRUG","Medications and pharmaceutical agents"),
        ]:
            col_map = {"DISEASE":C["DIS_BG"], "SYMPTOM":C["SYM_BG"], "DRUG":C["DRG_BG"]}
            bd_map  = {"DISEASE":C["DIS_BD"], "SYMPTOM":C["SYM_BD"], "DRUG":C["DRG_BD"]}
            st.markdown(f"""
<div style="background:{col_map[lbl]};border:1px solid {bd_map[lbl]};
     border-radius:10px;padding:13px 15px;margin-bottom:8px;">
  <div style="margin-bottom:5px;">{pill(lbl)}</div>
  <div style="font-size:.76rem;color:{C['text2']};">{desc}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(_lbl_hdr("🛠️ Technology Stack"), unsafe_allow_html=True)
        for name, role in [
            ("Python 3.8+","Core language"),
            ("re / json","NLP & dictionary logic"),
            ("pandas","DataFrames & analytics"),
            ("streamlit","Web interface"),
            ("pytest","31 unit tests"),
        ]:
            st.markdown(f"""
<div class="stack-row">
  <span class="stack-name">{name}</span>
  <span class="stack-role">{role}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:{C['accent2']};border:1px solid {C['accent3']};
     border-radius:10px;padding:14px 16px;">
  <div style="font-size:.78rem;font-weight:700;color:{C['accent']};margin-bottom:6px;">
    📦 New in v1.1
  </div>
  <div style="font-size:.74rem;color:{C['text2']};line-height:1.75;">
    ✅ Upload your own .txt note<br>
    ✅ Clinical Summary auto-card<br>
    ✅ Negation Report panel<br>
    ✅ BIO tag viewer + CSV export<br>
    ✅ Note comparison (diff view)<br>
    ✅ Entity Search across notes
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    engine = get_engine()
    topbar()
    page, show_neg = sidebar()

    if page.startswith("🔬"):
        page_extractor(engine, show_neg)
    elif page.startswith("📝"):
        page_compare(engine, show_neg)
    elif page.startswith("📊"):
        page_batch(engine, show_neg)
    elif page.startswith("🔍"):
        page_search(engine)
    else:
        page_about()


if __name__ == "__main__":
    main()
