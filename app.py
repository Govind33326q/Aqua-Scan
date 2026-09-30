"""AquaScan — Streamlit sonar debris detector (simple deployment of the AquaVisionaries model).

Self-contained: uses best.pt, aquascan_tiling.py and class_rules.yaml from this folder.
Everything shown comes from the model run on the uploaded frame; nothing is sample data.
"""

import time
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import yaml
from PIL import Image
from ultralytics import YOLO

from aquascan_tiling import draw_detections, run_tiled, run_whole

HERE = Path(__file__).parent
MODEL_PATH = HERE / "best.pt"
RULES_PATH = HERE / "class_rules.yaml"
UNKNOWN = "Unknown — operator review required"

st.set_page_config(page_title="AquaScan | Sonar debris detection", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --bg:#06121c; --panel:#0b1f2c; --panel2:#102a38; --line:rgba(120,200,200,.16); --text:#e3eef0; --t2:#a9bfc6; --t3:#7b949d;
        --cyan:#35d0c1; --amber:#f1b24a; --red:#ff6b5e; --green:#4fd6a6; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] { background: radial-gradient(1100px 520px at 75% -10%, rgba(20,148,136,.16), transparent 60%), var(--bg) !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1320px; padding-top: 2rem; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; letter-spacing: -.01em; }
p, li, label, span, div { color: inherit; }
.av-crumb { font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--cyan); }
.av-h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(2rem,4vw,3rem); font-weight:700; margin:.3rem 0 .4rem; color:var(--text); }
.av-h1 em { font-style:normal; color:var(--cyan); }
.av-lead { color:var(--t2); max-width:70ch; }
.av-kicker { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:var(--cyan); }
.av-panel { border:1px solid var(--line); border-radius:8px; background:linear-gradient(180deg,rgba(16,42,56,.75),rgba(11,31,44,.9)); padding:1rem 1.1rem; margin-bottom:.8rem; }
.av-status { display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:.06em; color:var(--t2); }
.av-status b { color:var(--green); font-weight:500; }
.av-finding { border:1px solid var(--line); border-left:3px solid var(--cyan); border-radius:6px; padding:.7rem .9rem; margin-bottom:.5rem; background:rgba(4,14,22,.5); }
.av-finding .id { font-family:'IBM Plex Mono',monospace; color:var(--cyan); border:1px solid var(--cyan); border-radius:4px; padding:0 .35rem; margin-right:.5rem; }
.av-finding .cls { font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:600; }
.av-finding .conf { float:right; font-family:'Space Grotesk',sans-serif; font-size:1.1rem; }
.av-chip { display:inline-block; font-size:.74rem; padding:.08rem .45rem; border-radius:4px; border:1px solid; margin:.35rem .3rem 0 0; }
.av-small { font-size:.82rem; color:var(--t3); }
.av-tag { font-family:'IBM Plex Mono',monospace; font-size:.64rem; letter-spacing:.08em; border:1px solid rgba(241,178,74,.5); color:var(--amber); padding:.05rem .35rem; border-radius:3px; }
[data-testid="stMetric"] { background:rgba(11,31,44,.9); border:1px solid var(--line); border-left:2px solid var(--cyan); border-radius:8px; padding:.7rem .9rem; }
[data-testid="stMetricLabel"] p { font-family:'IBM Plex Mono',monospace !important; font-size:.68rem !important; letter-spacing:.1em; text-transform:uppercase; color:var(--t3) !important; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; color:var(--text) !important; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] { background:rgba(4,14,22,.7) !important; border:1px dashed rgba(53,208,193,.5) !important; border-radius:8px !important; }
[data-testid="stFileUploader"] section *, [data-testid="stFileUploaderDropzone"] * { color:var(--t2) !important; }
[data-testid="stFileUploader"] button, [data-testid="stBaseButton-secondary"] { background:rgba(16,42,56,.9) !important; color:var(--text) !important; border:1px solid rgba(120,200,200,.35) !important; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background:linear-gradient(180deg,#1aa899,#128a7f) !important; border:1px solid #1fb3a4 !important; color:#fff !important; font-weight:600 !important; }
.stButton > button:disabled { opacity:.45; }
[data-testid="stDownloadButton"] button { background:rgba(16,42,56,.9) !important; color:var(--text) !important; border:1px solid rgba(120,200,200,.35) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div { background:rgba(4,14,22,.8) !important; border-color:rgba(120,200,200,.3) !important; color:var(--text) !important; }
[data-testid="stWidgetLabel"] p { font-family:'IBM Plex Mono',monospace; font-size:.7rem !important; letter-spacing:.08em; text-transform:uppercase; color:var(--t3) !important; }
.stRadio label p, .stCheckbox label p { color:var(--t2) !important; text-transform:none; font-family:'Inter'; font-size:.9rem !important; letter-spacing:0; }
[data-testid="stImageCaption"] { color:var(--t3) !important; font-family:'IBM Plex Mono',monospace; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stAlert"] { background:rgba(79,184,232,.08) !important; border:1px solid rgba(79,184,232,.3) !important; color:#cbe9f7 !important; }
[data-testid="stAlert"] * { color:#cbe9f7 !important; }
.stTabs [data-baseweb="tab"] { color:var(--t2); } .stTabs [aria-selected="true"] { color:var(--cyan) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }
[data-testid="stSidebar"] { background:#051019; }
hr { border-color: var(--line) !important; }
.av-foot { color:var(--t3); font-size:.78rem; border-top:1px solid var(--line); padding-top:.8rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {"harmful": "#ff8a5e", "useful": "#4fd6a6", "harmless": "#8fb3ff", "unknown": "#a4b6bd"}
PRIORITY_COLORS = {"high": "#ff6b5e", "medium": "#f1b24a", "low": "#7f98a6"}


def category_key(cat: str) -> str:
    c = (cat or "").lower()
    return "harmful" if c.startswith("potentially harmful") else "useful" if c.startswith("potentially useful") \
        else "harmless" if c.startswith("potentially harmless") else "unknown"


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return YOLO(model_path)


@st.cache_data(show_spinner=False)
def load_rules(mtime: float) -> dict:
    try:
        return yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def device_name() -> str:
    try:
        import torch
        return "CUDA GPU" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "Unavailable"


rules = load_rules(RULES_PATH.stat().st_mtime if RULES_PATH.exists() else 0.0)
model_ok = MODEL_PATH.exists()

st.markdown(f"""
<div class="av-crumb">AquaScan / Sonar debris detection</div>
<div class="av-h1">Sonar analysis <em>workspace.</em></div>
<p class="av-lead">Upload a forward-looking sonar frame. The YOLOv8n model marks potential marine debris; every result is a machine
suggestion that needs operator review.</p>
<div class="av-status">
  <span>MODEL <b>{'● LOADED' if model_ok else '○ MISSING'}</b></span>
  <span>ARCHITECTURE <b>● YOLOv8n · 11 classes · 640 px</b></span>
  <span>DEVICE <b>● {device_name()}</b></span>
  <span>SOURCE <b style="color:#a9bfc6">○ FILE ANALYSIS (no live sonar in this web app)</b></span>
</div>
""", unsafe_allow_html=True)
st.write("")

left, right = st.columns([1, 2.1], gap="large")
with left:
    st.markdown('<div class="av-kicker">01 / Input</div>', unsafe_allow_html=True)
    st.markdown("#### Sonar frame")
    uploaded = st.file_uploader("Sonar image (JPG / PNG)", type=["jpg", "jpeg", "png"],
                                help="Forward-looking imaging sonar frames like the training data work best.")
    st.markdown('<div class="av-kicker">02 / Configuration</div>', unsafe_allow_html=True)
    detection_mode = st.radio("Detection mode", ["Whole image", "Tiled"], horizontal=True,
                              help="Tiled runs the model on overlapping tiles and merges duplicates. On the 640×640 test set it "
                                   "lowered precision (92.3% → 78.3%) and was ~10× slower, so Whole image is recommended here.")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider("Model input size (px)", options=[320, 480, 640, 800, 960, 1280], value=640)
    tile_size, tile_overlap = 320, 0.25
    if detection_mode == "Tiled":
        c1, c2 = st.columns(2)
        tile_size = c1.select_slider("Tile size (px)", options=[160, 224, 320, 416, 512, 640], value=320)
        tile_overlap = c2.slider("Tile overlap", 0.0, 0.5, 0.25, 0.05)
    run = st.button("Analyse sonar  →", type="primary", use_container_width=True, disabled=uploaded is None or not model_ok)
    if not model_ok:
        st.error("best.pt is missing from this folder.")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    rgb = np.array(image)
    if run:
        with st.spinner("Running YOLOv8n on the frame…"):
            model = load_model(str(MODEL_PATH))
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            t0 = time.perf_counter()
            if detection_mode == "Tiled":
                res = run_tiled(model, bgr, conf=float(confidence), imgsz=int(image_size),
                                tile_size=int(tile_size), overlap=float(tile_overlap))
            else:
                res = run_whole(model, bgr, conf=float(confidence), imgsz=int(image_size))
            elapsed = time.perf_counter() - t0
            annotated = np.ascontiguousarray(draw_detections(bgr, res.detections, res.tiles if detection_mode == "Tiled" else None)[:, :, ::-1])
        st.session_state["analysis"] = {"name": uploaded.name, "original": rgb, "annotated": annotated, "dets": res.detections,
                                        "elapsed": elapsed, "mode": detection_mode, "tiles": len(res.tiles),
                                        "raw": res.raw_detection_count, "conf": confidence}

analysis = st.session_state.get("analysis")
if analysis and (uploaded is None or analysis["name"] != uploaded.name):
    analysis = None

with right:
    st.markdown('<div class="av-kicker">03 / Analysis</div>', unsafe_allow_html=True)
    st.markdown("#### Original vs annotated result")
    if analysis is None:
        if uploaded is not None:
            st.image(np.array(Image.open(uploaded).convert("RGB")), caption="A · Original sonar (not yet analysed)", use_container_width=True)
        else:
            st.markdown("""<div class="av-panel" style="text-align:center;padding:3rem 1rem">
              <div class="av-kicker">Visualisation standby</div>
              <h3 style="margin:.4rem 0">Detection viewport</h3>
              <p class="av-small">Upload a sonar frame on the left and press <b>Analyse sonar</b>. The original and annotated output appear here.</p>
            </div>""", unsafe_allow_html=True)
    else:
        a, b = st.columns(2)
        a.image(analysis["original"], caption="A · Original sonar", use_container_width=True)
        b.image(analysis["annotated"], caption="B · Annotated output (numbered detections)", use_container_width=True)

if analysis:
    dets = analysis["dets"]
    class_rules = rules.get("classes", {})
    prio = rules.get("priority", {})
    hi, med = float(prio.get("high_threshold", 0.7)), float(prio.get("medium_threshold", 0.45))
    st.write("")
    m = st.columns(5)
    m[0].metric("Detections", len(dets))
    m[1].metric("Unique classes", len({d.class_name for d in dets}))
    m[2].metric("Mean confidence", f"{np.mean([d.confidence for d in dets]):.1%}" if dets else "—")
    m[3].metric("Processing", f"{analysis['elapsed']:.2f} s")
    m[4].metric("Mode", "Tiled" if analysis["mode"] == "Tiled" else "Whole", help=f"{analysis['tiles']} tile(s) processed")

    tab_find, tab_table = st.tabs(["Findings", "Detection table"])
    with tab_find:
        if not dets:
            st.info(f"No objects above {analysis['conf']:.0%} confidence. Lower the threshold and analyse again.")
        for i, d in enumerate(dets, start=1):
            rule = class_rules.get(d.class_name, {})
            weight = float(rule.get("class_weight", 1.0))
            score = min(1.0, d.confidence * weight)
            level = "high" if score >= hi else "medium" if score >= med else "low"
            cat = rule.get("category", UNKNOWN)
            cc = CATEGORY_COLORS[category_key(cat)]
            pc = PRIORITY_COLORS[level]
            st.markdown(f"""<div class="av-finding">
              <span class="conf">{d.confidence:.1%}</span>
              <span class="id">#{i:02d}</span><span class="cls">{d.class_name}</span>
              <div>
                <span class="av-chip" style="color:{pc};border-color:{pc}">{level.upper()} priority · score {score:.2f}</span>
                <span class="av-chip" style="color:{cc};border-color:{cc}">{cat}</span>
                <span class="av-chip" style="color:#f1b24a;border-color:#f1b24a">◌ Needs operator review</span>
              </div>
              <p class="av-small" style="margin:.45rem 0 0">Box x {d.x1:.0f}–{d.x2:.0f}, y {d.y1:.0f}–{d.y2:.0f} px ·
                found in {'a tile' if d.origin == 'tile' else 'the whole image'} · priority = confidence × class weight ({weight}).<br>
                <b>Category reason:</b> {rule.get('category_reason', 'No documented rule for this class.')}
                {('<br><b>Source:</b> ' + rule['category_source']) if rule.get('category_source') else ''}</p>
            </div>""", unsafe_allow_html=True)
        st.markdown('<p class="av-small">Categories come from class_rules.yaml and describe the object type, not this specific item. '
                    'For evidence signals (shadow, echo contrast), history, review and reports use the full AquaVisionaries console.</p>',
                    unsafe_allow_html=True)
    with tab_table:
        rows = [{"ID": f"#{i:02d}", "Class": d.class_name, "Confidence": f"{d.confidence:.1%}",
                 "Box (x1, y1, x2, y2)": ", ".join(str(round(v)) for v in (d.x1, d.y1, d.x2, d.y2)),
                 "Category": class_rules.get(d.class_name, {}).get("category", UNKNOWN), "Review": "Needs operator review"}
                for i, d in enumerate(dets, start=1)]
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No detections.")
    buf = BytesIO()
    Image.fromarray(analysis["annotated"]).save(buf, format="PNG")
    st.download_button("Download annotated image", data=buf.getvalue(), file_name="aquascan_detection.png", mime="image/png")

st.markdown('<div class="av-foot">Machine-generated detections require operator review. Model: YOLOv8n trained on the Roboflow '
            'Marine-Debris v2 forward-looking sonar dataset (CC BY 4.0). Uploaded images are processed in memory and not stored.</div>',
            unsafe_allow_html=True)
