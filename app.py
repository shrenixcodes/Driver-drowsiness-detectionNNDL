"""
Streamlit dashboard for the Real-Time Driver Drowsiness Prediction system.

Run with:
    streamlit run app.py

Architecture note: this app deliberately avoids `streamlit-webrtc` (which
pulls in `aiortc`/`av` - native dependencies that are frequently painful to
build on Windows). Instead it uses a plain `cv2.VideoCapture` read loop: each
Streamlit script rerun processes exactly one webcam frame and then calls
`st.rerun()` to loop back to the top, which both refreshes the UI and lets
Streamlit register sidebar interactions (e.g. clicking "Stop"). This is
simple, dependency-free, and entirely sufficient for a single local user
monitoring their own webcam.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from src import config
from src.cnn_model import build_model_a
from src.explainability import GradCAM, overlay_heatmap_on_crop
from src.inference import DrowsinessInferenceEngine, list_available_cameras
from src.preprocessing import preprocess_frame_for_model
from src.utils import checkpoint_exists, load_checkpoint

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Driver Drowsiness Monitoring", page_icon="🚗", layout="wide")

_CUSTOM_CSS = """
<style>
.metric-card {
    background: rgba(127,127,127,0.08);
    border: 1px solid rgba(127,127,127,0.25);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .label { font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 2px; }
.status-badge {
    display: inline-block; padding: 10px 22px; border-radius: 8px;
    font-size: 1.3rem; font-weight: 800; letter-spacing: 0.03em;
}
.status-ALERT { background: #16a34a33; color: #16a34a; border: 1px solid #16a34a; }
.status-DROWSY { background: #f59e0b33; color: #f59e0b; border: 1px solid #f59e0b; }
.status-HIGH_RISK { background: #dc262633; color: #dc2626; border: 1px solid #dc2626; }
.warning-banner {
    background: #dc2626; color: white; padding: 14px; border-radius: 10px;
    text-align: center; font-size: 1.1rem; font-weight: 800; margin-bottom: 10px;
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

MODEL_LABELS = {
    "cnn_lstm": "CNN + LSTM/GRU (Proposed)",
    "cnn": "CNN Baseline (Model A)",
    "heuristic": "Heuristic (EAR/MAR, no trained model)",
}

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
_defaults = {
    "running": False,
    "engine": None,
    "cap": None,
    "history": deque(maxlen=180),
    "last_eye_crop": None,
    "last_active_mode": None,
    "frame_errors": 0,
}
for key, value in _defaults.items():
    st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def _load_cnn_for_explainability():
    if not checkpoint_exists(config.CNN_CHECKPOINT_PATH):
        return None
    ckpt = load_checkpoint(config.CNN_CHECKPOINT_PATH)
    if ckpt is None:
        return None
    model = build_model_a(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _stop_monitoring():
    st.session_state.running = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    if st.session_state.engine is not None:
        st.session_state.engine.close()
        st.session_state.engine = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Driver Monitoring Controls")

st.sidebar.subheader("Model")
model_label = st.sidebar.selectbox("Detection model", list(MODEL_LABELS.values()), index=0)
model_choice = [k for k, v in MODEL_LABELS.items() if v == model_label][0]
for key, path in (("cnn_lstm", config.LSTM_CHECKPOINT_PATH), ("cnn", config.CNN_CHECKPOINT_PATH)):
    if model_choice == key and not checkpoint_exists(path):
        st.sidebar.warning(f"No checkpoint at `{path.name}` yet - will fall back automatically. Train it first (see README).")

st.sidebar.subheader("Thresholds")
drowsy_threshold = st.sidebar.slider("Drowsy threshold", 0.0, 1.0, config.DROWSY_THRESHOLD, 0.01)
high_risk_threshold = st.sidebar.slider("High-risk / warning threshold", 0.0, 1.0, config.HIGH_RISK_THRESHOLD, 0.01)
if high_risk_threshold <= drowsy_threshold:
    st.sidebar.error("High-risk threshold must be greater than the drowsy threshold.")

st.sidebar.subheader("Temporal Settings")
sequence_length = st.sidebar.number_input("Sequence length (frames, Model B)", min_value=5, max_value=60, value=config.SEQUENCE_LENGTH, step=1)
warning_duration = st.sidebar.slider("Warning duration (consecutive high-risk frames)", 1, 60, config.WARNING_DURATION)
smoothing_window = st.sidebar.slider("Smoothing window (frames)", 1, 30, config.SMOOTHING_WINDOW)

st.sidebar.subheader("Camera")
camera_index = st.sidebar.number_input("Camera index", min_value=0, max_value=10, value=config.DEFAULT_CAMERA_INDEX, step=1)
if st.sidebar.button("Scan available cameras"):
    with st.sidebar:
        with st.spinner("Probing camera indices..."):
            found = list_available_cameras(max_index=5)
    st.sidebar.info(f"Detected camera indices: {found}")

audio_enabled = st.sidebar.checkbox("Enable audio warning", value=True)

st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)
start_clicked = col_start.button("Start", type="primary", use_container_width=True, disabled=st.session_state.running)
stop_clicked = col_stop.button("Stop", use_container_width=True, disabled=not st.session_state.running)

if st.session_state.running:
    st.sidebar.caption("Model / sequence length / camera changes require Stop then Start to take effect.")

st.sidebar.markdown("---")
st.sidebar.caption(config.DISCLAIMER)

# --------------------------------------------------------------------------
# Handle Start / Stop
# --------------------------------------------------------------------------
if stop_clicked:
    _stop_monitoring()

if start_clicked:
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.sidebar.error(
            f"Could not open camera index {camera_index}. Make sure a webcam is connected, not in use by another "
            "application, and that this app has camera permission."
        )
    else:
        st.session_state.cap = cap
        st.session_state.engine = DrowsinessInferenceEngine(
            model_type=model_choice,
            sequence_length=int(sequence_length),
            drowsy_threshold=drowsy_threshold,
            high_risk_threshold=high_risk_threshold,
            warning_duration=int(warning_duration),
            smoothing_window=int(smoothing_window),
            enable_audio=audio_enabled,
        )
        st.session_state.history.clear()
        st.session_state.running = True

# Live threshold updates without a restart (cheap - no model reload needed)
if st.session_state.engine is not None:
    st.session_state.engine.logic.drowsy_threshold = drowsy_threshold
    st.session_state.engine.logic.high_risk_threshold = high_risk_threshold
    st.session_state.engine.logic.warning_duration = int(warning_duration)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚗 Driver Monitoring System")
st.caption("Spatiotemporal Deep Learning for Real-Time Driver Drowsiness Prediction and Early Warning")

tab_live, tab_compare, tab_explain = st.tabs(["Live Monitoring", "Model Comparison", "Explainability & About"])

# --------------------------------------------------------------------------
# Tab 1: Live Monitoring
# --------------------------------------------------------------------------
with tab_live:
    if st.session_state.engine is not None and st.session_state.engine.status_message:
        st.info(st.session_state.engine.status_message)

    warning_placeholder = st.empty()
    video_col, metrics_col = st.columns([2, 1])
    frame_placeholder = video_col.empty()
    chart_placeholder = video_col.empty()

    with metrics_col:
        status_placeholder = st.empty()
        m1, m2 = st.columns(2)
        prob_placeholder = m1.empty()
        eye_placeholder = m2.empty()
        m3, m4 = st.columns(2)
        blink_placeholder = m3.empty()
        fps_placeholder = m4.empty()
        m5, m6 = st.columns(2)
        yawn_placeholder = m5.empty()
        mode_placeholder = m6.empty()

    def _metric_card(container, label, value):
        container.markdown(
            f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.running:
        frame_placeholder.info("Click **Start** in the sidebar to begin live monitoring.")
        status_placeholder.markdown('<span class="status-badge status-ALERT">IDLE</span>', unsafe_allow_html=True)
        _metric_card(prob_placeholder, "Drowsiness Probability", "--")
        _metric_card(eye_placeholder, "Eye State", "--")
        _metric_card(blink_placeholder, "Blink Rate", "--")
        _metric_card(fps_placeholder, "FPS", "--")
        _metric_card(yawn_placeholder, "Yawns Detected", "--")
        _metric_card(mode_placeholder, "Active Model", "--")
    else:
        ok, frame = st.session_state.cap.read()
        if not ok:
            st.session_state.frame_errors += 1
            if st.session_state.frame_errors > 10:
                st.error("Lost connection to the webcam (10 consecutive failed reads). Stopping.")
                _stop_monitoring()
            else:
                st.warning("Failed to read a frame from the webcam, retrying...")
                time.sleep(0.2)
                st.rerun()
        else:
            st.session_state.frame_errors = 0
            frame = cv2.flip(frame, 1)
            result = st.session_state.engine.process_frame(frame, overlay_style="boxes_only")
            state = result.state

            if result.detection.success and result.detection.left_eye_crop is not None:
                st.session_state.last_eye_crop = result.detection.left_eye_crop
                st.session_state.last_active_mode = result.active_mode

            rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if state.warning_active:
                warning_placeholder.markdown(
                    '<div class="warning-banner">⚠ DROWSINESS DETECTED &mdash; TAKE A BREAK</div>',
                    unsafe_allow_html=True,
                )
            elif state.status == "DROWSY":
                warning_placeholder.warning("Signs of drowsiness detected - monitoring closely.")
            else:
                warning_placeholder.empty()

            status_placeholder.markdown(
                f'<span class="status-badge status-{state.status}">{state.status.replace("_", " ")}</span>',
                unsafe_allow_html=True,
            )
            _metric_card(prob_placeholder, "Drowsiness Probability", f"{state.smoothed_probability * 100:.0f}%")
            _metric_card(eye_placeholder, "Eye State", state.eye_state)
            _metric_card(blink_placeholder, "Blink Rate", f"{state.blink_rate_per_min:.0f}/min")
            _metric_card(fps_placeholder, "FPS", f"{result.fps:.0f}")
            _metric_card(yawn_placeholder, "Yawns Detected", state.total_yawns)
            _metric_card(mode_placeholder, "Active Model", MODEL_LABELS.get(result.active_mode, result.active_mode))

            st.session_state.history.append(state.smoothed_probability)
            history_df = pd.DataFrame(
                {"frame": range(len(st.session_state.history)), "probability": list(st.session_state.history)}
            )
            base = alt.Chart(history_df).mark_area(opacity=0.35, color="#3b82f6").encode(
                x=alt.X("frame", title="Recent frames"),
                y=alt.Y("probability", title="Drowsiness Probability", scale=alt.Scale(domain=[0, 1])),
            ) + alt.Chart(history_df).mark_line(color="#3b82f6").encode(x="frame", y="probability")

            rule_data = pd.DataFrame({"y": [drowsy_threshold, high_risk_threshold], "label": ["Drowsy", "High-risk"]})
            rules = alt.Chart(rule_data).mark_rule(strokeDash=[4, 4]).encode(
                y="y", color=alt.Color("label", scale=alt.Scale(domain=["Drowsy", "High-risk"], range=["#f59e0b", "#dc2626"])),
            )
            chart_placeholder.altair_chart((base + rules).properties(height=220), use_container_width=True)

            time.sleep(0.01)
            st.rerun()

# --------------------------------------------------------------------------
# Tab 2: Model Comparison
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("Baseline CNN vs. Proposed CNN + LSTM/GRU")
    st.caption(
        "Populated directly from `outputs/metrics/comparison_table.csv`, generated by running "
        "`python src/evaluate.py` after training. No values are fabricated - a model that has not "
        "been trained/evaluated simply does not appear in the table."
    )

    comparison_path = config.METRICS_DIR / "comparison_table.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(df) > 0:
            metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score"] if c in df.columns]
            melted = df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("Model", title=None), y=alt.Y("Score", scale=alt.Scale(domain=[0, 1])),
                color="Model", column="Metric",
            ).properties(width=140, height=280)
            st.altair_chart(chart, use_container_width=False)

            if "Inference Time (ms)" in df.columns:
                st.markdown("**Inference latency (lower is better)**")
                st.altair_chart(
                    alt.Chart(df).mark_bar().encode(x="Model", y="Inference Time (ms)", color="Model").properties(height=260),
                    use_container_width=True,
                )
    else:
        st.info(
            "No evaluation results yet. Train at least one model with `python src/train.py` and then run "
            "`python src/evaluate.py` to populate this comparison."
        )

    st.markdown("---")
    st.subheader("Training curves & confusion matrices")
    figure_cols = st.columns(2)
    figures = {
        "CNN Baseline": ["cnn_baseline_training_curves.png", "cnn_baseline_confusion_matrix.png"],
        "CNN + LSTM/GRU": ["cnn_lstm_proposed_training_curves.png", "cnn_lstm_proposed_confusion_matrix.png"],
    }
    for col, (name, files) in zip(figure_cols, figures.items()):
        with col:
            st.markdown(f"**{name}**")
            any_found = False
            for fname in files:
                fpath = config.FIGURES_DIR / fname
                if fpath.exists():
                    st.image(str(fpath), use_container_width=True)
                    any_found = True
            if not any_found:
                st.caption("Not yet generated - train and evaluate this model to populate.")

# --------------------------------------------------------------------------
# Tab 3: Explainability & About
# --------------------------------------------------------------------------
with tab_explain:
    st.subheader("Grad-CAM: what the CNN is looking at")
    st.caption(
        "Grad-CAM is only defined for the frame-level CNN baseline (Model A) - it highlights, on the most "
        "recent detected eye crop, which pixels most increased the drowsy-class score."
    )

    cnn_for_xai = _load_cnn_for_explainability()
    if cnn_for_xai is None:
        st.info(f"No CNN checkpoint found at `{config.CNN_CHECKPOINT_PATH}`. Train Model A first: `python src/train.py --model cnn`.")
    elif st.session_state.last_eye_crop is None:
        st.info("Start live monitoring on the first tab so a recent eye crop is available, then return here.")
    else:
        if st.button("Generate Grad-CAM for the most recent frame"):
            gradcam = GradCAM(cnn_for_xai)
            tensor = preprocess_frame_for_model(st.session_state.last_eye_crop).unsqueeze(0)
            cam = gradcam.generate(tensor, class_idx=1)
            overlay = overlay_heatmap_on_crop(st.session_state.last_eye_crop, cam)
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(st.session_state.last_eye_crop, cv2.COLOR_BGR2RGB), caption="Eye crop", use_container_width=True)
            c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM (drowsy-class activation)", use_container_width=True)

    st.markdown("---")
    st.subheader("About this project")
    st.markdown(
        f"""
**Pipeline:** Webcam → Face/Eye Detection (MediaPipe Face Mesh, Haar-cascade fallback) → CNN feature
extraction (MobileNetV2) → temporal sequence → GRU/LSTM → drowsiness probability → smoothing & debounce → warning.

**Signals used:** eye closure (EAR), prolonged closure, blink rate, yawning (MAR), and head pose, feeding a
temporal-smoothing state machine so a single bad frame never triggers a false alarm.

**Current sequence length:** `{config.SEQUENCE_LENGTH}` frames (configurable in the sidebar).

{config.DISCLAIMER}
        """
    )
