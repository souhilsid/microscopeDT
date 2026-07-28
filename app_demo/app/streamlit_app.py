from __future__ import annotations

import pickle
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
PROJECT_ROOT = ROOT.parent
FULL_READY_DATA = ROOT / "data" / "Ready_data.csv"
DEMO_READY_DATA = ROOT / "data" / "Ready_data_demo.csv"
MODEL_DIR = ROOT / "model"
SRC_DIR = ROOT / "src"
READY_IMAGES = ROOT / "data" / "Ready_images.pkl"
DEMO_READY_IMAGES = ROOT / "data" / "Ready_images_demo.pkl"
FALLBACK_READY_IMAGES = PROJECT_ROOT / "Ready_images.pkl"
SHARED_READY_IMAGES = PROJECT_ROOT / "shared_data" / "Ready_images.pkl"
HF_STORAGE_READY_IMAGES = Path("/data/Ready_images.pkl")

sys.path.insert(0, str(SRC_DIR))

from bubble_convlstm_serious import (  # noqa: E402
    SeriousBubbleConvLSTM,
    SeriousConvLSTMConfig,
    prepare_dataframe,
    seconds_from_scaled_log,
)


def deployment_data_paths() -> tuple[Path, Path]:
    if READY_IMAGES.exists():
        return FULL_READY_DATA, READY_IMAGES
    if SHARED_READY_IMAGES.exists():
        return FULL_READY_DATA, SHARED_READY_IMAGES
    if HF_STORAGE_READY_IMAGES.exists():
        return FULL_READY_DATA, HF_STORAGE_READY_IMAGES
    if DEMO_READY_DATA.exists() and DEMO_READY_IMAGES.exists():
        return DEMO_READY_DATA, DEMO_READY_IMAGES
    return FULL_READY_DATA, FALLBACK_READY_IMAGES


READY_DATA, ACTIVE_READY_IMAGES = deployment_data_paths()

st.set_page_config(
    page_title="Microscopy Bubble Forecasting",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: #ffffff !important;
        color: #111827 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.96) !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: #f7f8fa !important;
        color: #111827 !important;
    }
    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    p, li, label, span, h1, h2, h3, h4, h5, h6 {
        color: #111827;
    }
    h1 {
        font-size: 2.35rem !important;
        line-height: 1.14 !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    h2, h3 {
        letter-spacing: 0 !important;
    }
    .hero {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .hero-subtitle {
        color: #4b5563;
        font-size: 1rem;
        line-height: 1.45;
        margin-top: 0.15rem;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1.1rem;
    }
    .summary-item {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 76px;
    }
    .summary-label {
        color: #64748b;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }
    .summary-value {
        color: #111827;
        font-size: 1rem;
        font-weight: 750;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e6e8ec;
        padding: 0.28rem 0.42rem;
        border-radius: 8px;
        min-height: 58px;
    }
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.68rem !important;
        color: #4b5563 !important;
        white-space: normal !important;
        line-height: 1.15 !important;
        margin-bottom: 0.15rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
        line-height: 1.08 !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.68rem !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #d1d5db !important;
    }
    .small-note {
        color: #5d6470;
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .image-caption {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.45rem;
    }
    .prediction-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .prediction-label {
        color: #475569;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .prediction-value {
        color: #111827;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
    }
    .prediction-unit {
        color: #4b5563;
        font-size: 1rem;
        margin-top: 0.35rem;
    }
    .prediction-note {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 1.35;
        margin-top: 0.45rem;
    }
    .predicted-card {
        background: #eff6ff;
        border-color: #bfdbfe;
    }
    .actual-card {
        background: #f8fafc;
        border-color: #e2e8f0;
    }
    @media (max-width: 900px) {
        .summary-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 560px) {
        .summary-grid {
            grid-template-columns: 1fr;
        }
        .prediction-value {
            font-size: 2.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt(value, digits: int = 3, suffix: str = "") -> str:
    try:
        value = float(value)
        if np.isfinite(value):
            return f"{value:.{digits}f}{suffix}"
    except Exception:
        pass
    return "NA"


def conc_label(value) -> str:
    try:
        return f"{float(value):.5f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", low_memory=False)


def load_config() -> SeriousConvLSTMConfig:
    raw = pd.read_json(MODEL_DIR / "config.json", typ="series").to_dict()
    cfg = SeriousConvLSTMConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.hidden_channels = tuple(cfg.hidden_channels)
    cfg.artifact_dir = str(MODEL_DIR)
    cfg.data_csv = str(READY_DATA)
    cfg.images_pkl = str(ACTIVE_READY_IMAGES)
    cfg.show_progress = False
    return cfg


def image_tensor(arr: np.ndarray, image_size: int) -> torch.Tensor:
    x = torch.from_numpy(arr).float().unsqueeze(0)
    if x.shape[-1] != image_size:
        x = F.interpolate(
            x.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return x.clamp(0, 1)


def array_to_image(arr: np.ndarray, size: int = 288) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
    return ImageOps.pad(img, (size, size), method=Image.Resampling.BILINEAR, color=(255, 255, 255))


@st.cache_resource(show_spinner="Loading AI model and microscopy data...")
def load_live_bundle():
    cfg = load_config()
    if not Path(cfg.images_pkl).exists():
        raise FileNotFoundError("The microscopy image data file was not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = prepare_dataframe(cfg)
    df_model = df[df["image_array"].notna()].copy().reset_index(drop=True)
    df_model["run_key_code"] = pd.factorize(df_model["run_key"])[0]
    df_model["concentration_label"] = pd.to_numeric(df_model["concentration"], errors="coerce").map(conc_label)

    with open(MODEL_DIR / "tabular_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    features = preprocessor.transform(df_model)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    try:
        checkpoint = torch.load(MODEL_DIR / "best_serious_convlstm.pt", map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_DIR / "best_serious_convlstm.pt", map_location=device)

    model = SeriousBubbleConvLSTM(feature_dim=features.shape[1], cfg=cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "checkpoint": checkpoint,
        "df_model": df_model,
        "features": features,
    }


@st.cache_data(show_spinner=False)
def load_ready_data(path: str) -> pd.DataFrame:
    df = read_csv_any_encoding(Path(path))
    needed = ["surfactant", "nanoparticle", "concentration", "run_id", "bd_row_idx", "bd_t [s]"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Ready data is missing required columns: {missing}")

    out = df[needed].copy()
    out["surfactant"] = out["surfactant"].astype(str)
    out["nanoparticle"] = out["nanoparticle"].astype(str)
    out["concentration_value"] = pd.to_numeric(out["concentration"], errors="coerce")
    out["concentration_label"] = out["concentration_value"].map(conc_label)
    out["run_id_label"] = out["run_id"].astype(str)
    out["bd_row_idx"] = pd.to_numeric(out["bd_row_idx"], errors="coerce")
    out["bd_t [s]"] = pd.to_numeric(out["bd_t [s]"], errors="coerce")
    return out.dropna(subset=["surfactant", "nanoparticle", "concentration_label"])


def generated_live_gif(predictions: list[dict], frame_count: int = 30, size: int = 320) -> bytes:
    source_images = [array_to_image(pred["predicted"], size=size) for pred in sorted(predictions, key=lambda p: p["bd_t_sec"])]
    if not source_images:
        return b""

    if len(source_images) == 1:
        animation_frames = [source_images[0].copy() for _ in range(frame_count)]
    else:
        animation_frames = []
        for idx in range(frame_count):
            pos = idx * (len(source_images) - 1) / max(frame_count - 1, 1)
            left = int(np.floor(pos))
            right = min(left + 1, len(source_images) - 1)
            alpha = float(pos - left)
            animation_frames.append(Image.blend(source_images[left], source_images[right], alpha))

    out = BytesIO()
    animation_frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=animation_frames[1:],
        duration=140,
        loop=0,
    )
    return out.getvalue()


def build_results_document(config: dict, live_run_key: str | None, live_hl: pd.DataFrame, selected_detail: dict) -> bytes:
    predicted_half_life = float(live_hl["pred_half_life_sec"].mean()) if not live_hl.empty else np.nan
    actual_half_life = float(live_hl["actual_half_life_sec"].mean()) if "actual_half_life_sec" in live_hl and not live_hl.empty else np.nan
    lines = [
        "# Microscopy Bubble Forecasting Result",
        "",
        f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Selection",
        "",
        f"- Surfactant: {config['surfactant']}",
        f"- Nanoparticle: {config['nanoparticle']}",
        f"- Concentration: {config['concentration']}",
        f"- Window start: {config['start_t']:.1f} s",
        f"- Window end: {config['end_t']:.1f} s",
        "",
        "## Prediction",
        "",
        f"- Actual half-life: {fmt(actual_half_life, 1, ' s')}",
        f"- Predicted half-life: {fmt(predicted_half_life, 1, ' s')}",
        f"- Generated frames: {len(live_hl)}",
        "",
        "## Selected Frame",
        "",
    ]
    for key, value in selected_detail.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).encode("utf-8")


def run_prediction(config: dict) -> dict:
    live_bundle = load_live_bundle()
    live_run_key, live_predictions = live_sequence_predictions(
        live_bundle,
        config["surfactant"],
        config["nanoparticle"],
        config["concentration"],
        config["start_t"],
        config["end_t"],
        config["frame_gap"],
    )
    live_predictions = sorted(live_predictions, key=lambda item: item["bd_t_sec"])
    gif_bytes = generated_live_gif(live_predictions, frame_count=config["generated_frame_count"]) if live_predictions else b""
    live_hl = pd.DataFrame(
        [
            {
                "time_sec": pred["bd_t_sec"],
                "actual_half_life_sec": pred["actual_half_life_sec"],
                "pred_half_life_sec": pred["pred_half_life_sec"],
            }
            for pred in live_predictions
        ]
    )
    return {
        "config": config,
        "live_run_key": live_run_key,
        "live_predictions": live_predictions,
        "gif_bytes": gif_bytes,
        "live_hl": live_hl,
    }


def filtered_options(df: pd.DataFrame, column: str) -> list[str]:
    return sorted([str(v) for v in df[column].dropna().unique()])


def selected_ready_subset(
    ready: pd.DataFrame,
    surfactant: str,
    nanoparticle: str,
    concentration: str,
) -> pd.DataFrame:
    subset = ready[
        (ready["surfactant"] == surfactant)
        & (ready["nanoparticle"] == nanoparticle)
        & (ready["concentration_label"] == concentration)
    ].copy()
    return subset.sort_values(["bd_t [s]", "bd_row_idx"])


def run_positions(bundle: dict, run_key: str) -> np.ndarray:
    df_model = bundle["df_model"]
    group = df_model[df_model["run_key"] == run_key].sort_values(["bd_row_idx", "bd_t [s]"])
    return group.index.to_numpy(dtype=np.int64)


def valid_starts(bundle: dict, positions: np.ndarray) -> list[int]:
    cfg = bundle["cfg"]
    needed = (cfg.seq_len - 1) * max(1, int(getattr(cfg, "input_frame_stride", 1))) + cfg.pred_gap + 1
    if len(positions) < needed:
        return []
    rows = bundle["df_model"].loc[positions, "bd_row_idx"].to_numpy()
    starts: list[int] = []
    for start in range(0, len(positions) - needed + 1):
        row_window = rows[start : start + needed]
        if cfg.require_contiguous_bd_row_idx and not np.all(np.diff(row_window) == 1):
            continue
        starts.append(start)
    return starts


def predict_live_window(bundle: dict, positions: np.ndarray, start: int) -> dict:
    cfg = bundle["cfg"]
    input_stride = max(1, int(getattr(cfg, "input_frame_stride", 1)))
    input_offsets = start + np.arange(cfg.seq_len) * input_stride
    target_offset = start + (cfg.seq_len - 1) * input_stride + cfg.pred_gap
    input_positions = positions[input_offsets]
    target_pos = int(positions[target_offset])

    df_model = bundle["df_model"]
    images = torch.stack(
        [image_tensor(df_model.loc[int(pos), "image_array"], cfg.image_size) for pos in input_positions],
        dim=0,
    ).unsqueeze(0)
    features = torch.from_numpy(bundle["features"][input_positions]).float().unsqueeze(0)

    with torch.no_grad():
        pred_image, pred_hl = bundle["model"](images.to(bundle["device"]), features.to(bundle["device"]))

    pred = pred_image.detach().cpu().numpy()[0, 0]
    reference = image_tensor(df_model.loc[target_pos, "image_array"], cfg.image_size).numpy()[0]
    last_input = image_tensor(df_model.loc[int(input_positions[-1]), "image_array"], cfg.image_size).numpy()[0]
    pred_hl_sec = float(
        seconds_from_scaled_log(
            pred_hl,
            bundle["checkpoint"]["half_life_log_mean"],
            bundle["checkpoint"]["half_life_log_std"],
        ).reshape(-1)[0]
    )
    row = df_model.loc[target_pos]
    return {
        "target_pos": target_pos,
        "bd_t_sec": float(row["bd_t [s]"]),
        "bd_row_idx": int(row["bd_row_idx"]),
        "last_input": last_input,
        "reference": reference,
        "predicted": pred,
        "pred_half_life_sec": pred_hl_sec,
        "actual_half_life_sec": float(row["half_life_sec"]) if pd.notna(row.get("half_life_sec")) else np.nan,
    }


def select_live_run(bundle: dict, surfactant: str, nanoparticle: str, concentration: str) -> tuple[str | None, np.ndarray, list[int]]:
    df_model = bundle["df_model"]
    subset = df_model[
        (df_model["surfactant"].astype(str) == surfactant)
        & (df_model["nanoparticle"].astype(str) == nanoparticle)
        & (df_model["concentration_label"].astype(str) == concentration)
    ].copy()
    if subset.empty:
        return None, np.array([], dtype=np.int64), []

    candidates = []
    for run_key in sorted(subset["run_key"].dropna().unique()):
        positions = run_positions(bundle, str(run_key))
        starts = valid_starts(bundle, positions)
        if starts:
            candidates.append((len(starts), str(run_key), positions, starts))
    if not candidates:
        return None, np.array([], dtype=np.int64), []

    _, run_key, positions, starts = max(candidates, key=lambda item: item[0])
    return run_key, positions, starts


def live_sequence_predictions(
    bundle: dict,
    surfactant: str,
    nanoparticle: str,
    concentration: str,
    start_t: float,
    end_t: float,
    frame_gap: int,
) -> tuple[str | None, list[dict]]:
    run_key, positions, starts = select_live_run(bundle, surfactant, nanoparticle, concentration)
    if run_key is None:
        return None, []

    cfg = bundle["cfg"]
    input_stride = max(1, int(getattr(cfg, "input_frame_stride", 1)))
    target_offsets = [s + (cfg.seq_len - 1) * input_stride + cfg.pred_gap for s in starts]
    target_positions = positions[target_offsets]
    df_model = bundle["df_model"]
    target_times = df_model.loc[target_positions, "bd_t [s]"].to_numpy(dtype=float)

    requested_times = np.arange(start_t, end_t + max(frame_gap, 1), max(frame_gap, 1), dtype=float)
    if len(requested_times) == 0:
        requested_times = np.array([(start_t + end_t) / 2], dtype=float)

    chosen_starts: list[int] = []
    for requested in requested_times:
        nearest = int(np.argmin(np.abs(target_times - requested)))
        chosen_starts.append(starts[nearest])
    chosen_starts = sorted(dict.fromkeys(chosen_starts), key=lambda s: target_times[starts.index(s)])

    return run_key, [predict_live_window(bundle, positions, s) for s in chosen_starts]


st.markdown(
    """
    <div class="hero">
        <h1>Microscopy Bubble Forecasting</h1>
        <div class="hero-subtitle">AI-assisted bubble evolution and foam half-life prediction.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not READY_DATA.exists():
    st.error("The app data file was not found.")
    st.stop()

try:
    ready = load_ready_data(str(READY_DATA))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Formulation")
    surf_options = filtered_options(ready, "surfactant")
    surfactant = st.selectbox("Surfactant", surf_options, index=0)
    surf_df = ready[ready["surfactant"] == surfactant]

    nano_options = filtered_options(surf_df, "nanoparticle")
    nanoparticle = st.selectbox("Nanoparticle", nano_options, index=0)
    nano_df = surf_df[surf_df["nanoparticle"] == nanoparticle]

    conc_options = filtered_options(nano_df, "concentration_label")
    concentration = st.selectbox("Concentration", conc_options, index=0)

selected_data = selected_ready_subset(ready, surfactant, nanoparticle, concentration)

if selected_data.empty:
    st.subheader("Selected Sample")
    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-item"><div class="summary-label">Surfactant</div><div class="summary-value">{surfactant}</div></div>
            <div class="summary-item"><div class="summary-label">Nanoparticle</div><div class="summary-value">{nanoparticle}</div></div>
            <div class="summary-item"><div class="summary-label">Concentration</div><div class="summary-value">{concentration}</div></div>
            <div class="summary-item"><div class="summary-label">Status</div><div class="summary-value">No sample</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    time_range = f"{fmt(selected_data['bd_t [s]'].min(), 1)} - {fmt(selected_data['bd_t [s]'].max(), 1)} s"
    frame_range = f"{fmt(selected_data['bd_row_idx'].min(), 0)} - {fmt(selected_data['bd_row_idx'].max(), 0)}"
    st.subheader("Selected Sample")
    st.markdown(
        f"""
        <div class="summary-grid">
            <div class="summary-item"><div class="summary-label">Formulation</div><div class="summary-value">{surfactant}</div></div>
            <div class="summary-item"><div class="summary-label">Nanoparticle</div><div class="summary-value">{nanoparticle}</div></div>
            <div class="summary-item"><div class="summary-label">Concentration</div><div class="summary-value">{concentration}</div></div>
            <div class="summary-item"><div class="summary-label">Time Range</div><div class="summary-value">{time_range}</div></div>
        </div>
        <div class="summary-grid">
            <div class="summary-item"><div class="summary-label">Frames</div><div class="summary-value">{len(selected_data)}</div></div>
            <div class="summary-item"><div class="summary-label">Frame Range</div><div class="summary-value">{frame_range}</div></div>
            <div class="summary-item"><div class="summary-label">AI Model</div><div class="summary-value">Embedded</div></div>
            <div class="summary-item"><div class="summary-label">Status</div><div class="summary-value">Ready</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Time Window")
if selected_data.empty:
    st.write("Select another formulation to generate a prediction.")
else:
    min_t = float(selected_data["bd_t [s]"].min())
    max_t = float(selected_data["bd_t [s]"].max())
    if max_t <= min_t:
        max_t = min_t + 10.0

    control_cols = st.columns([0.48, 0.22, 0.30])
    with control_cols[0]:
        start_t, end_t = st.slider(
            "Prediction time window",
            min_value=float(min_t),
            max_value=float(max_t),
            value=(float(min_t), float(max_t)),
            step=max((max_t - min_t) / 100, 1.0),
            format="%.1f s",
        )
    with control_cols[1]:
        frame_gap = st.selectbox("Frame interval", [5, 10, 20, 30, 60], index=0, format_func=lambda x: f"{x} seconds")
    with control_cols[2]:
        generated_frame_count = st.slider("Animation frames", min_value=30, max_value=120, value=30, step=5)

    current_config = {
        "surfactant": surfactant,
        "nanoparticle": nanoparticle,
        "concentration": concentration,
        "start_t": float(start_t),
        "end_t": float(end_t),
        "frame_gap": int(frame_gap),
        "generated_frame_count": int(generated_frame_count),
    }

    predict_clicked = st.button("Generate prediction", type="primary", use_container_width=True)
    if predict_clicked:
        try:
            with st.spinner("Generating microscopy sequence and half-life prediction..."):
                st.session_state["prediction_result"] = run_prediction(current_config)
        except Exception as exc:
            st.session_state["prediction_result"] = {
                "config": current_config,
                "error": str(exc),
                "live_run_key": None,
                "live_predictions": [],
                "gif_bytes": b"",
                "live_hl": pd.DataFrame(),
            }

    prediction_result = st.session_state.get("prediction_result")
    live_run_key = None
    live_predictions = []
    gif_bytes = b""
    live_hl = pd.DataFrame()
    if not prediction_result or prediction_result.get("config") != current_config:
        st.info("Choose the time window, then click Generate prediction.")
    elif prediction_result.get("error"):
        st.error(f"The prediction could not complete: {prediction_result['error']}")
    else:
        live_run_key = prediction_result["live_run_key"]
        live_predictions = prediction_result["live_predictions"]
        gif_bytes = prediction_result["gif_bytes"]
        live_hl = prediction_result["live_hl"]

    if prediction_result and prediction_result.get("config") == current_config and live_predictions:
        predicted_half_life = float(live_hl["pred_half_life_sec"].mean()) if not live_hl.empty else np.nan
        actual_half_life = float(live_hl["actual_half_life_sec"].mean()) if "actual_half_life_sec" in live_hl and not live_hl.empty else np.nan
        st.subheader("Prediction Result")
        result_cols = st.columns([0.46, 0.54], gap="large")
        with result_cols[0]:
            st.markdown(
                f"""
                <div class="prediction-card actual-card">
                    <div class="prediction-label">Actual half-life</div>
                    <div class="prediction-value">{fmt(actual_half_life, 1)}</div>
                    <div class="prediction-unit">seconds</div>
                </div>
                <div class="prediction-card predicted-card">
                    <div class="prediction-label">Predicted half-life</div>
                    <div class="prediction-value">{fmt(predicted_half_life, 1)}</div>
                    <div class="prediction-unit">seconds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with result_cols[1]:
            st.image(gif_bytes, width=460)
            st.caption(f"Predicted bubble evolution from {fmt(start_t, 1, ' s')} to {fmt(end_t, 1, ' s')}.")

        action_cols = st.columns([0.32, 0.32, 0.36])
        with action_cols[0]:
            st.download_button(
                "Download generated sequence",
                data=gif_bytes,
                file_name=f"{surfactant}_{nanoparticle}_{concentration}_predicted_sequence.gif".replace(" ", "_"),
                mime="image/gif",
                use_container_width=True,
            )

        st.subheader("Frame Preview")
        frame_options = [
            f"{fmt(pred['bd_t_sec'], 1, ' s')}"
            for pred in live_predictions
        ]
        frame_choice = st.selectbox("Preview time", frame_options)
        sample = live_predictions[frame_options.index(frame_choice)]

        img_cols = st.columns(3, gap="medium")
        image_items = [
            (array_to_image(sample["last_input"]), "Starting frame"),
            (array_to_image(sample["predicted"]), "Predicted next frame"),
            (array_to_image(sample["reference"]), "Reference frame"),
        ]
        for col, (img, caption) in zip(img_cols, image_items):
            with col:
                st.image(img, use_container_width=True)
                st.markdown(f"<div class='image-caption'>{caption}</div>", unsafe_allow_html=True)

        selected_detail = {
            "target_time_sec": fmt(sample["bd_t_sec"], 2),
            "actual_half_life_sec": fmt(actual_half_life, 2),
            "predicted_half_life_sec": fmt(predicted_half_life, 2),
        }
        report_bytes = build_results_document(current_config, live_run_key, live_hl, selected_detail)
        with action_cols[1]:
            st.download_button(
                "Download prediction summary",
                data=report_bytes,
                file_name=f"{surfactant}_{nanoparticle}_{concentration}_prediction_summary.md".replace(" ", "_"),
                mime="text/markdown",
                use_container_width=True,
            )
    elif prediction_result and prediction_result.get("config") == current_config and not prediction_result.get("error"):
        st.write("No valid microscopy sequence was found for this formulation and time range.")
