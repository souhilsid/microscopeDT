from __future__ import annotations

import gc
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


@dataclass
class SeriousConvLSTMConfig:
    data_csv: str = "Ready_data.csv"
    images_pkl: str = "Ready_images.pkl"
    artifact_dir: str = "artifacts_convlstm_serious"
    seed: int = 42

    quick_start: bool = False
    seq_len: int = 8
    pred_gap: int = 1
    window_stride: int = 1
    input_frame_stride: int = 1
    require_contiguous_bd_row_idx: bool = True

    image_size: int = 128
    batch_size: int = 8
    epochs: int = 60
    patience: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    augment_train: bool = True
    show_progress: bool = True

    image_l1_weight: float = 0.5
    image_mse_weight: float = 0.1
    image_ssim_weight: float = 0.15
    image_gradient_weight: float = 0.05
    half_life_weight: float = 1.0
    half_life_seconds_mae_weight: float = 0.15
    half_life_seconds_mse_weight: float = 0.35
    half_life_log_cosh_weight: float = 0.10
    half_life_log_mean: float = 0.0
    half_life_log_std: float = 1.0
    half_life_seconds_loss_scale: float = 0.0
    censored_half_life_weight: float = 0.20

    hidden_channels: tuple[int, int] = (48, 64)
    feature_hidden: int = 96
    delta_scale: float = 0.35

    train_fraction: float = 0.70
    val_fraction_of_remaining: float = 0.50
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    max_test_windows: int | None = None

    categorical_cols: list[str] = field(default_factory=lambda: ["surfactant", "nanoparticle"])
    forced_numeric_cols: list[str] = field(
        default_factory=lambda: [
            "concentration",
            "bd_row_idx",
            "bd_t [s]",
            "hd_t [s]",
            "hd_h_foam [mm]",
            "hd_h_liquid [mm]",
            "hd_h_total [mm]",
            "hd_V_foam [mL]",
            "hd_V_liquid [mL]",
            "hd_V_total [mL]",
            "hd_FVS [%]",
            "hd_FLS [%]",
        ]
    )
    morphology_threshold: float = 0.5

    def apply_quick_start(self) -> "SeriousConvLSTMConfig":
        self.quick_start = True
        self.image_size = 64
        self.seq_len = min(self.seq_len, 5)
        self.batch_size = 16
        self.epochs = 3
        self.patience = 3
        self.max_train_windows = 3000
        self.max_val_windows = 700
        self.max_test_windows = 700
        self.hidden_channels = (32, 48)
        self.feature_hidden = 64
        return self

    def apply_balanced_serious(self) -> "SeriousConvLSTMConfig":
        """A practical serious preset for an 8 GB laptop GPU."""
        self.quick_start = False
        self.image_size = 96
        self.seq_len = 6
        self.batch_size = 8
        self.epochs = 15
        self.patience = 5
        self.window_stride = 2
        self.max_train_windows = 6000
        self.max_val_windows = 1200
        self.max_test_windows = 1200
        self.hidden_channels = (32, 48)
        self.feature_hidden = 64
        return self

    def apply_full_research(self) -> "SeriousConvLSTMConfig":
        """Full-resolution preset. Expect much longer epochs."""
        self.quick_start = False
        self.image_size = 128
        self.seq_len = 8
        self.batch_size = 8
        self.epochs = 60
        self.patience = 12
        self.window_stride = 1
        self.max_train_windows = None
        self.max_val_windows = None
        self.max_test_windows = None
        self.hidden_channels = (48, 64)
        self.feature_hidden = 96
        return self


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ready_images_pickle(path: str | Path) -> pd.DataFrame:
    """Load the image pickle despite pandas StringArray pickle incompatibilities."""

    from pandas.core.arrays.string_ import StringArray, StringDtype
    from pandas._libs.arrays import NDArrayBacked

    original_dtype_init = StringDtype.__init__

    def compat_stringdtype_init(self, storage=None, na_value=None):
        original_dtype_init(self, storage)

    StringDtype.__init__ = compat_stringdtype_init

    original_stringarray_setstate = StringArray.__setstate__

    def compat_stringarray_setstate(self, state):
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], np.ndarray):
            dtype, values = state
            try:
                NDArrayBacked.__init__(self, values, dtype)
            except Exception:
                NDArrayBacked.__init__(self, values, StringDtype("python"))
            return
        return original_stringarray_setstate(self, state)

    StringArray.__setstate__ = compat_stringarray_setstate

    with open(path, "rb") as f:
        return pickle.load(f)


def load_combined_dataframe(cfg: SeriousConvLSTMConfig) -> pd.DataFrame:
    data_df = pd.read_csv(cfg.data_csv, encoding="cp1252", low_memory=False)
    images_df = load_ready_images_pickle(cfg.images_pkl)

    shared_cols = ["surfactant", "nanoparticle", "concentration", "run_id", "bd_row_idx", "frame_path"]
    if len(data_df) != len(images_df):
        raise ValueError(f"CSV rows {len(data_df)} != pickle rows {len(images_df)}")

    for col in shared_cols:
        left = data_df[col].astype("string").fillna("<NA>").reset_index(drop=True)
        right = images_df[col].astype("string").fillna("<NA>").reset_index(drop=True)
        mismatch_count = int((left != right).sum())
        if mismatch_count:
            raise ValueError(f"Column {col} differs between CSV and pickle in {mismatch_count} rows")

    df = data_df.copy()
    df["image_array"] = images_df["image_array"].values
    df["run_id_str"] = df["run_id"].astype("string")
    df["run_key"] = (
        df["surfactant"].astype(str)
        + "|"
        + df["nanoparticle"].astype(str)
        + "|"
        + df["concentration"].astype(str)
        + "|"
        + df["run_id_str"].astype(str)
    )
    del data_df, images_df
    gc.collect()
    return df


def add_run_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = "hd_t [s]" if "hd_t [s]" in df.columns else "bd_t [s]"
    df["_time_for_run"] = df[time_col].fillna(df.get("bd_t [s]", np.nan))
    df["run_start_time_s"] = df.groupby("run_key")["_time_for_run"].transform("min")
    df["run_end_time_s"] = df.groupby("run_key")["_time_for_run"].transform("max")
    df["run_elapsed_s"] = df["_time_for_run"] - df["run_start_time_s"]
    duration = (df["run_end_time_s"] - df["run_start_time_s"]).replace(0, np.nan)
    df["run_time_fraction"] = (df["run_elapsed_s"] / duration).clip(0, 1).fillna(0)
    return df.drop(columns=["_time_for_run"])


def _first_threshold_crossing_duration(time_values: np.ndarray, metric_values: np.ndarray, threshold: float) -> float:
    time_values = np.asarray(time_values, dtype=np.float64)
    metric_values = np.asarray(metric_values, dtype=np.float64)
    order = np.argsort(time_values)
    t = time_values[order]
    y = metric_values[order]

    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]
    if len(t) < 2 or not np.isfinite(threshold):
        return np.nan

    t0 = t[0]
    if y[0] <= threshold:
        return 0.0

    below = np.where(y <= threshold)[0]
    if len(below) == 0:
        return np.nan

    i = int(below[0])
    if i == 0:
        return 0.0

    t_prev, t_curr = t[i - 1], t[i]
    y_prev, y_curr = y[i - 1], y[i]
    if y_curr == y_prev:
        crossing_time = t_curr
    else:
        ratio = (threshold - y_prev) / (y_curr - y_prev)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        crossing_time = t_prev + ratio * (t_curr - t_prev)
    return float(max(0.0, crossing_time - t0))


def _initial_value(series: pd.Series, n: int = 3) -> float:
    values = series.dropna().to_numpy(dtype=np.float64)
    if len(values) == 0:
        return np.nan
    return float(np.nanmedian(values[: min(n, len(values))]))


def _observed_duration_seconds(group: pd.DataFrame, time_col: str) -> float:
    t = group[time_col].dropna().to_numpy(dtype=np.float64)
    if len(t) < 2:
        return np.nan
    return float(np.nanmax(t) - np.nanmin(t))


def compute_half_life_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute observed or right-censored foam half-life target per run."""

    run_cols = ["surfactant", "nanoparticle", "concentration", "run_id_str"]
    time_col = "hd_t [s]" if "hd_t [s]" in df.columns else "bd_t [s]"
    sources = [
        ("hd_V_foam [mL]", "foam_volume", "relative_half_initial"),
        ("hd_h_foam [mm]", "foam_height", "relative_half_initial"),
        ("hd_FVS [%]", "foam_volume_stability", "absolute_50_percent"),
    ]

    def compute_one(group: pd.DataFrame) -> pd.Series:
        best_censor_duration = _observed_duration_seconds(group, time_col)
        best_censor_source = "not_observed"
        for metric_col, source_name, mode in sources:
            if metric_col not in group.columns:
                continue
            sub = group[[time_col, metric_col]].dropna().sort_values(time_col)
            if len(sub) < 2:
                continue

            source_duration = _observed_duration_seconds(sub, time_col)
            if np.isfinite(source_duration) and (
                not np.isfinite(best_censor_duration) or source_duration > best_censor_duration
            ):
                best_censor_duration = source_duration
                best_censor_source = source_name

            if mode == "absolute_50_percent":
                threshold = 50.0
            else:
                initial = _initial_value(sub[metric_col])
                if not np.isfinite(initial) or initial <= 0:
                    continue
                threshold = 0.5 * initial

            duration = _first_threshold_crossing_duration(
                sub[time_col].to_numpy(), sub[metric_col].to_numpy(), threshold
            )
            if np.isfinite(duration):
                return pd.Series(
                    {
                        "half_life_sec": duration,
                        "half_life_observed": 1.0,
                        "half_life_censor_sec": duration,
                        "half_life_source": source_name,
                        "half_life_threshold": threshold,
                    }
                )

        return pd.Series(
            {
                "half_life_sec": np.nan,
                "half_life_observed": 0.0,
                "half_life_censor_sec": best_censor_duration,
                "half_life_source": best_censor_source,
                "half_life_threshold": np.nan,
            }
        )

    return df.groupby(run_cols, dropna=False).apply(compute_one).reset_index()


def attach_half_life_targets(df: pd.DataFrame) -> pd.DataFrame:
    half_life_df = compute_half_life_table(df)
    run_cols = ["surfactant", "nanoparticle", "concentration", "run_id_str"]
    return df.merge(half_life_df, on=run_cols, how="left")


def prepare_dataframe(cfg: SeriousConvLSTMConfig) -> pd.DataFrame:
    df = load_combined_dataframe(cfg)
    df = add_run_time_features(df)
    df = attach_half_life_targets(df)
    return df


def dataframe_report(df: pd.DataFrame) -> dict[str, Any]:
    valid = df["image_array"].dropna()
    return {
        "rows": int(len(df)),
        "runs": int(df["run_key"].nunique()),
        "missing_images": int(df["image_array"].isna().sum()),
        "image_shapes": dict(Counter(tuple(x.shape) for x in valid)),
        "image_dtypes": dict(Counter(str(x.dtype) for x in valid)),
        "half_life_sources": df.drop_duplicates("run_key")["half_life_source"].value_counts(dropna=False).to_dict(),
        "observed_half_life_runs": int(
            df.drop_duplicates("run_key")["half_life_observed"].fillna(0).astype(float).sum()
        ),
    }


def split_run_codes(df_model: pd.DataFrame, cfg: SeriousConvLSTMConfig) -> tuple[set[int], set[int], set[int]]:
    groups = df_model["run_key_code"].to_numpy()
    idx = np.arange(len(df_model))
    test_size = 1.0 - cfg.train_fraction

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed)
    train_idx, temp_idx = next(splitter.split(idx, groups=groups))

    temp_groups = groups[temp_idx]
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=1.0 - cfg.val_fraction_of_remaining, random_state=cfg.seed + 1
    )
    val_rel, test_rel = next(splitter2.split(temp_idx, groups=temp_groups))
    return set(groups[train_idx]), set(groups[temp_idx[val_rel]]), set(groups[temp_idx[test_rel]])


def build_windows(
    df_model: pd.DataFrame, allowed_run_codes: set[int], cfg: SeriousConvLSTMConfig
) -> list[tuple[np.ndarray, int]]:
    windows: list[tuple[np.ndarray, int]] = []
    # needed = cfg.seq_len + cfg.pred_gap
    input_stride = max(1, int(getattr(cfg, "input_frame_stride", 1)))
    needed = (cfg.seq_len - 1) * input_stride + cfg.pred_gap + 1
    for _, group in df_model[df_model["run_key_code"].isin(allowed_run_codes)].groupby("run_key_code"):
        group = group.sort_values(["bd_row_idx", "bd_t [s]"])
        positions = group.index.to_numpy()
        bd_rows = group["bd_row_idx"].to_numpy()
        if len(group) < needed:
            continue

        for start in range(0, len(group) - needed + 1, cfg.window_stride):
            # input_pos = positions[start : start + cfg.seq_len]
            # target_pos = int(positions[start + cfg.seq_len + cfg.pred_gap - 1])
            input_offsets = start + np.arange(cfg.seq_len) * input_stride
            target_offset = start + (cfg.seq_len - 1) * input_stride + cfg.pred_gap
            input_pos = positions[input_offsets]
            target_pos = int(positions[target_offset])

            if cfg.require_contiguous_bd_row_idx:
                row_window = bd_rows[start : start + needed]
                if not np.all(np.diff(row_window) == 1):
                    continue
            windows.append((input_pos.astype(np.int64), target_pos))
    return windows


def _limit_windows(windows: list[tuple[np.ndarray, int]], max_windows: int | None, seed: int) -> list[tuple[np.ndarray, int]]:
    if max_windows is None or len(windows) <= max_windows:
        return windows
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(windows), size=max_windows, replace=False)
    return [windows[int(i)] for i in chosen]


def make_splits_and_windows(
    df: pd.DataFrame, cfg: SeriousConvLSTMConfig
) -> tuple[pd.DataFrame, dict[str, list[tuple[np.ndarray, int]]], dict[str, set[int]]]:
    df_model = df[df["image_array"].notna()].copy().reset_index(drop=True)
    df_model["run_key_code"] = pd.factorize(df_model["run_key"])[0]

    train_runs, val_runs, test_runs = split_run_codes(df_model, cfg)
    windows = {
        "train": build_windows(df_model, train_runs, cfg),
        "val": build_windows(df_model, val_runs, cfg),
        "test": build_windows(df_model, test_runs, cfg),
    }
    windows["train"] = _limit_windows(windows["train"], cfg.max_train_windows, cfg.seed)
    windows["val"] = _limit_windows(windows["val"], cfg.max_val_windows, cfg.seed + 10)
    windows["test"] = _limit_windows(windows["test"], cfg.max_test_windows, cfg.seed + 20)
    runs = {"train": train_runs, "val": val_runs, "test": test_runs}
    return df_model, windows, runs


def unique_positions_from_windows(windows: list[tuple[np.ndarray, int]]) -> np.ndarray:
    if not windows:
        return np.array([], dtype=np.int64)
    pieces = []
    for input_pos, target_pos in windows:
        pieces.append(input_pos)
        pieces.append(np.array([target_pos], dtype=np.int64))
    return np.unique(np.concatenate(pieces))


class TabularPreprocessor:
    def __init__(self, cfg: SeriousConvLSTMConfig):
        self.cfg = cfg
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.numeric_medians: pd.Series | None = None
        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.feature_dim: int | None = None

    def fit(self, df_model: pd.DataFrame, train_windows: list[tuple[np.ndarray, int]]) -> "TabularPreprocessor":
        train_positions = unique_positions_from_windows(train_windows)
        exclude = {
            "Unnamed: 0",
            "run_key_code",
            "half_life_sec",
            "half_life_observed",
            "half_life_censor_sec",
            "half_life_threshold",
        }
        numeric = [
            col for col in df_model.columns if col not in exclude and pd.api.types.is_numeric_dtype(df_model[col])
        ]
        forced = [col for col in self.cfg.forced_numeric_cols if col in numeric]
        extras = [col for col in ["run_elapsed_s", "run_time_fraction"] if col in numeric and col not in forced]
        rest = [col for col in numeric if col not in forced and col not in extras]
        self.numeric_cols = forced + extras + rest
        self.categorical_cols = [col for col in self.cfg.categorical_cols if col in df_model.columns]

        train_numeric = df_model.loc[train_positions, self.numeric_cols].replace([np.inf, -np.inf], np.nan)
        observed_counts = train_numeric.notna().sum()
        self.numeric_cols = [col for col in self.numeric_cols if int(observed_counts.get(col, 0)) > 0]
        if not self.numeric_cols:
            raise ValueError("No usable numeric columns found in the training windows.")

        train_numeric = train_numeric[self.numeric_cols]
        self.numeric_medians = train_numeric.median(numeric_only=True).fillna(0.0)
        self.scaler = StandardScaler()
        self.scaler.fit(train_numeric.fillna(self.numeric_medians))

        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder.fit(df_model.loc[train_positions, self.categorical_cols].astype(str))

        features = self.transform(df_model.iloc[:1])
        self.feature_dim = int(features.shape[1])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.numeric_medians is None or self.scaler is None or self.encoder is None:
            raise RuntimeError("Preprocessor is not fit.")
        numeric_raw = frame[self.numeric_cols].replace([np.inf, -np.inf], np.nan)
        missing_flags = numeric_raw.isna().astype(np.float32).to_numpy()
        numeric = self.scaler.transform(numeric_raw.fillna(self.numeric_medians))
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
        categorical = self.encoder.transform(frame[self.categorical_cols].astype(str))
        return np.hstack([numeric, missing_flags, categorical]).astype(np.float32)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)


def compute_half_life_scaler(df_model: pd.DataFrame, train_windows: list[tuple[np.ndarray, int]]) -> tuple[float, float]:
    target_positions = np.array([target for _, target in train_windows], dtype=np.int64)
    observed = df_model.loc[target_positions, "half_life_observed"].to_numpy(dtype=np.float32) > 0.5
    values = df_model.loc[target_positions, "half_life_sec"].to_numpy(dtype=np.float32)
    values = values[observed & np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    log_values = np.log1p(values)
    mean = float(np.mean(log_values))
    std = float(np.std(log_values))
    return mean, std if std > 1e-6 else 1.0


class BubbleSequenceDataset(Dataset):
    def __init__(
        self,
        frame_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        windows: list[tuple[np.ndarray, int]],
        cfg: SeriousConvLSTMConfig,
        half_life_log_mean: float,
        half_life_log_std: float,
        training: bool = False,
    ):
        self.frame_df = frame_df
        self.features = feature_matrix
        self.windows = windows
        self.cfg = cfg
        self.training = training
        self.half_life_log_mean = half_life_log_mean
        self.half_life_log_std = half_life_log_std
        self.images = frame_df["image_array"].to_numpy()
        self.half_life_sec = frame_df["half_life_sec"].to_numpy(dtype=np.float32)
        self.half_life_observed = frame_df["half_life_observed"].fillna(0).to_numpy(dtype=np.float32)
        self.half_life_censor_sec = frame_df["half_life_censor_sec"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def _image_tensor(self, pos: int) -> torch.Tensor:
        arr = self.images[pos]
        x = torch.from_numpy(arr).float().unsqueeze(0)
        if self.cfg.image_size is not None and x.shape[-1] != self.cfg.image_size:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return x.clamp(0, 1)

    @staticmethod
    def _augment(images: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            images = torch.flip(images, dims=[-1])
            target = torch.flip(target, dims=[-1])
        if torch.rand(()) < 0.5:
            images = torch.flip(images, dims=[-2])
            target = torch.flip(target, dims=[-2])
        k = int(torch.randint(0, 4, ()).item())
        if k:
            images = torch.rot90(images, k=k, dims=[-2, -1])
            target = torch.rot90(target, k=k, dims=[-2, -1])
        return images, target

    def _scale_log_seconds(self, seconds: float) -> float:
        return (math.log1p(max(0.0, float(seconds))) - self.half_life_log_mean) / self.half_life_log_std

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_pos, target_pos = self.windows[idx]
        image_seq = torch.stack([self._image_tensor(int(pos)) for pos in input_pos], dim=0)
        target_image = self._image_tensor(target_pos)
        if self.training and self.cfg.augment_train:
            image_seq, target_image = self._augment(image_seq, target_image)

        feature_seq = torch.from_numpy(self.features[input_pos]).float()

        observed = float(self.half_life_observed[target_pos] > 0.5)
        half_life = self.half_life_sec[target_pos]
        censor = self.half_life_censor_sec[target_pos]

        if observed and np.isfinite(half_life):
            half_life_scaled = self._scale_log_seconds(float(half_life))
        else:
            half_life_scaled = 0.0

        if np.isfinite(censor):
            censor_scaled = self._scale_log_seconds(float(censor))
            censor_mask = 1.0 - observed
        else:
            censor_scaled = 0.0
            censor_mask = 0.0

        return {
            "images": image_seq,
            "features": feature_seq,
            "target_image": target_image,
            "half_life": torch.tensor([half_life_scaled], dtype=torch.float32),
            "half_life_observed_mask": torch.tensor([observed], dtype=torch.float32),
            "half_life_censor": torch.tensor([censor_scaled], dtype=torch.float32),
            "half_life_censor_mask": torch.tensor([censor_mask], dtype=torch.float32),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
        }


def make_dataloaders(
    df_model: pd.DataFrame,
    windows: dict[str, list[tuple[np.ndarray, int]]],
    cfg: SeriousConvLSTMConfig,
) -> tuple[dict[str, DataLoader], TabularPreprocessor, np.ndarray, tuple[float, float]]:
    preprocessor = TabularPreprocessor(cfg).fit(df_model, windows["train"])
    features = preprocessor.transform(df_model)
    hl_mean, hl_std = compute_half_life_scaler(df_model, windows["train"])
    cfg.half_life_log_mean = hl_mean
    cfg.half_life_log_std = hl_std
    cfg.half_life_seconds_loss_scale = max(1.0, float(np.expm1(hl_mean + hl_std)))

    datasets = {
        split: BubbleSequenceDataset(
            df_model,
            features,
            split_windows,
            cfg,
            half_life_log_mean=hl_mean,
            half_life_log_std=hl_std,
            training=(split == "train"),
        )
        for split, split_windows in windows.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, preprocessor, features, (hl_mean, hl_std)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state(self, batch_size: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SeriousBubbleConvLSTM(nn.Module):
    def __init__(self, feature_dim: int, cfg: SeriousConvLSTMConfig):
        super().__init__()
        h1, h2 = cfg.hidden_channels
        self.cfg = cfg
        self.encoder = nn.Sequential(
            ConvBlock(1, 24, stride=1),
            ConvBlock(24, h1, stride=2),
        )
        self.convlstm1 = ConvLSTMCell(h1, h1)
        self.convlstm2 = ConvLSTMCell(h1, h2)
        self.feature_gru = nn.GRU(feature_dim, cfg.feature_hidden, batch_first=True)
        self.film = nn.Linear(cfg.feature_hidden, 2 * h2)
        self.decoder = nn.Sequential(
            nn.Conv2d(h2, h1, kernel_size=3, padding=1),
            nn.BatchNorm2d(h1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(h1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 1, kernel_size=3, padding=1),
        )
        self.half_life_head = nn.Sequential(
            nn.Linear(h2 + cfg.feature_hidden, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(192, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 1),
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, timesteps, _, height, width = images.shape
        encoded0 = self.encoder(images[:, 0])
        _, _, eh, ew = encoded0.shape
        h1, c1 = self.convlstm1.init_state(batch_size, eh, ew, images.device)
        h2, c2 = self.convlstm2.init_state(batch_size, eh, ew, images.device)

        for t in range(timesteps):
            encoded = encoded0 if t == 0 else self.encoder(images[:, t])
            h1, c1 = self.convlstm1(encoded, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))

        _, feature_state = self.feature_gru(features)
        feature_state = feature_state[-1]
        gamma_beta = self.film(feature_state)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        h2_film = h2 * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        delta = torch.tanh(self.decoder(h2_film)) * self.cfg.delta_scale
        last_frame = images[:, -1]
        next_image = torch.clamp(last_frame + delta, 0.0, 1.0)

        image_state = F.adaptive_avg_pool2d(h2, 1).flatten(1)
        half_life_scaled_log = self.half_life_head(torch.cat([image_state, feature_state], dim=1))
        return next_image, half_life_scaled_log


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def image_gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def ssim_index(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    padding = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return value.mean()


def count_binary_components(mask: np.ndarray) -> int:
    """Count 8-connected foreground components in one 2D binary mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return 0
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    components = 0
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            components += 1
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def binary_component_counts(images: np.ndarray, threshold: float) -> np.ndarray:
    """Return component counts for a batch shaped [B, 1, H, W] or [B, H, W]."""
    images = np.asarray(images)
    if images.ndim == 4:
        images = images[:, 0]
    return np.asarray([count_binary_components(image > threshold) for image in images], dtype=np.float32)


def seconds_from_scaled_log(value: np.ndarray | torch.Tensor, mean: float, std: float) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.expm1(value * std + mean)


def log_cosh_loss(error: torch.Tensor) -> torch.Tensor:
    return torch.mean(error + F.softplus(-2.0 * error) - math.log(2.0))


def multitask_loss(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    pred_hl: torch.Tensor,
    target_hl: torch.Tensor,
    observed_mask: torch.Tensor,
    censor_hl: torch.Tensor,
    censor_mask: torch.Tensor,
    cfg: SeriousConvLSTMConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(pred_image, target_image)
    mse = F.mse_loss(pred_image, target_image)
    ssim_loss = 1.0 - ssim_index(pred_image, target_image)
    grad = gradient_l1_loss(pred_image, target_image)

    observed_count = observed_mask.sum().clamp_min(1.0)
    half_life = (F.smooth_l1_loss(pred_hl, target_hl, reduction="none") * observed_mask).sum() / observed_count
    hl_error_scaled = (pred_hl - target_hl) * observed_mask
    half_life_mse_scaled = ((hl_error_scaled ** 2).sum() / observed_count).clamp_min(0.0)
    half_life_log_cosh = (
        (hl_error_scaled + F.softplus(-2.0 * hl_error_scaled) - math.log(2.0)) * observed_mask
    ).sum() / observed_count

    hl_mean = float(getattr(cfg, "half_life_log_mean", 0.0))
    hl_std = float(getattr(cfg, "half_life_log_std", 1.0))
    seconds_scale = float(getattr(cfg, "half_life_seconds_loss_scale", 0.0))
    if seconds_scale <= 0:
        seconds_scale = max(1.0, math.expm1(hl_mean + hl_std))
    pred_seconds_norm = torch.expm1(pred_hl * hl_std + hl_mean).clamp_min(0.0) / seconds_scale
    target_seconds_norm = torch.expm1(target_hl * hl_std + hl_mean).clamp_min(0.0) / seconds_scale
    seconds_error = (pred_seconds_norm - target_seconds_norm) * observed_mask
    half_life_seconds_mae = seconds_error.abs().sum() / observed_count
    half_life_seconds_mse = (seconds_error ** 2).sum() / observed_count

    censored_count = censor_mask.sum().clamp_min(1.0)
    censored = (F.relu(censor_hl - pred_hl) ** 2 * censor_mask).sum() / censored_count

    total = (
        cfg.image_l1_weight * l1
        + cfg.image_mse_weight * mse
        + cfg.image_ssim_weight * ssim_loss
        + cfg.image_gradient_weight * grad
        + cfg.half_life_weight * half_life
        + cfg.half_life_seconds_mae_weight * half_life_seconds_mae
        + cfg.half_life_seconds_mse_weight * half_life_seconds_mse
        + cfg.half_life_log_cosh_weight * half_life_log_cosh
        + cfg.censored_half_life_weight * censored
    )
    parts = {
        "loss": total.detach(),
        "image_l1": l1.detach(),
        "image_mse": mse.detach(),
        "image_ssim": (1.0 - ssim_loss).detach(),
        "image_grad": grad.detach(),
        "half_life_smooth_l1_scaled": half_life.detach(),
        "half_life_mse_scaled": half_life_mse_scaled.detach(),
        "half_life_log_cosh": half_life_log_cosh.detach(),
        "half_life_seconds_mae_norm": half_life_seconds_mae.detach(),
        "half_life_seconds_mse_norm": half_life_seconds_mse.detach(),
        "censored_loss": censored.detach(),
        "observed_half_life_count": observed_mask.sum().detach(),
        "censored_half_life_count": censor_mask.sum().detach(),
    }
    return total, parts


class MetricTracker:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

    def update(self, metrics: dict[str, torch.Tensor | float], n: int) -> None:
        for key, value in metrics.items():
            scalar = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
            if key.endswith("_count"):
                self.sums[key] += scalar
                self.counts[key] += 1.0
            else:
                self.sums[key] += scalar * n
                self.counts[key] += n

    def average(self) -> dict[str, float]:
        return {key: self.sums[key] / max(1.0, self.counts[key]) for key in self.sums}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: SeriousConvLSTMConfig,
    desc: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    tracker = MetricTracker()

    iterator = loader
    if cfg.show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc or ("train" if training else "eval"), leave=False)

    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        target_image = batch["target_image"].to(device, non_blocking=True)
        target_hl = batch["half_life"].to(device, non_blocking=True)
        observed_mask = batch["half_life_observed_mask"].to(device, non_blocking=True)
        censor_hl = batch["half_life_censor"].to(device, non_blocking=True)
        censor_mask = batch["half_life_censor_mask"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred_image, pred_hl = model(images, features)
            loss, parts = multitask_loss(
                pred_image,
                target_image,
                pred_hl,
                target_hl,
                observed_mask,
                censor_hl,
                censor_mask,
                cfg,
            )

        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        batch_size = int(images.shape[0])
        tracker.update(parts, batch_size)

        if cfg.show_progress and tqdm is not None:
            avg = tracker.average()
            iterator.set_postfix(
                loss=f"{avg.get('loss', float('nan')):.4f}",
                l1=f"{avg.get('image_l1', float('nan')):.4f}",
                hl=f"{avg.get('half_life_smooth_l1_scaled', float('nan')):.3f}",
            )

    return tracker.average()


def evaluate_extra_metrics(
    model: nn.Module,
    loader: DataLoader,
    df_model: pd.DataFrame,
    device: torch.device,
    half_life_log_mean: float,
    half_life_log_std: float,
    show_progress: bool = False,
    desc: str = "metrics",
) -> dict[str, float]:
    model.eval()
    image_sq_error = 0.0
    image_abs_error = 0.0
    image_count = 0
    image_ssim_sum = 0.0
    image_sample_count = 0
    edge_abs_error = 0.0
    edge_count = 0
    binary_area_abs_error = []
    component_count_abs_error = []
    hl_pred = []
    hl_true = []
    censored_pred = []
    censored_lower_bound = []

    with torch.no_grad():
        iterator = loader
        if show_progress and tqdm is not None:
            iterator = tqdm(loader, desc=desc, leave=False)
        for batch in iterator:
            images = batch["images"].to(device)
            features = batch["features"].to(device)
            target_image = batch["target_image"].to(device)
            pred_image, pred_hl = model(images, features)

            diff = pred_image - target_image
            image_sq_error += float((diff * diff).sum().item())
            image_abs_error += float(diff.abs().sum().item())
            image_count += int(diff.numel())
            batch_size = int(diff.shape[0])
            image_ssim_sum += float(ssim_index(pred_image, target_image).item()) * batch_size
            image_sample_count += batch_size

            pred_edges = image_gradient_magnitude(pred_image)
            target_edges = image_gradient_magnitude(target_image)
            edge_diff = pred_edges - target_edges
            edge_abs_error += float(edge_diff.abs().sum().item())
            edge_count += int(edge_diff.numel())

            threshold = float(getattr(getattr(loader, "dataset", None), "cfg", SeriousConvLSTMConfig()).morphology_threshold)
            pred_np = pred_image.detach().cpu().float().numpy()
            target_np = target_image.detach().cpu().float().numpy()
            pred_binary = pred_np > threshold
            target_binary = target_np > threshold
            pred_area = pred_binary.reshape(batch_size, -1).mean(axis=1)
            target_area = target_binary.reshape(batch_size, -1).mean(axis=1)
            binary_area_abs_error.append(np.abs(pred_area - target_area).astype(np.float32))
            pred_components = binary_component_counts(pred_np, threshold)
            target_components = binary_component_counts(target_np, threshold)
            component_count_abs_error.append(np.abs(pred_components - target_components).astype(np.float32))

            target_pos = batch["target_pos"].cpu().numpy()
            observed = df_model.loc[target_pos, "half_life_observed"].to_numpy(dtype=np.float32) > 0.5
            actual = df_model.loc[target_pos, "half_life_sec"].to_numpy(dtype=np.float32)
            censor = df_model.loc[target_pos, "half_life_censor_sec"].to_numpy(dtype=np.float32)
            pred_seconds = seconds_from_scaled_log(pred_hl, half_life_log_mean, half_life_log_std).reshape(-1)
            valid = observed & np.isfinite(actual)
            if np.any(valid):
                hl_pred.append(pred_seconds[valid])
                hl_true.append(actual[valid])
            valid_censored = (~observed) & np.isfinite(censor)
            if np.any(valid_censored):
                censored_pred.append(pred_seconds[valid_censored])
                censored_lower_bound.append(censor[valid_censored])

    mse = image_sq_error / max(1, image_count)
    mae = image_abs_error / max(1, image_count)
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))
    metrics = {
        "pixel_mae": mae,
        "pixel_rmse": math.sqrt(mse),
        "psnr_db": psnr,
        "ssim": image_ssim_sum / max(1, image_sample_count),
        "edge_mae": edge_abs_error / max(1, edge_count),
    }
    if binary_area_abs_error:
        metrics["binary_area_fraction_mae"] = float(np.mean(np.concatenate(binary_area_abs_error)))
    else:
        metrics["binary_area_fraction_mae"] = np.nan
    if component_count_abs_error:
        metrics["component_count_mae"] = float(np.mean(np.concatenate(component_count_abs_error)))
    else:
        metrics["component_count_mae"] = np.nan
    if hl_pred:
        pred = np.concatenate(hl_pred)
        true = np.concatenate(hl_true)
        metrics.update(
            {
                "half_life_mae_sec": float(np.mean(np.abs(pred - true))),
                "half_life_rmse_sec": float(np.sqrt(np.mean((pred - true) ** 2))),
                "half_life_n": int(len(true)),
            }
        )
    else:
        metrics.update({"half_life_mae_sec": np.nan, "half_life_rmse_sec": np.nan, "half_life_n": 0})
    if censored_pred:
        pred = np.concatenate(censored_pred)
        lower = np.concatenate(censored_lower_bound)
        violation = np.maximum(0.0, lower - pred)
        metrics.update(
            {
                "censored_half_life_n": int(len(lower)),
                "censored_lower_bound_satisfied_rate": float(np.mean(pred >= lower)),
                "censored_violation_mae_sec": float(np.mean(violation)),
            }
        )
    else:
        metrics.update(
            {
                "censored_half_life_n": 0,
                "censored_lower_bound_satisfied_rate": np.nan,
                "censored_violation_mae_sec": np.nan,
            }
        )
    return metrics


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    df_model: pd.DataFrame,
    cfg: SeriousConvLSTMConfig,
    device: torch.device,
    half_life_log_mean: float,
    half_life_log_std: float,
) -> pd.DataFrame:
    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(exist_ok=True)
    best_path = artifact_dir / "best_serious_convlstm.pt"
    history_path = artifact_dir / "training_history.csv"
    config_path = artifact_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = math.inf
    bad_epochs = 0
    rows = []
    print(
        "training setup | "
        f"device={device} | "
        f"amp={device.type == 'cuda'} | "
        f"train_batches={len(loaders['train'])} | "
        f"val_batches={len(loaders['val'])} | "
        f"batch_size={cfg.batch_size} | "
        f"image_size={cfg.image_size} | "
        f"seq_len={cfg.seq_len}"
    )
    for epoch in range(1, cfg.epochs + 1):
        print(f"epoch {epoch:03d}/{cfg.epochs}")
        train_metrics = run_epoch(
            model, loaders["train"], optimizer, scaler, device, cfg, desc=f"epoch {epoch:03d} train"
        )
        val_metrics = run_epoch(model, loaders["val"], None, None, device, cfg, desc=f"epoch {epoch:03d} val")
        val_extra = evaluate_extra_metrics(
            model,
            loaders["val"],
            df_model,
            device,
            half_life_log_mean,
            half_life_log_std,
            show_progress=cfg.show_progress,
            desc=f"epoch {epoch:03d} metrics",
        )
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"val_{k}": v for k, v in val_extra.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(history_path, index=False)

        print(
            f"epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.4f} | "
            f"val {val_metrics['loss']:.4f} | "
            f"val L1 {val_metrics['image_l1']:.4f} | "
            f"val PSNR {val_extra['psnr_db']:.2f} dB | "
            f"val half-life MAE {val_extra['half_life_mae_sec']:.1f}s"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "half_life_log_mean": half_life_log_mean,
                    "half_life_log_std": half_life_log_std,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stopping at epoch {epoch}; best val loss {best_val:.4f}")
                break

    return pd.DataFrame(rows)


def load_best_model(
    model: nn.Module, cfg: SeriousConvLSTMConfig, device: torch.device
) -> dict[str, Any]:
    checkpoint_path = Path(cfg.artifact_dir) / "best_serious_convlstm.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


def collect_half_life_predictions(
    model: nn.Module,
    loader: DataLoader,
    df_model: pd.DataFrame,
    device: torch.device,
    half_life_log_mean: float,
    half_life_log_std: float,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            pred_image, pred_hl = model(batch["images"].to(device), batch["features"].to(device))
            del pred_image
            pred_seconds = seconds_from_scaled_log(pred_hl, half_life_log_mean, half_life_log_std).reshape(-1)
            target_pos = batch["target_pos"].cpu().numpy()
            for pos, pred in zip(target_pos, pred_seconds):
                row = df_model.loc[int(pos)]
                observed = float(row["half_life_observed"]) > 0.5
                censor_sec = float(row["half_life_censor_sec"]) if np.isfinite(row["half_life_censor_sec"]) else np.nan
                rows.append(
                    {
                        "target_pos": int(pos),
                        "run_key": row["run_key"],
                        "surfactant": row["surfactant"],
                        "nanoparticle": row["nanoparticle"],
                        "concentration": row["concentration"],
                        "run_id": row["run_id_str"],
                        "pred_half_life_sec": float(pred),
                        "actual_half_life_sec": float(row["half_life_sec"])
                        if np.isfinite(row["half_life_sec"])
                        else np.nan,
                        "half_life_observed": float(row["half_life_observed"]),
                        "half_life_label_type": "observed" if observed else "right_censored",
                        "half_life_censor_sec": censor_sec,
                        "prediction_satisfies_censor_bound": bool(pred >= censor_sec)
                        if (not observed and np.isfinite(censor_sec))
                        else np.nan,
                        "half_life_confidence_flag": "exact_threshold_crossing" if observed else "lower_bound_only",
                        "half_life_source": row["half_life_source"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_windows(windows: dict[str, list[tuple[np.ndarray, int]]], runs: dict[str, set[int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"split": split, "runs": len(runs[split]), "windows": len(windows[split])}
            for split in ["train", "val", "test"]
        ]
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
