from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
READY_DATA = DATA_DIR / "Ready_data.csv"
OUT_CSV = DATA_DIR / "Ready_data_demo.csv"
OUT_PKL = DATA_DIR / "Ready_images_demo.pkl"
DEPLOY_DIR = DATA_DIR / "deployment_demo"

IMAGE_SIZE = 96

DEMO_RUNS = [
    ("Bioterg i wt.%", "AL2O3_Done", 0.00625, "1"),
    ("Bioterg i wt.%", "CeO", 0.0125, "3"),
    ("Bioterg i wt.%", "MnO2", 0.0125, "1"),
    ("Cocobetain 1 wt.%", "CNT", 0.00625, "1"),
    ("Cocobetain 1 wt.%", "Zno", 0.025, "1"),
]


def load_frame(path: str) -> np.ndarray | None:
    if not isinstance(path, str) or not path.strip():
        return None
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def main() -> None:
    df = pd.read_csv(READY_DATA, encoding="cp1252", low_memory=False)
    concentration_num = pd.to_numeric(df["concentration"], errors="coerce")
    run_id_str = df["run_id"].astype(str)

    masks = []
    for surfactant, nanoparticle, concentration, run_id in DEMO_RUNS:
        masks.append(
            (df["surfactant"].astype(str) == surfactant)
            & (df["nanoparticle"].astype(str) == nanoparticle)
            & (concentration_num == concentration)
            & (run_id_str == run_id)
        )

    keep = masks[0]
    for mask in masks[1:]:
        keep = keep | mask

    demo_df = df.loc[keep].copy().sort_values(["surfactant", "nanoparticle", "concentration", "run_id", "bd_row_idx"])
    demo_df = demo_df.reset_index(drop=True)
    demo_df.to_csv(OUT_CSV, index=False, encoding="cp1252")

    image_rows = demo_df[["surfactant", "nanoparticle", "concentration", "run_id", "bd_row_idx", "frame_path"]].copy()
    image_rows["image_array"] = [load_frame(path) for path in image_rows["frame_path"]]
    missing = int(image_rows["image_array"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} selected frames could not be loaded from frame_path.")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(image_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    DEPLOY_DIR.mkdir(exist_ok=True)
    shutil.copy2(OUT_CSV, DEPLOY_DIR / "Ready_data.csv")
    shutil.copy2(OUT_PKL, DEPLOY_DIR / "Ready_images.pkl")

    print(f"Wrote {OUT_CSV} with {len(demo_df)} rows")
    print(f"Wrote {OUT_PKL}")
    print(f"Wrote deployment-ready copies to {DEPLOY_DIR}")
    print(f"Demo pickle size: {OUT_PKL.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
