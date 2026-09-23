import argparse
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split


class _LegacyPandasUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("pandas.indexes"):
                new_module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
            elif module.startswith("pandas.core.index") and "indexes" not in module:
                new_module = module.replace("pandas.core.index", "pandas.core.indexes", 1)
            elif module.startswith("pandas.core.internals") and not module.endswith("managers"):
                new_module = "pandas.core.internals.managers"
            else:
                raise
            return super().find_class(new_module, name)


VALID_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Random", "Scratch", "Near-full", "none"
]

WAFER_MAP_SIZE = 64


def load_raw(pickle_path: str) -> pd.DataFrame:
    with open(pickle_path, "rb") as f:
        df = _LegacyPandasUnpickler(f, encoding="latin1").load()
    return df


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    def flatten_label(x):
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            # A malformed multi-element label array gets stringified here and then
            # dropped below by the isin(VALID_CLASSES) filter, since it won't match
            # any valid class name.
            return flatten_label(x.item(0)) if x.size == 1 else str(x)
        return x

    df = df.copy()
    df["failureType"] = df["failureType"].apply(flatten_label)
    df = df[df["failureType"].isin(VALID_CLASSES)].reset_index(drop=True)
    return df


def resize_wafer_map(wafer_map: np.ndarray, size: int = WAFER_MAP_SIZE) -> np.ndarray:
    h, w = wafer_map.shape
    zoom_factors = (size / h, size / w)
    resized = zoom(wafer_map, zoom_factors, order=0)
    return resized


def build_dataset(df: pd.DataFrame, size: int = WAFER_MAP_SIZE):
    label_to_idx = {label: i for i, label in enumerate(VALID_CLASSES)}

    X = np.zeros((len(df), size, size, 1), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)

    for i, row in enumerate(df.itertuples()):
        resized = resize_wafer_map(row.waferMap, size)
        X[i, :, :, 0] = resized / 2.0
        y[i] = label_to_idx[row.failureType]

    return X, y, label_to_idx


def train_val_test_split(X, y, test_size=0.15, val_size=0.15, seed=42):
    # NOTE: this split does not deduplicate wafer maps. Exact-duplicate wafer maps
    # across splits are detected and removed downstream in the notebook (Section 1)
    # before training — see wafer_defect_cnn.ipynb for the leakage check.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=seed
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val/test arrays from raw WM-811K pickle.")
    parser.add_argument("--input", default="data/LSWMD.pkl", help="Path to raw LSWMD.pkl")
    parser.add_argument("--output", default="data/wm811k_processed.npz", help="Path to save processed .npz")
    args = parser.parse_args()

    df = load_raw(args.input)
    df = clean_labels(df)
    print(f"Labeled wafers: {len(df)}")
    print(df["failureType"].value_counts())

    X, y, label_map = build_dataset(df)
    print("X shape:", X.shape, "y shape:", y.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    np.savez_compressed(
        args.output,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print(f"Saved processed arrays to {args.output}")