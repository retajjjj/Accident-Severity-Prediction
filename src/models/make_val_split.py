"""
Run ONCE after preprocessing, before train.py:
    poetry run python src/models/make_val_split.py
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42
VAL_FRACTION = 0.20          

def load(name: str):
    path = PROCESSED_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def save(obj, name: str):
    path = PROCESSED_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {name}.pkl  shape={getattr(obj, 'shape', len(obj))}")

def main():
    print("=" * 55)
    print("Creating validation split from X_train / y_train")
    print("=" * 55)

    X_train_full = load("X_train")
    y_train_full = load("y_train")

    y_arr = np.array(y_train_full, dtype=str)

    print(f"\n Original X_train shape : {X_train_full.shape}")
    print(f" Class distribution:\n{dict(zip(*np.unique(y_arr, return_counts=True)))}\n")

    # stratified split so class ratios are preserved in both halves
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_arr,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y_arr,
    )

    print("  After split:")
    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print()

    save(X_train, "X_train")
    save(y_train, "y_train")
    save(X_val,   "X_val")
    save(y_val,   "y_val")

if __name__ == "__main__":
    main()