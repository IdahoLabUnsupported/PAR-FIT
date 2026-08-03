# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:47:04 2026

Copyright 2026, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

@author: Edward Chen
@email: edward.chen@inl.gov
"""

import os
import re
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Expected columns in each CSV
EXPECTED_COLUMNS = ["FL1", "FL6", "TA21s1", "TL14s1", "PS1", "PS2"]


def find_history_files(folder: str, file_name) -> List[str]:
    """
    Find files matching histories_X.csv and sort them by numeric X.
    """
    pattern = os.path.join(folder, file_name)
    files = glob.glob(pattern)

    def parse_index(path: str) -> int:
        m = re.search(r"histories_(\d+)\.csv$", os.path.basename(path))
        return int(m.group(1)) if m else float("inf")

    files.sort(key=parse_index)
    return files


def load_matrix_per_file(path: str) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame containing only EXPECTED_COLUMNS.
    Validates presence and numeric content (raises if NaNs).
    """
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    # Coerce to numeric; fail if non-numeric/NaN found in expected columns
    df = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df.isnull().values.any():
        bad_rows = df.isnull().any(axis=1).sum()
        raise ValueError(f"{path} contains {bad_rows} row(s) with non-numeric or NaN values in expected columns.")
    return df


def compute_time_stats_across_files(
    files: List[str],
    ddof: int = 1,
    align: str = "trim",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute per-time-step mean and std across files, for each column.

    Parameters
    ----------
    files : list of str
        Paths to histories_X.csv files.
    ddof : int
        Delta degrees of freedom for std. Use 1 for sample std (default), 0 for population std.
    align : {'trim', 'strict'}
        - 'trim': trims all files to the minimum number of rows across files.
        - 'strict': raises if files have differing number of rows.

    Returns
    -------
    mean_ts : ndarray, shape (T, K)
        Mean over files for each time step and column (T rows, K=6 columns).
    std_ts : ndarray, shape (T, K)
        Std over files for each time step and column (ddof controllable).
    columns : list[str]
        Column names (order matches EXPECTED_COLUMNS).
    """
    if not files:
        raise FileNotFoundError("No histories_X.csv files provided.")

    matrices: List[pd.DataFrame] = []
    lengths: List[int] = []

    for path in files:
        df = load_matrix_per_file(path)
        matrices.append(df)
        lengths.append(len(df))

    if len(set(lengths)) != 1:
        if align == "strict":
            raise ValueError(f"Files have differing number of rows: {lengths}. "
                             f"Use align='trim' to proceed by trimming.")
        # Trim to min length
        T = min(lengths)
        print(f"[INFO] Files have differing lengths {lengths}. Trimming all to min length T={T}.")
        matrices = [df.iloc[:T].copy() for df in matrices]
    else:
        T = lengths[0]

    K = len(EXPECTED_COLUMNS)
    F = len(matrices)

    # Stack data: shape (F, T, K)
    stack = np.empty((F, T, K), dtype=np.float64)
    for i, df in enumerate(matrices):
        stack[i, :, :] = df.to_numpy(dtype=np.float64)

    # Mean and std across files (axis=0), per time step and column
    mean_ts = stack.mean(axis=0)                  # (T, K)
    std_ts = stack.std(axis=0, ddof=ddof)         # (T, K)

    return mean_ts, std_ts, EXPECTED_COLUMNS


def plot_time_stats(
    mean_ts: np.ndarray,
    std_ts: np.ndarray,
    columns: List[str],
    time: Optional[np.ndarray] = None,
    ddof: int = 1,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mean and mean ± std for each column in a 3×2 grid.

    Parameters
    ----------
    mean_ts : ndarray (T, K)
        Means per time step and column.
    std_ts : ndarray (T, K)
        Std per time step and column.
    columns : list[str]
        Column names (length K).
    time : ndarray or None
        Optional time array of length T. If None, uses np.arange(T).
    ddof : int
        ddof used for std (for labeling).
    title : str or None
        Figure title.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    T, K = mean_ts.shape
    assert K == len(columns), "Column count mismatch."

    # Time axis: either provided or row index 0..T-1
    t = time if time is not None else np.arange(T, dtype=int)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for j, col in enumerate(columns):
        ax = axes[j]
        m = mean_ts[:, j]
        s = std_ts[:, j]

        ax.plot(t, m, color="C0", lw=1.8, label="Mean")
        ax.fill_between(t, m - s, m + s, color="C0", alpha=0.25, label="Mean ± Std")
        ax.set_title(col)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    # Common X label on bottom row
    axes[-2].set_xlabel("Time")
    axes[-1].set_xlabel("Time")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved figure to: {save_path}")

    plt.show()


def plot_time_stats_for_folder(
    folder: str,
    file_name = None,
    ddof: int = 1,
    align: str = "trim",
    time: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Convenience function: load histories_X.csv files in a folder, compute stats,
    and plot the 6 subplots (one per column).

    Parameters
    ----------
    folder : str
        Folder containing histories_X.csv files.
    ddof : int
        1 for sample std (default), 0 for population std.
    align : {'trim', 'strict'}
        Handling of differing row counts across files.
    time : ndarray or None
        Optional time array to use on the x-axis (length T). If None, uses row index.
    save_path : str or None
        Path to save the figure. If None, just displays.
    """
    files = find_history_files(folder, file_name=file_name)
    if not files:
        raise FileNotFoundError(f"No files found matching {file_name} in {folder}")
    print(f"[INFO] Found {len(files)} files:\n  " + "\n  ".join(os.path.basename(f) for f in files))

    mean_ts, std_ts, columns = compute_time_stats_across_files(files, ddof=ddof, align=align)
    plot_time_stats(
        mean_ts,
        std_ts,
        columns,
        time=time,
        ddof=ddof,
        title=f"Per-time-step Mean & Std across {len(files)} files (ddof={ddof})",
        save_path=save_path,
    )

if __name__ == "__main__":
    file_name = "histories_*.csv"
    folder = "../datasets/007_Q2_015_0768_T/"
    plot_time_stats_for_folder(folder, file_name=file_name, ddof=1, align="trim", time=None, save_path="./Plot_016_Q8_175_5000_short.png")

