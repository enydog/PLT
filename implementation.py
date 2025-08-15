#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure‑Load Tensor (PLT) + Impulse–Response (IR) implementation
Gabriel Della Mattia 
Ednacore AI 2024
=============================================================

This module implements the minimal end‑to‑end pipeline described in your paper:
- Build PLT session vectors from daily/session aggregates
- Order‑aware inheritance for two stimuli in a day (HRV carry‑over)
- Aggregate to a per‑day PLT vector and compute a daily impulse u_d
- Run a Banister‑style IR model (fitness–fatigue) to obtain chronic load
- Compute PMC CTL/ATL/TSB for comparison
- Plot dual‑axis chart: CTL(TSS) vs tensorial chronic load F_d

Dependencies: numpy, pandas, matplotlib (for plotting)

Data schema (per session row):
    required columns (case-insensitive):
        date                : YYYY‑MM‑DD or timestamp
        session_id          : any hashable (string/int)
        tss                 : Training Stress Score (a.u.)
        mass_kg             : body mass (kg)               (if missing, use constant)
        kj                  : mechanical work (kJ)         (if missing, we can skip Ekg)
        hrv_pre_ms          : morning/pre rMSSD (ms)       (day-level, may repeat across sessions)
        hrv_post_ms         : first post-session rMSSD (ms)
        hrv_post2_ms        : (optional) late post (ms)
        dt01_h              : (optional) lag H0→H1 (hours); default 0.25 h
        dt12_h              : (optional) lag H1→H2 (hours); default 3.0 h
        hr_q1..hr_q4        : mean heart rate per time quartile (bpm)
        duration_s          : (optional) session duration in seconds (for weights)
        start_time          : (optional) timestamp to order same‑day sessions

    optional columns:
        athlete_id          : used for per‑athlete robust scaling
        gap_h_to_prev       : (optional) same‑day gap in hours; if absent we compute from start/end

If your file only has daily rows (no multiple sessions per day), keep one row per day.

Author: gato + assistant
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(x))


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z‑score using median/IQR; returns 0 if IQR≈0."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-8:
        return np.zeros_like(x)
    return (x - med) / iqr


def per_athlete_scale(df: pd.DataFrame, cols: List[str], athlete_col: Optional[str] = None) -> pd.DataFrame:
    """Apply robust_z per athlete (or globally if athlete_id not provided)."""
    out = df.copy()
    if athlete_col is None or athlete_col not in df.columns:
        for c in cols:
            out[c + "_rz"] = robust_z(out[c].values)
        return out
    for aid, grp in df.groupby(athlete_col):
        idx = grp.index
        for c in cols:
            out.loc[idx, c + "_rz"] = robust_z(grp[c].values)
    return out


# -----------------------------
# PLT feature construction
# -----------------------------
@dataclass
class PLTConfig:
    use_ekg: bool = True                   # include specific energy (kJ/kg)
    use_hr_quartiles: bool = True          # include HR quartiles (Q1..Q4)
    use_hrv_post: bool = True              # include HRV post
    use_lambda_hrv: bool = True            # include λ_HRV when H2 available, else v_HRV proxy
    default_mass: float = 70.0             # kg if mass_kg missing
    default_dt01_h: float = 0.25           # hours
    default_dt12_h: float = 3.0            # hours
    carry_over: bool = True                # enable same‑day inheritance when 2+ sessions
    weight_by_duration: bool = True        # daily aggregation weights


class PLTFeatures:
    """Build PLT session vectors and aggregate to daily vectors with carry‑over."""

    REQ_HR_QUARTS = ["hr_q1", "hr_q2", "hr_q3", "hr_q4"]

    def __init__(self, cfg: PLTConfig = PLTConfig(), athlete_col: Optional[str] = "athlete_id"):
        self.cfg = cfg
        self.athlete_col = athlete_col

    @staticmethod
    def _lambda_hrv(H0: float, H1: Optional[float], H2: Optional[float], dt01_h: float, dt12_h: Optional[float]) -> Tuple[float, float, bool]:
        """Return (lambda_or_v, hrv_post_used, is_lambda). If H2 is None, use v_HRV proxy."""
        if H0 is None or np.isnan(H0) or H1 is None or np.isnan(H1):
            return (np.nan, np.nan, False)
        if H2 is not None and not np.isnan(H2) and dt12_h and dt12_h > 0:
            lam = (1.0 / dt12_h) * np.log(max(abs(H1 - H0), 1e-6) / max(abs(H2 - H0), 1e-6))
            return (lam, H1, True)
        # proxy velocity
        dt01_h = dt01_h if dt01_h and dt01_h > 0 else 0.25
        v = (H1 - H0) / dt01_h
        return (v, H1, False)

    @staticmethod
    def _inherit_hrv_pre(H0: float, H1: float, lam_hrv: float, gap_h: float) -> float:
        """Exponential return toward baseline over the gap."""
        lam = lam_hrv if np.isfinite(lam_hrv) else 0.0
        gap = max(gap_h, 0.0)
        return float(H0 + (H1 - H0) * np.exp(-lam * gap))

    def _row_to_session_vec(self, row: pd.Series) -> Dict[str, float]:
        # Basic fields
        tss = float(row.get("tss", np.nan))
        mass = float(row.get("mass_kg", self.cfg.default_mass))
        kj = float(row.get("kj", np.nan))
        ekg = (kj / mass) if self.cfg.use_ekg and np.isfinite(kj) and mass > 0 else np.nan

        # HR quartiles
        hrq = [float(row.get(c, np.nan)) for c in self.REQ_HR_QUARTS] if self.cfg.use_hr_quartiles else [np.nan]*4

        # HRV pre/post
        H0 = float(row.get("hrv_pre_ms", np.nan))
        H1 = float(row.get("hrv_post_ms", np.nan)) if self.cfg.use_hrv_post else np.nan
        H2 = float(row.get("hrv_post2_ms", np.nan)) if self.cfg.use_lambda_hrv else np.nan
        dt01_h = float(row.get("dt01_h", self.cfg.default_dt01_h))
        dt12_h = float(row.get("dt12_h", self.cfg.default_dt12_h)) if np.isfinite(H2) else None
        lam_or_v, H1_used, is_lambda = self._lambda_hrv(H0, H1, H2, dt01_h, dt12_h)

        return {
            "TSS": tss,
            "Ekg": ekg,
            "HR_Q1": hrq[0],
            "HR_Q2": hrq[1],
            "HR_Q3": hrq[2],
            "HR_Q4": hrq[3],
            "HRV_pre": H0,
            "HRV_post": H1_used,
            "lambda_HRV": lam_or_v,   # λ if available, else v_HRV proxy
            "is_lambda": float(1.0 if is_lambda else 0.0),
        }

    def build_daily(self, df_sessions: pd.DataFrame) -> pd.DataFrame:
        """Return per‑day PLT vector after carry‑over and robust per‑athlete scaling.

        Output columns: date, athlete_id (if present), v_PLT_* (raw), v_PLT_*_rz (scaled),
                        u_plt (impulse with default β=1), plus diagnostics.
        """
        df = df_sessions.copy()
        # Normalize column names to lower for robustness
        df.columns = [c.strip() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        elif "fecha" in df.columns:
            df.rename(columns={"fecha": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            raise ValueError("A 'date' (or 'fecha') column is required.")

        # Fill defaults for time gaps if missing (used only for 2+ sessions/day)
        if "gap_h_to_prev" not in df.columns:
            df["gap_h_to_prev"] = np.nan

        # Build session vectors
        sess_vecs: List[Dict[str, float]] = []
        for _, r in df.iterrows():
            sess_vecs.append(self._row_to_session_vec(r))
        V = pd.DataFrame(sess_vecs)
        df = pd.concat([df.reset_index(drop=True), V], axis=1)

        # Order within day (if start_time present)
        if "start_time" in df.columns:
            df = df.sort_values(["athlete_id", "date", "start_time"], na_position="last") if "athlete_id" in df.columns else df.sort_values(["date", "start_time"])  # type: ignore
        else:
            df = df.sort_values(["athlete_id", "date"], na_position="last") if "athlete_id" in df.columns else df.sort_values(["date"])  # type: ignore

        # Carry‑over (two stimuli in a day): update HRV_pre of second using inheritance if gap available
        if self.cfg.carry_over:
            group_cols = ["athlete_id", "date"] if "athlete_id" in df.columns else ["date"]
            rows = []
            for _, g in df.groupby(group_cols):
                g = g.copy()
                if len(g) >= 2:
                    # compute gaps if missing: assume consecutive rows are ordered
                    gaps = g["gap_h_to_prev"].values
                    for i in range(1, len(g)):
                        gap = gaps[i]
                        if not np.isfinite(gap):
                            gap = 2.0  # sensible default when unknown
                        # apply inheritance using the *first* session's post & lambda
                        H0 = g.iloc[0]["HRV_pre"]
                        H1 = g.iloc[0]["HRV_post"]
                        lam = g.iloc[0]["lambda_HRV"] if np.isfinite(g.iloc[0]["lambda_HRV"]) else 0.0
                        g.iat[i, g.columns.get_loc("HRV_pre")] = self._inherit_hrv_pre(H0, H1, lam, float(gap))
                rows.append(g)
            df = pd.concat(rows, axis=0).sort_index()

        # Daily aggregation: weights
        if self.cfg.weight_by_duration and "duration_s" in df.columns:
            # per day normalize weights to sum 1
            df["w"] = df["duration_s"].clip(lower=0)
        else:
            df["w"] = 1.0
        # guard: if a day has all zeros, fallback to equal weights
        def _normalize_w(grp: pd.DataFrame) -> pd.Series:
            s = grp["w"].astype(float)
            tot = s.sum()
            if tot <= 0:
                return pd.Series(np.ones(len(s)) / max(len(s), 1), index=s.index)
            return s / tot
        df["w"] = df.groupby(["date"] + (["athlete_id"] if "athlete_id" in df.columns else []))["w"].transform(_normalize_w)

        # Aggregate to daily vector
        feat_cols = ["TSS", "Ekg", "HR_Q1", "HR_Q2", "HR_Q3", "HR_Q4", "HRV_pre", "HRV_post", "lambda_HRV"]
        def _agg_daily(grp: pd.DataFrame) -> pd.Series:
            w = grp["w"].values.reshape(-1, 1)
            X = grp[feat_cols].astype(float).values
            # weighted mean ignoring NaNs
            num = np.nansum(w * X, axis=0)
            den = np.nansum(w * (~np.isnan(X)), axis=0)
            m = np.divide(num, den, out=np.nan * np.ones_like(num), where=den > 0)
            s = pd.Series(m, index=[f"v_{c}" for c in feat_cols])
            return s

        group_cols = (["athlete_id", "date"] if "athlete_id" in df.columns else ["date"])
        daily = df.groupby(group_cols).apply(_agg_daily).reset_index()

        # Robust scaling per athlete
        scale_cols = [f"v_{c}" for c in feat_cols]
        daily = per_athlete_scale(daily, scale_cols, athlete_col=("athlete_id" if "athlete_id" in daily.columns else None))

        # Default β weights: emphasize TSS, Ekg, decoupling (Q4-Q1), HRV drop (pre-post), lambda
        daily["decoupling_rz"] = robust_z(daily["v_HR_Q4"].values - daily["v_HR_Q1"].values)
        daily["hrv_drop_rz"] = robust_z(daily["v_HRV_pre"].values - daily["v_HRV_post"].values)
        daily["lambda_rz"] = daily.get("v_lambda_HRV_rz", robust_z(daily["v_lambda_HRV"].values))

        rz_cols = [
            "v_TSS_rz", "v_Ekg_rz", "v_HR_Q1_rz", "v_HR_Q2_rz", "v_HR_Q3_rz", "v_HR_Q4_rz",
            "v_HRV_pre_rz", "v_HRV_post_rz", "v_lambda_HRV_rz",
            "decoupling_rz", "hrv_drop_rz", "lambda_rz"
        ]
        beta = np.array([1.0, 0.8, 0.2, 0.2, 0.4, 0.6, 0.3, 0.3, 0.6, 0.8, 0.8, 0.5])
        Xrz = np.column_stack([daily[c].values for c in rz_cols])
        proj = np.nan_to_num(Xrz, nan=0.0) @ beta
        daily["u_plt"] = softplus(proj)
        return daily


# -----------------------------
# IR model (fitness–fatigue) and PMC
# -----------------------------
@dataclass
class IRParams:
    tau_f: float = 42.0
    tau_g: float = 7.0
    k_f: float = 1.0
    k_g: float = 2.0
    P0: float = 0.0


def simulate_ir(u_f: np.ndarray, u_g: np.ndarray, pars: IRParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(u_f)
    F = np.zeros(n)
    G = np.zeros(n)
    rho_f = np.exp(-1.0 / float(pars.tau_f))
    rho_g = np.exp(-1.0 / float(pars.tau_g))
    for i in range(1, n):
        F[i] = rho_f * F[i - 1] + u_f[i - 1]
        G[i] = rho_g * G[i - 1] + u_g[i - 1]
    P_hat = float(pars.P0) + float(pars.k_f) * F - float(pars.k_g) * G
    return F, G, P_hat


def gridfit_ir(u: np.ndarray, y: np.ndarray, taus_f=(20, 30, 42, 60), taus_g=(3, 5, 7, 10, 14),
               kfs=(0.3, 0.5, 1.0, 1.5), kgs=(0.8, 1.0, 1.5, 2.0, 3.0)) -> IRParams:
    mask = np.isfinite(y)
    if mask.sum() < 6:
        return IRParams()
    P0 = float(np.nanmedian(y[mask]))
    best = None
    for tf in taus_f:
        for tg in taus_g:
            rho_f = np.exp(-1.0 / tf)
            rho_g = np.exp(-1.0 / tg)
            # precompute F,G for speed (single‑impulse assumption)
            F = np.zeros_like(u)
            G = np.zeros_like(u)
            for i in range(1, len(u)):
                F[i] = rho_f * F[i - 1] + u[i - 1]
                G[i] = rho_g * G[i - 1] + u[i - 1]
            for kf in kfs:
                for kg in kgs:
                    P_hat = P0 + kf * F - kg * G
                    mse = np.nanmean((y[mask] - P_hat[mask]) ** 2)
                    if best is None or mse < best[0]:
                        best = (mse, tf, tg, kf, kg, P0)
    _, tf, tg, kf, kg, P0 = best
    return IRParams(tau_f=tf, tau_g=tg, k_f=kf, k_g=kg, P0=P0)


def compute_ctl(tss: np.ndarray, tau_ctl: float = 42.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / tau_ctl)
    ctl = np.zeros_like(tss, dtype=float)
    if len(tss):
        ctl[0] = tss[0]
    for i in range(1, len(tss)):
        ctl[i] = ctl[i - 1] + alpha * (tss[i] - ctl[i - 1])
    return ctl


def compute_atl(tss: np.ndarray, tau_atl: float = 7.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / tau_atl)
    atl = np.zeros_like(tss, dtype=float)
    if len(tss):
        atl[0] = tss[0]
    for i in range(1, len(tss)):
        atl[i] = atl[i - 1] + alpha * (tss[i] - atl[i - 1])
    return atl


# -----------------------------
# High-level API
# -----------------------------

def run_pipeline(
    sessions_csv: Path | str,
    athlete_col: Optional[str] = "athlete_id",
    save_daily_csv: Optional[Path | str] = None,
    target_col: Optional[str] = None,
) -> Dict[str, any]:
    """Load sessions, build daily PLT + impulse, fit IR (if target provided),
    and compute PMC CTL for comparison.

    Returns a dict with daily dataframe and arrays.
    """
    df = pd.read_csv(sessions_csv)
    feat = PLTFeatures(athlete_col=athlete_col)
    daily = feat.build_daily(df)

    # Daily series
    daily = daily.sort_values(["athlete_id", "date"]) if "athlete_id" in daily.columns else daily.sort_values(["date"])  # type: ignore
    u = daily["u_plt"].values.astype(float)

    # Target if available (e.g., mmp5/ftp/CP estimate)
    pars = IRParams()
    if target_col and target_col in daily.columns:
        y = daily[target_col].values.astype(float)
        if np.isfinite(y).sum() >= 6:
            pars = gridfit_ir(u, y)
    F, G, P_hat = simulate_ir(u, u, pars)

    # CTL from TSS (if TSS present in daily vector)
    tss_daily = daily.get("v_TSS", None)
    ctl = compute_ctl(tss_daily.values.astype(float)) if tss_daily is not None else None

    if save_daily_csv is not None:
        out = daily.copy()
        out["F_tensor"] = F
        out["G_tensor"] = G
        out["P_hat"] = P_hat
        if ctl is not None:
            out["CTL_TSS"] = ctl
        out.to_csv(save_daily_csv, index=False)

    return {
        "daily": daily,
        "u": u,
        "ir_params": pars,
        "F": F,
        "G": G,
        "P_hat": P_hat,
        "CTL": ctl,
    }


# -----------------------------
# Example CLI usage
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PLT → IR pipeline")
    parser.add_argument("sessions_csv", type=str, help="CSV with per-session rows")
    parser.add_argument("--athlete-col", type=str, default="athlete_id")
    parser.add_argument("--target-col", type=str, default=None, help="Daily target column to fit IR (e.g., mmp5)")
    parser.add_argument("--save-daily", type=str, default="daily_plt_ir.csv")
    args = parser.parse_args()

    res = run_pipeline(args.sessions_csv, athlete_col=args.athlete_col, save_daily_csv=args.save_daily, target_col=args.target_col)
    print("Saved:", args.save_daily)
    print("IR params:", res["ir_params"])
