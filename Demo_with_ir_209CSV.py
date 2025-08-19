# -*- coding: utf-8 -*-
"""
IR-Model (42,7) con TSS vs PLT (vector diario = [TSS, HRVpre, kJ/kg, HM, HRVpost])
Archivo esperado: IR_209.csv (del atleta 209) con columnas:
index (fechas como 'YYYY-MM-DD'), fecha (TSS), tss (HRV_pre), kjkg, hm, hrv_post
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "IR_209.csv"  # ajustá la ruta si hace falta

# ---------- utilidades ----------
def softplus(x: np.ndarray) -> np.ndarray:
    # versión estable numéricamente
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def ir_recursive(u: np.ndarray, tau: float) -> np.ndarray:
    """
    IR-Model discreto (EMA): x_t = ρ x_{t-1} + (1-ρ) u_t,  ρ = exp(-1/τ)
    Arranque: x_0 = (1-ρ) * u_0
    """
    rho = float(np.exp(-1.0 / tau))
    x = np.zeros(len(u), dtype=float)
    if len(u) == 0:
        return x
    x[0] = (1.0 - rho) * u[0]
    for i in range(1, len(u)):
        x[i] = rho * x[i-1] + (1.0 - rho) * u[i]
    return x

# ---------- carga y adaptación al formato de tu CSV ----------
df = pd.read_csv(PATH)
df.columns = [c.strip().lower() for c in df.columns]


if not df.index.is_monotonic_increasing:
    df = df.copy()

# Fechas desde el índice:
try:
    dates = pd.to_datetime(df.index, errors="raise")
    use_index_as_dates = True
except Exception:
    use_index_as_dates = False

if use_index_as_dates:
    # Mapear columnas según tu CSV:
    ser_tss      = pd.to_numeric(df["fecha"], errors="coerce").fillna(0.0)   # TSS
    ser_hrvpre   = pd.to_numeric(df["tss"], errors="coerce").fillna(0.0)     # HRV pre (rMSSD)
    ser_kjkg     = pd.to_numeric(df.get("kjkg", 0.0), errors="coerce").fillna(0.0)
    ser_hm       = pd.to_numeric(df.get("hm", 0.0), errors="coerce").fillna(0.0)
    ser_hrv_post = pd.to_numeric(df.get("hrv_post", 0.0), errors="coerce").fillna(0.0)

    order = np.argsort(dates.values)
    dates = dates.values[order]
    tss      = ser_tss.values[order]
    hrvpre   = ser_hrvpre.values[order]
    kjkg     = ser_kjkg.values[order]
    hm       = ser_hm.values[order]
    hrv_post = ser_hrv_post.values[order]
else:
    # Fallback general (si las fechas no estuvieran en el índice):
    dates = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.assign(
        tss=pd.to_numeric(df["tss"], errors="coerce"),
        hrvpre=pd.to_numeric(df.get("hrv", 0.0), errors="coerce"),  # si viniera en 'hrv'
        kjkg=pd.to_numeric(df.get("kjkg", 0.0), errors="coerce"),
        hm=pd.to_numeric(df.get("hm", 0.0), errors="coerce"),
        hrv_post=pd.to_numeric(df.get("hrv_post", 0.0), errors="coerce"),
    ).sort_values(dates.name)
    dates = pd.to_datetime(df["fecha"], errors="coerce").values
    tss      = df["tss"].fillna(0.0).to_numpy()
    hrvpre   = df["hrvpre"].fillna(0.0).to_numpy()
    kjkg     = df["kjkg"].fillna(0.0).to_numpy()
    hm       = df["hm"].fillna(0.0).to_numpy()
    hrv_post = df["hrv_post"].fillna(0.0).to_numpy()

# ---------- reindexado diario (completa días faltantes con 0 de impulso) ----------
start = pd.to_datetime(dates.min())
end   = pd.to_datetime(dates.max())
idx   = pd.date_range(start, end, freq="D")

g = pd.DataFrame(index=idx)
g["tss"]      = 0.0
g["hrvpre"]   = 0.0
g["kjkg"]     = 0.0
g["hm"]       = 0.0
g["hrv_post"] = 0.0

g.loc[pd.to_datetime(dates), "tss"]      = tss
g.loc[pd.to_datetime(dates), "hrvpre"]   = hrvpre
g.loc[pd.to_datetime(dates), "kjkg"]     = kjkg
g.loc[pd.to_datetime(dates), "hm"]       = hm
g.loc[pd.to_datetime(dates), "hrv_post"] = hrv_post

# ---------- PLT diario: robust scaling + proyección β + softplus ----------
plt_cols = ["tss", "hrvpre", "kjkg", "hm", "hrv_post"]
med  = g[plt_cols].median()
iqr  = (g[plt_cols].quantile(0.75) - g[plt_cols].quantile(0.25)).replace(0, np.nan)
std_fb = g[plt_cols].std().replace(0, np.nan)
denom = iqr.fillna(std_fb).fillna(1.0)

Xrs = ((g[plt_cols] - med) / denom).fillna(0.0).values  # robust-scaled
C   = len(plt_cols)
beta = np.ones(C, dtype=float) / np.sqrt(C)             # β neutra
u_plt = softplus(Xrs.dot(beta))                          # impulso PLT diario

# Impulso TSS directo (por si querés comparar)
u_tss = g["tss"].values.astype(float)

# ---------- IR-Model (42,7) y TSB con lag=1 ----------
ctl_tss = ir_recursive(u_tss, tau=42.0)
atl_tss = ir_recursive(u_tss, tau=7.0)
tsb_tss = np.r_[np.nan, ctl_tss[:-1] - atl_tss[:-1]]

ctl_plt = ir_recursive(u_plt, tau=42.0)
atl_plt = ir_recursive(u_plt, tau=7.0)
tsb_plt = np.r_[np.nan, ctl_plt[:-1] - atl_plt[:-1]]

# ---------- gráfico ----------
fig, ax_left = plt.subplots(figsize=(12, 5))
ax_left.plot(g.index, ctl_tss, label="CTL (TSS, τ=42)", color="blue", linewidth=1.8)
ax_left.plot(g.index, tsb_tss, label="TSB (TSS, lag1)", color="green", linewidth=1.5)
ax_left.set_ylabel("CTL / TSB (TSS)")
ax_left.grid(True, alpha=0.3)

ax_right = ax_left.twinx()
ax_right.plot(g.index, ctl_plt, label="CTL (PLT, τ=42)", color="red", linewidth=1.8)
ax_right.plot(g.index, tsb_plt, label="TSB (PLT, lag1)", color="pink", linewidth=1.5)
ax_right.set_ylabel("CTL / TSB (PLT)")

ax_left.set_title("Athlete 209 — IR-Model (42,7): TSS (izq) vs PLT (der)\nPLT=[TSS, HRVpre, kJ/kg, HM, HRVpost] (robust-scaled, β=1/√C)")
lines_l, labs_l = ax_left.get_legend_handles_labels()
lines_r, labs_r = ax_right.get_legend_handles_labels()
ax_left.legend(lines_l + lines_r, labs_l + labs_r, loc="upper left", fontsize=9)
fig.tight_layout()
plt.show()

# ---------- (opcional) exportar CSV de resultados ----------
# df_out = pd.DataFrame({
#     "fecha": g.index,
#     "ctl_tss": ctl_tss,
#     "tsb_tss": tsb_tss,
#     "ctl_plt": ctl_plt,
#     "tsb_plt": tsb_plt,
# })
# df_out.to_csv("IR_209_TSS_vs_PLT.csv", index=False)
