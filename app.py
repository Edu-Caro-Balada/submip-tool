import io
import os
import glob
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go


# ============================
# Config
# ============================
CANON_COLS = [
    "Player_Id", "Player_displ", "Time", "Elap_Time", "Lat", "Lon",
    "Speedkmh", "Inst_acc", "HR", "V1", "filename"
]
MIN_RAW_COLS = 9

ROLLING_VARS = {
    "AccDens": "AccdensVen",
    "MetPow": "MetPowVen",
    "Dist": "DistVen",
    "HMLD": "HMLDVen",
    "HSR": "Dist_HSVen",
    "SP": "Dist_SDVen",
}

MIP_COLS = {
    "MetPow": "MIP_met",
    "HMLD": "MIP_HMLD",
    "HSR": "MIP_HSR",
    "SP": "MIP_sprint",
    "Dist": "MIP_dist",
    "AccDens": "MIP_accdens",
}

ASSETS_DIR = "assets"

RAW_PLOT_MAP = {
    "Dist": ("Speedmmin", "Velocidad (m/min)"),
    "AccDens": ("Accdens", "Acc dens (abs acc)"),
    "MetPow": ("MetPowf", "Potencia metabólica (W)"),
    "HMLD": ("HMLD", "HMLD (m/min condicional)"),
    "HSR": ("HSR", "HSR (m/min)"),
    "SP": ("Sprint", "Sprint (m/min)"),
}


@dataclass
class Params:
    fs: int = 10
    window_sec: int = 60
    thresholds_pct: List[int] = None


# ============================
# Lectura robusta
# ============================
def _looks_like_header(first_row: List[str]) -> bool:
    joined = " ".join([str(x).strip().lower() for x in first_row])
    keys = ["player", "display", "time", "elapsed", "lat", "lon", "speed", "acceleration", "heart", "rate", "bpm"]
    return any(k in joined for k in keys)


def _try_read(buf: io.BytesIO, sep: Optional[str], header: Optional[int]) -> pd.DataFrame:
    buf.seek(0)
    return pd.read_csv(buf, sep=sep, engine="python", header=header)


def read_csv_flexible(file_bytes: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)

    try:
        df = _try_read(buf, sep=None, header=None)
    except Exception:
        df = _try_read(buf, sep="\t", header=None)

    if df.shape[1] == 1:
        for sep in ["\t", ";", ",", "|"]:
            try:
                dft = _try_read(buf, sep=sep, header=None)
                if dft.shape[1] >= MIN_RAW_COLS:
                    df = dft
                    break
            except Exception:
                continue

    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).tolist()
        if _looks_like_header(first_row):
            try:
                df = _try_read(buf, sep=None, header=0)
            except Exception:
                ok = False
                for sep in ["\t", ";", ",", "|"]:
                    try:
                        df = _try_read(buf, sep=sep, header=0)
                        ok = True
                        break
                    except Exception:
                        continue
                if not ok:
                    raise

    if df.shape[1] < MIN_RAW_COLS:
        raise ValueError(f"El archivo {name} tiene {df.shape[1]} columnas. Se esperaban >= {MIN_RAW_COLS}.")

    if df.shape[1] > 11:
        df = df.iloc[:, :11].copy()

    n = df.shape[1]
    df = df.copy()
    base_name = os.path.splitext(os.path.basename(name))[0]

    if n == 9:
        df.columns = CANON_COLS[:9]
        df["V1"] = np.nan
        df["filename"] = base_name
    elif n == 10:
        df.columns = CANON_COLS[:10]
        df["filename"] = base_name
    else:
        df.columns = CANON_COLS[:11]
        df["filename"] = base_name

    return df


def read_raw_files_from_uploads(uploaded_files: List) -> pd.DataFrame:
    dfs = [read_csv_flexible(f.getvalue(), f.name) for f in uploaded_files]
    return pd.concat(dfs, ignore_index=True)


def list_example_files() -> List[str]:
    if not os.path.isdir(ASSETS_DIR):
        return []
    return sorted(glob.glob(os.path.join(ASSETS_DIR, "*.csv")))


def zip_examples_bytes(paths: List[str]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    mem.seek(0)
    return mem.read()


def read_raw_files_from_assets(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        with open(p, "rb") as f:
            dfs.append(read_csv_flexible(f.read(), os.path.basename(p)))
    return pd.concat(dfs, ignore_index=True)


# ============================
# Cálculo variables + rolling
# ============================
def butter_filters() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    b_speed, a_speed = butter(N=4, Wn=0.15, btype="low")
    b_acc, a_acc = butter(N=1, Wn=0.65, btype="low")
    return (b_speed, a_speed), (b_acc, a_acc)


def compute_signals_and_vars(df_raw: pd.DataFrame, fs: int) -> pd.DataFrame:
    dt = 1.0 / fs
    (b_speed, a_speed), (b_acc, a_acc) = butter_filters()

    df = df_raw.copy()

    for c in ["Elap_Time", "Lat", "Lon", "Speedkmh", "Inst_acc", "HR", "V1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Speed_ms"] = df["Speedkmh"] / 3.6

    df["_row"] = np.arange(len(df))
    df["Time_num"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.sort_values(["filename", "Player_displ", "Time_num", "_row"]).reset_index(drop=True)

    def _filt_speed(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        y = filtfilt(b_speed, a_speed, x)
        y[y < 0] = 0
        return y

    df["Speed_msf3"] = (
        df.groupby(["filename", "Player_displ"])["Speed_ms"]
          .transform(lambda s: pd.Series(_filt_speed(s.to_numpy()), index=s.index))
    )

    df["Acc"] = df.groupby(["filename", "Player_displ"])["Speed_msf3"].diff() / dt

    def _filt_acc(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        return filtfilt(b_acc, a_acc, x)

    df["Accf"] = (
        df.groupby(["filename", "Player_displ"])["Acc"]
          .transform(lambda s: pd.Series(_filt_acc(s.to_numpy()), index=s.index))
    )

    df["ES"] = (0.0037 * (df["Speed_msf3"] ** 2) / 9.81) + (df["Accf"] / 9.81)
    df["EM"] = np.sqrt((df["Accf"] ** 2) + (9.81 ** 2)) / 9.81

    ES = df["ES"]
    EM = df["EM"]
    v = df["Speed_msf3"]
    accf = df["Accf"]

    ec_pos = (
        ((155.4 * ES**5) - (30.4 * ES**4) - (43.3 * ES**3) + (46.3 * ES**2) + (19.5 * ES) + (3.6 * 1.29))
        * EM + (0.01 * (v**2))
    )
    ec_neg = (
        (-(30.4 * ES**4) - (5.0975 * ES**3) + (46.3 * ES**2) + (17.696 * ES) + (3.6 * 1.29))
        * EM + (0.01 * (v**2))
    )
    df["EC"] = np.where(accf > 0, ec_pos, ec_neg)

    df["MetPowf"] = df["EC"] * df["Speed_msf3"]
    df["dist"] = df["Speed_msf3"] * dt
    df["Accdens"] = np.abs(df["Accf"])

    df["SpeedHS"] = np.where(df["Speed_msf3"] > 5.5, df["Speed_msf3"], 0.0)
    df["SpeedSD"] = np.where(df["Speed_msf3"] > 7.0, df["Speed_msf3"], 0.0)

    df["Speedmmin"] = df["Speed_msf3"] * 60.0
    df["HSR"] = df["SpeedHS"] * 60.0
    df["Sprint"] = df["SpeedSD"] * 60.0

    df["HMLD"] = np.where(df["MetPowf"] > 25.5, df["Speed_msf3"] * 60.0, 0.0)

    df["sec"] = df.groupby(["filename", "Player_displ"]).cumcount() / fs

    keep = [
        "filename", "Player_Id", "Player_displ", "Time", "sec",
        "Speedmmin", "Accf", "MetPowf", "dist", "Accdens", "HSR", "Sprint", "HMLD"
    ]
    return df[keep].copy()


def compute_rollings(df_vars: pd.DataFrame, fs: int, window_sec: int) -> pd.DataFrame:
    window_n = fs * window_sec
    df = df_vars.copy()
    g = df.groupby(["filename", "Player_displ"])

    df["AccdensVen"] = g["Accdens"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)
    df["MetPowVen"] = g["MetPowf"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)
    df["Dist_HSVen"] = g["HSR"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)
    df["Dist_SDVen"] = g["Sprint"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)
    df["HMLDVen"] = g["HMLD"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)
    df["DistVen"] = g["Speedmmin"].rolling(window_n, min_periods=window_n).mean().reset_index(level=[0, 1], drop=True)

    for c in ["AccdensVen", "MetPowVen", "Dist_HSVen", "Dist_SDVen", "HMLDVen", "DistVen"]:
        df[c] = df[c].clip(lower=0)

    return df


# ============================
# MIPs: referencia + BULK + manual + fallback
# ============================
def read_mip_reference(uploaded_mip_file) -> Optional[pd.DataFrame]:
    if uploaded_mip_file is None:
        return None
    name = uploaded_mip_file.name.lower()
    content = uploaded_mip_file.getvalue()
    bio = io.BytesIO(content)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        ref = pd.read_excel(bio)
    else:
        ref = pd.read_csv(bio)
    ref.columns = [c.strip() for c in ref.columns]
    if "Player_displ" not in ref.columns:
        raise ValueError("El archivo de MIPs debe incluir la columna 'Player_displ'.")
    return ref


def make_mip_bulk_template(players: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"Player_displ": [str(p) for p in players]})
    for col in MIP_COLS.values():
        df[col] = np.nan
    return df


def read_mip_bulk(uploaded_bulk) -> Optional[pd.DataFrame]:
    if uploaded_bulk is None:
        return None
    content = uploaded_bulk.getvalue()
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip() for c in df.columns]
    if "Player_displ" not in df.columns:
        raise ValueError("El bulk de MIPs debe incluir la columna 'Player_displ'.")
    for col in MIP_COLS.values():
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Player_displ"] = df["Player_displ"].astype(str)
    return df[["Player_displ"] + list(MIP_COLS.values())].copy()


def build_manual_mip_dict_from_table(edited_table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if edited_table is None or edited_table.empty:
        return out
    for _, row in edited_table.iterrows():
        player = str(row["Player_displ"])
        d = {}
        for mip_col in MIP_COLS.values():
            if mip_col in edited_table.columns:
                val = row[mip_col]
                if pd.notna(val) and float(val) > 0:
                    d[mip_col] = float(val)
        if d:
            out[player] = d
    return out


def attach_mips(
    df_roll: pd.DataFrame,
    mip_ref: Optional[pd.DataFrame],
    mip_bulk: Optional[pd.DataFrame],
    manual_mips_by_player: Optional[Dict[str, Dict[str, float]]],
    use_match_fallback: bool,
) -> pd.DataFrame:
    out = df_roll.copy()

    # 1) referencia (si existe)
    if mip_ref is not None:
        out = out.merge(mip_ref, on="Player_displ", how="left")

    # asegurar columnas
    for mip_col in MIP_COLS.values():
        if mip_col not in out.columns:
            out[mip_col] = np.nan

    # 2) bulk (pisa referencia)
    if mip_bulk is not None and not mip_bulk.empty:
        cols = ["Player_displ"] + list(MIP_COLS.values())
        tmp = mip_bulk[cols].copy()
        out = out.merge(tmp, on="Player_displ", how="left", suffixes=("", "_bulk"))
        for mip_col in MIP_COLS.values():
            bcol = f"{mip_col}_bulk"
            if bcol in out.columns:
                out[mip_col] = out[bcol].combine_first(out[mip_col])
                out.drop(columns=[bcol], inplace=True)

    # 3) manual (pisa bulk + referencia)
    if manual_mips_by_player:
        for player, d in manual_mips_by_player.items():
            mask = out["Player_displ"] == player
            for mip_col, val in d.items():
                out.loc[mask, mip_col] = float(val)

    # 4) fallback (solo si sigue NA)
    if use_match_fallback:
        for var_label, mip_col in MIP_COLS.items():
            roll_col = ROLLING_VARS[var_label]
            grp_max = (
                out.groupby(["filename", "Player_displ"])[roll_col]
                   .max()
                   .rename(f"__fb_{mip_col}")
                   .reset_index()
            )
            out = out.merge(grp_max, on=["filename", "Player_displ"], how="left")
            out[mip_col] = out[mip_col].fillna(out[f"__fb_{mip_col}"])
            out.drop(columns=[f"__fb_{mip_col}"], inplace=True)

    return out


# ============================
# Eventos
# ============================
def merge_short_runs(x: np.ndarray, min_len: int) -> np.ndarray:
    if x.size == 0:
        return x
    x2 = x.copy()
    changes = np.where(np.diff(x2.astype(int)) != 0)[0] + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, x2.size]
    for s, e in zip(starts, ends):
        if (e - s) < min_len:
            x2[s:e] = True
    return x2


def detect_events_for_variable(
    df_one: pd.DataFrame,
    var_label: str,
    rolling_col: str,
    mip_col: str,
    thr: float,
    fs: int,
    window_sec: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window_n = fs * window_sec
    dfv = df_one.copy()
    dfv = dfv[dfv[rolling_col].notna()].reset_index(drop=True)

    base = {
        "filename": df_one["filename"].iloc[0],
        "Player_displ": df_one["Player_displ"].iloc[0],
        "threshold_pct": None,
        "Variable": var_label,
    }

    if dfv.empty:
        return pd.DataFrame(), pd.DataFrame([{**base, "events": 0, "dur_sec": 0.0}])

    mip_val = dfv[mip_col].iloc[0]
    if pd.isna(mip_val) or float(mip_val) <= 0:
        return pd.DataFrame(), pd.DataFrame([{**base, "events": 0, "dur_sec": 0.0}])

    x = (dfv[rolling_col].to_numpy() > (thr * float(mip_val)))
    x_merged = merge_short_runs(x, min_len=window_n)

    run_id = np.zeros_like(x_merged, dtype=int)
    run_id[0] = 1
    run_id[1:] = 1 + np.cumsum(x_merged[1:] != x_merged[:-1])

    dfv["_x"] = x_merged
    dfv["_run"] = run_id

    raw_col, _ = RAW_PLOT_MAP[var_label]

    ev_rows = []
    for _, g in dfv[dfv["_x"]].groupby("_run"):
        start_i = int(g.index.min())
        end_i = int(g.index.max())

        start_sec = float(dfv.loc[start_i, "sec"] - window_sec + (1.0 / fs))
        start_sec = max(start_sec, 0.0)
        end_sec = float(dfv.loc[end_i, "sec"])
        duration_sec = (end_sec - float(dfv.loc[start_i, "sec"])) + window_sec

        # métricas del evento (promedios)
        mean_roll = float(dfv.loc[start_i:end_i, rolling_col].mean())
        mean_raw = float(dfv.loc[start_i:end_i, raw_col].mean())

        row = {
            "filename": dfv["filename"].iloc[0],
            "Player_displ": dfv["Player_displ"].iloc[0],
            "Variable": var_label,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": float(duration_sec),
            "mip": float(mip_val),
            "thr": float(thr),
            "mean_roll": mean_roll,
            "mean_raw": mean_raw,
        }
        ev_rows.append(row)

    events_df = pd.DataFrame(ev_rows)
    if events_df.empty:
        summary_df = pd.DataFrame([{**base, "events": 0, "dur_sec": 0.0}])
    else:
        summary_df = pd.DataFrame([{
            **base,
            "events": int(events_df.shape[0]),
            "dur_sec": float(events_df["duration_sec"].sum())
        }])

    return events_df, summary_df


def compute_submip_for_thresholds(
    df_roll: pd.DataFrame,
    params: Params,
    mip_ref: Optional[pd.DataFrame],
    mip_bulk: Optional[pd.DataFrame],
    manual_mips_by_player: Optional[Dict[str, Dict[str, float]]],
    use_match_fallback: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    dfm = attach_mips(
        df_roll=df_roll,
        mip_ref=mip_ref,
        mip_bulk=mip_bulk,
        manual_mips_by_player=manual_mips_by_player,
        use_match_fallback=use_match_fallback
    )

    summaries_all = []
    events_all = []

    for thr_pct in params.thresholds_pct:
        thr = thr_pct / 100.0
        for (_, _), g in dfm.groupby(["filename", "Player_displ"]):
            for var_label, rolling_col in ROLLING_VARS.items():
                mip_col = MIP_COLS[var_label]
                ev, sm = detect_events_for_variable(
                    g, var_label=var_label, rolling_col=rolling_col, mip_col=mip_col,
                    thr=thr, fs=params.fs, window_sec=params.window_sec
                )
                sm["threshold_pct"] = thr_pct
                summaries_all.append(sm)
                if ev is not None and not ev.empty:
                    ev["threshold_pct"] = thr_pct
                    events_all.append(ev)

    summary_all = pd.concat(summaries_all, ignore_index=True) if summaries_all else pd.DataFrame()
    events_all = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()
    return summary_all, events_all


def summary_wide_from_long(summary_long: pd.DataFrame) -> pd.DataFrame:
    if summary_long.empty:
        return pd.DataFrame()

    piv_e = summary_long.pivot_table(
        index=["filename", "Player_displ", "threshold_pct"],
        columns="Variable",
        values="events",
        aggfunc="first",
        fill_value=0
    )
    piv_d = summary_long.pivot_table(
        index=["filename", "Player_displ", "threshold_pct"],
        columns="Variable",
        values="dur_sec",
        aggfunc="first",
        fill_value=0.0
    )

    piv_e.columns = [f"events_{c}" for c in piv_e.columns]
    piv_d.columns = [f"dur_sec_{c}" for c in piv_d.columns]

    wide = pd.concat([piv_e, piv_d], axis=1).reset_index()
    for var_label in ROLLING_VARS.keys():
        c = f"dur_sec_{var_label}"
        if c in wide.columns:
            wide[f"dur_min_{var_label}"] = wide[c] / 60.0
    return wide


# ============================
# Plot: señal bruta + sombreado como TRACES + hover con dur/media
# ============================
def make_plot_raw_with_threshold_traces(
    df_pm: pd.DataFrame,
    events_pm: pd.DataFrame,
    var_label: str,
    thresholds_pct: List[int]
) -> go.Figure:
    raw_col, ylab = RAW_PLOT_MAP[var_label]

    fig = go.Figure()

    # rango y para sombrear (rectángulos)
    y = df_pm[raw_col].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_max == y_min:
            y_max = y_min + 1.0

    pad = 0.05 * (y_max - y_min)
    y0, y1 = y_min - pad, y_max + pad

    # serie bruta
    fig.add_trace(go.Scatter(
        x=df_pm["sec"],
        y=df_pm[raw_col],
        mode="lines",
        name=f"{ylab} (bruto)",
        hovertemplate="t=%{x:.1f}s<br>valor=%{y:.2f}<extra></extra>",
    ))

    # sombreado por umbral como trazas -> leyenda ON/OFF funciona
    palette = [
        "rgba(0, 123, 255, 0.20)",
        "rgba(40, 167, 69, 0.20)",
        "rgba(255, 193, 7, 0.20)",
        "rgba(220, 53, 69, 0.20)",
    ]
    thr_sorted = sorted(list(dict.fromkeys(thresholds_pct)))
    thr_to_color = {thr: palette[i % len(palette)] for i, thr in enumerate(thr_sorted)}

    if events_pm is not None and not events_pm.empty:
        evv = events_pm[events_pm["Variable"] == var_label].copy()
        if not evv.empty:
            for thr in thr_sorted:
                ev_thr = evv[evv["threshold_pct"] == thr]
                color = thr_to_color[thr]

                if ev_thr.empty:
                    fig.add_trace(go.Scatter(
                        x=[], y=[],
                        mode="lines",
                        fill="toself",
                        name=f"Eventos {thr}%",
                        line=dict(width=0),
                        fillcolor=color,
                        hoverinfo="skip",
                    ))
                    continue

                # Construimos polígonos y además un "hover" con duración/media por evento
                xs, ys, htxt = [], [], []
                for _, r in ev_thr.iterrows():
                    x0 = float(r["start_sec"])
                    x1 = float(r["end_sec"])

                    # polígono
                    xs.extend([x0, x1, x1, x0, x0, None])
                    ys.extend([y0, y0, y1, y1, y0, None])

                    # hover: repetimos texto para el tramo del polígono (lo más simple)
                    t = (
                        f"<b>Evento {thr}%</b><br>"
                        f"Inicio: {x0:.1f}s<br>"
                        f"Fin: {x1:.1f}s<br>"
                        f"Duración: {float(r['duration_sec']):.1f}s<br>"
                        f"Promedio (bruto): {float(r['mean_raw']):.2f}<br>"
                        f"Promedio (rolling): {float(r['mean_roll']):.2f}"
                    )
                    # asignamos hover al primer punto del polígono; el resto no importa mucho
                    htxt.extend([t, t, t, t, t, None])

                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    fill="toself",
                    name=f"Eventos {thr}%",
                    line=dict(width=0),
                    fillcolor=color,
                    hovertext=htxt,
                    hovertemplate="%{hovertext}<extra></extra>",
                ))

    fig.update_layout(
        xaxis_title="Tiempo (s)",
        yaxis_title=ylab,
        margin=dict(l=10, r=10, t=30, b=10),
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig


# ============================
# Session State init
# ============================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
    st.session_state.df_roll = None
    st.session_state.summary_long = None
    st.session_state.events_long = None
    st.session_state.thresholds_pct = None


# ============================
# UI
# ============================
st.set_page_config(page_title="SubMIP Tool", layout="wide")

# Sidebar: guía rápida + autor
with st.sidebar:
    st.markdown("### Guía rápida")
    st.markdown(
        """
1) Sube raw (uno o varios CSV) o activa “usar ejemplos”  
2) (Opcional) Descarga el **bulk MIPs**, rellénalo y súbelo  
3) (Opcional) Sube MIPs de referencia (XLSX/CSV)  
4) (Opcional) Activa edición manual por jugador (pisa todo)  
5) Elige umbrales (85% fijo + extras) y ejecuta  
6) Explora tablas y gráfico (la leyenda permite ocultar umbrales)
        """
    )
    st.markdown("---")
    st.caption("Autor: **Edu Caro**")

# Header con autor arriba a la derecha
h1, h2 = st.columns([0.78, 0.22])
with h1:
    st.title("Herramienta SubMIP (Streamlit) — 85% fijo + umbrales extra")
with h2:
    st.markdown(
        "<div style='text-align:right; padding-top: 18px; font-weight:600;'>Edu Caro</div>",
        unsafe_allow_html=True
    )

example_paths = list_example_files()

st.subheader("0) Archivos de ejemplo")
col1, col2 = st.columns([1, 2])
with col1:
    if example_paths:
        st.download_button(
            "Descargar TODOS los ejemplos (ZIP)",
            data=zip_examples_bytes(example_paths),
            file_name="examples_assets.zip",
            mime="application/zip",
            key="dl_zip_examples"
        )
    else:
        st.warning("No hay CSVs en assets/. Mete ahí tus ejemplos.")
with col2:
    st.markdown("Raws admitidos: 9/10/11 columnas (TAB / ; / ,). Se crea `filename` desde el nombre del fichero.")

use_examples = st.checkbox("Usar archivos de ejemplo (assets/) en lugar de subir", value=False, key="use_examples")

st.subheader("1) Subir raw + (opcional) MIPs")
raw_files = st.file_uploader(
    "Sube uno o varios CSV raw",
    type=["csv", "txt"],
    accept_multiple_files=True,
    disabled=use_examples,
    key="raw_uploader"
)
mip_file = st.file_uploader(
    "Opcional: MIPs de referencia (XLSX o CSV) con 'Player_displ' y columnas MIP_*",
    type=["xlsx", "xls", "csv"],
    key="mip_uploader"
)

st.subheader("2) Parámetros")
c1, c2 = st.columns([1, 1])
with c1:
    fs = st.number_input("Frecuencia (Hz)", min_value=1, max_value=50, value=10, step=1, key="fs")
    window_sec = st.number_input("Ventana rolling (s)", min_value=10, max_value=120, value=60, step=5, key="win")
    use_match_fallback = st.checkbox("Si falta MIP, usar MIP del partido (fallback)", value=True, key="fallback")
with c2:
    st.write("✅ El 85% se calcula siempre.")
    extras = st.multiselect(
        "Añade hasta 3 umbrales extra",
        options=list(range(70, 96)),
        default=[],
        key="thr_extras"
    )
    if len(extras) > 3:
        st.error("Has elegido más de 3 umbrales extra. Deja máximo 3.")
    thresholds_pct = sorted(list(set([85] + extras[:3])))

params = Params(fs=int(fs), window_sec=int(window_sec), thresholds_pct=thresholds_pct)

# Preview para jugadores (para bulk y tabla manual)
players_preview: List[str] = []
df_raw_preview: Optional[pd.DataFrame] = None


def _load_raw_preview() -> Optional[pd.DataFrame]:
    if use_examples:
        if not example_paths:
            return None
        return read_raw_files_from_assets(example_paths)
    else:
        if not raw_files:
            return None
        return read_raw_files_from_uploads(raw_files)


try:
    df_raw_preview = _load_raw_preview()
    if df_raw_preview is not None and not df_raw_preview.empty:
        players_preview = sorted(df_raw_preview["Player_displ"].astype(str).unique().tolist())
except Exception as e:
    st.warning(f"No se pudo previsualizar raw todavía: {e}")

# BULK
st.subheader("3) Bulk MIPs (descarga + subida)")
st.caption("Plantilla con una fila por jugador (Player_displ). Al subir el bulk, se aplica por jugador y pisa la referencia.")

mip_bulk_df: Optional[pd.DataFrame] = None
if players_preview:
    bulk_tpl = make_mip_bulk_template(players_preview)
    st.download_button(
        "Descargar plantilla Bulk MIPs (CSV)",
        data=bulk_tpl.to_csv(index=False).encode("utf-8"),
        file_name="mip_bulk_template.csv",
        mime="text/csv",
        key="dl_mip_bulk_template"
    )
else:
    st.info("Sube raw (o usa ejemplos) para poder generar la plantilla de bulk por jugador.")

mip_bulk_file = st.file_uploader(
    "Subir Bulk MIPs (CSV) (opcional)",
    type=["csv"],
    key="mip_bulk_uploader"
)
if mip_bulk_file is not None:
    try:
        mip_bulk_df = read_mip_bulk(mip_bulk_file)
        st.success("Bulk cargado ✅")
    except Exception as e:
        st.error(f"No pude leer el bulk: {e}")
        mip_bulk_df = None

# Manual por jugador (manteniendo tu mecánica)
st.subheader("4) MIPs manuales individualizados (tabla por jugador)")
st.caption("Lo manual sobrescribe referencia + bulk + fallback. Deja 0 o vacío para NO aplicar override.")
manual_enable = st.checkbox("Activar edición manual por jugador", value=False, key="manual_enable")

manual_mips_by_player: Optional[Dict[str, Dict[str, float]]] = None

if manual_enable:
    if not players_preview:
        st.warning("Sube raw (o activa 'usar ejemplos') para poder listar jugadores y editar MIPs.")
    else:
        mip_ref_preview = None
        if mip_file is not None:
            try:
                mip_ref_preview = read_mip_reference(mip_file)
            except Exception as e:
                st.warning(f"No pude leer el fichero de MIPs todavía: {e}")

        base = pd.DataFrame({"Player_displ": players_preview})
        for col in MIP_COLS.values():
            base[col] = np.nan

        # precargar con referencia si existe
        if mip_ref_preview is not None:
            cols_present = ["Player_displ"] + [c for c in MIP_COLS.values() if c in mip_ref_preview.columns]
            base = base.merge(mip_ref_preview[cols_present], on="Player_displ", how="left", suffixes=("", "_ref"))
            for col in MIP_COLS.values():
                ref_col = f"{col}_ref"
                if ref_col in base.columns:
                    base[col] = base[ref_col]
                    base.drop(columns=[ref_col], inplace=True)

        # precargar con bulk si existe (bulk pisa referencia)
        if mip_bulk_df is not None and not mip_bulk_df.empty:
            base = base.merge(mip_bulk_df, on="Player_displ", how="left", suffixes=("", "_bulk"))
            for col in MIP_COLS.values():
                bcol = f"{col}_bulk"
                if bcol in base.columns:
                    base[col] = base[bcol].combine_first(base[col])
                    base.drop(columns=[bcol], inplace=True)

        edited_mip_table = st.data_editor(
            base,
            use_container_width=True,
            num_rows="fixed",
            key="mip_table_editor"
        )
        manual_mips_by_player = build_manual_mip_dict_from_table(edited_mip_table)

st.subheader("5) Ejecutar cálculo")
run = st.button(
    f"Calcular SubMIP (umbral(es): {', '.join(map(str, thresholds_pct))}%)",
    type="primary",
    disabled=(len(extras) > 3),
    key="run_btn"
)

if run:
    try:
        with st.status("Procesando...", expanded=True) as status:
            st.write("Leyendo raw...")
            if use_examples:
                if not example_paths:
                    raise ValueError("No hay archivos en assets/.")
                df_raw = read_raw_files_from_assets(example_paths)
            else:
                if not raw_files:
                    raise ValueError("No has subido archivos raw.")
                df_raw = read_raw_files_from_uploads(raw_files)

            st.write("Calculando señales y variables...")
            df_vars = compute_signals_and_vars(df_raw, fs=params.fs)

            st.write("Calculando rolling mean...")
            df_roll = compute_rollings(df_vars, fs=params.fs, window_sec=params.window_sec)

            st.write("Leyendo MIPs referencia (si existe)...")
            mip_ref = read_mip_reference(mip_file) if mip_file is not None else None

            st.write("Detectando eventos (6 variables) para todos los umbrales...")
            summary_long, events_long = compute_submip_for_thresholds(
                df_roll=df_roll,
                params=params,
                mip_ref=mip_ref,
                mip_bulk=mip_bulk_df,
                manual_mips_by_player=manual_mips_by_player,
                use_match_fallback=use_match_fallback
            )

            # Persistir resultados para que NO desaparezcan al cambiar selects
            st.session_state.df_roll = df_roll
            st.session_state.summary_long = summary_long
            st.session_state.events_long = events_long
            st.session_state.thresholds_pct = thresholds_pct
            st.session_state.results_ready = True

            status.update(label="Listo ✅", state="complete")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()


# ============================
# Mostrar resultados SIEMPRE que existan
# ============================
if st.session_state.results_ready:
    df_roll = st.session_state.df_roll
    summary_long = st.session_state.summary_long
    events_long = st.session_state.events_long
    thresholds_pct_saved = st.session_state.thresholds_pct

    st.subheader("6) Resumen (long)")
    st.dataframe(summary_long, use_container_width=True)
    st.download_button(
        "Descargar resumen (long) CSV",
        data=summary_long.to_csv(index=False).encode("utf-8"),
        file_name="submip_summary_long.csv",
        mime="text/csv",
        key="dl_summary_long"
    )

    st.subheader("7) Resumen (wide)")
    summary_wide = summary_wide_from_long(summary_long)
    st.dataframe(summary_wide, use_container_width=True)
    st.download_button(
        "Descargar resumen (wide) CSV",
        data=summary_wide.to_csv(index=False).encode("utf-8"),
        file_name="submip_summary_wide.csv",
        mime="text/csv",
        key="dl_summary_wide"
    )

    st.subheader("8) Eventos (detalle)")
    if events_long is None or events_long.empty:
        st.info("No se detectaron eventos.")
    else:
        st.dataframe(events_long, use_container_width=True)
        st.download_button(
            "Descargar eventos CSV",
            data=events_long.to_csv(index=False).encode("utf-8"),
            file_name="submip_events.csv",
            mime="text/csv",
            key="dl_events"
        )

    st.subheader("9) Gráfico (señal bruta + sombreado por umbral)")

    matches = sorted(df_roll["filename"].unique().tolist())
    players = sorted(df_roll["Player_displ"].unique().tolist())
    vars_list = list(ROLLING_VARS.keys())

    g1, g2, g3 = st.columns([1, 1, 1])
    with g1:
        sel_match = st.selectbox("Archivo / partido", matches, key="plot_match")
    with g2:
        sel_player = st.selectbox("Jugador", players, key="plot_player")
    with g3:
        sel_var = st.selectbox("Variable", vars_list, key="plot_var", index=vars_list.index("AccDens"))

    df_pm = df_roll[(df_roll["filename"] == sel_match) & (df_roll["Player_displ"] == sel_player)].copy()

    if events_long is None or events_long.empty:
        ev_pm = pd.DataFrame()
    else:
        ev_pm = events_long[
            (events_long["filename"] == sel_match) &
            (events_long["Player_displ"] == sel_player)
        ].copy()

    if df_pm.empty:
        st.warning("No hay datos para esa selección.")
    else:
        fig = make_plot_raw_with_threshold_traces(
            df_pm=df_pm,
            events_pm=ev_pm,
            var_label=sel_var,
            thresholds_pct=thresholds_pct_saved
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla bonita/útil: eventos del jugador/partido/variable con dur + promedios
        st.caption("Eventos detectados para la selección actual (duración y promedio).")
        if ev_pm is None or ev_pm.empty:
            st.info("No hay eventos para esta selección.")
        else:
            ev_table = ev_pm[ev_pm["Variable"] == sel_var].copy()
            if ev_table.empty:
                st.info("No hay eventos para esta variable.")
            else:
                ev_table = ev_table.sort_values(["threshold_pct", "start_sec"]).reset_index(drop=True)
                ev_table["umbral_%"] = ev_table["threshold_pct"].astype(int)
                ev_table["inicio_s"] = ev_table["start_sec"].round(1)
                ev_table["fin_s"] = ev_table["end_sec"].round(1)
                ev_table["dur_s"] = ev_table["duration_sec"].round(1)
                ev_table["prom_bruto"] = ev_table["mean_raw"].round(2)
                ev_table["prom_rolling"] = ev_table["mean_roll"].round(2)
                show_cols = ["umbral_%", "inicio_s", "fin_s", "dur_s", "prom_bruto", "prom_rolling"]
                st.dataframe(ev_table[show_cols], use_container_width=True)

else:
    st.info("Calcula primero para ver tablas y gráfico.")
