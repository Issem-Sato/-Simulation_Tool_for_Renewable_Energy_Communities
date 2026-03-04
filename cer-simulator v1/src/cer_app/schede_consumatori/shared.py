from __future__ import annotations

"""
Shared helpers for the *Consumatori* Streamlit panels (UI layer).

Questo modulo appartiene a `cer_app/schede_consumatori` e centralizza funzioni
di utilità usate da più schede consumatori (carichi base, cucina, lavanderia,
occupancy, clima).

Obiettivo
---------
Dopo la refactorizzazione della pagina monolitica `2_Consumatori.py`, le schede
sono diventate moduli indipendenti; per evitare duplicazioni e dipendenze
circolari, le utility comuni vivono qui.

Responsabilità principali
-------------------------
1) Contratto timebase + meteo (session scope)
   - Legge `meteo_hourly.csv` generato da `cer_app/app.py`
   - Costruisce INDEX a 15 minuti (UTC)
   - Interpola temperatura oraria -> 15 minuti

2) Helper UI per finestre settimanali (slot matrix)
   - Conversione tra matrice booleana 7×N (UI) e dict (JSON)
   - Recupero SLOT_BANDS, SLOT_LABELS, DAY_NAMES dal core (lavanderia) se presenti

3) Persistenza consumatori e cache per-consumer
   - `consumers.json` (lista dict)
   - `cache/consumer_<id>/` (directory cache dei moduli)

4) Serializzazione
   - Filtraggio dict per dataclass
   - Normalizzazione di strutture (numpy/pandas) in tipi JSON-friendly

Invarianti
----------
- Tutti i timestamp vengono normalizzati a **UTC**.
- Il file canonico meteo è `<SESSION_DIR>/meteo_hourly.csv`.

Note Streamlit
--------------
Streamlit viene importato solo dentro le funzioni UI (es. start_matrix_editor)
per evitare dipendenze hard quando il modulo viene importato in contesti non-UI.
"""

import hashlib
import json
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Tuple as Tup, Any

import pandas as pd


# ---------------------------------------------------------------------
# Costanti: contratti file di sessione
# ---------------------------------------------------------------------
CANON_METEO_HOURLY_CSV = "meteo_hourly.csv"
CONSUMERS_JSON = "consumers.json"
CACHE_DIRNAME = "cache"


# ---------------------------------------------------------------------
# Slot matrix: bande, labels e conversioni (UI <-> JSON)
# ---------------------------------------------------------------------
def _default_slot_bands() -> List[Tup[int, int]]:
    """Return the canonical SLOT_BANDS used by the simulator.

    Storicamente le schede consumatori usavano `cer_core.consumatori.lavanderia.SLOT_BANDS`.
    Per mantenere le schede indipendenti dal vecchio file monolitico, recuperiamo qui
    i valori dal core quando disponibili; in caso contrario usiamo un fallback conservativo.
    """
    try:
        from cer_core.consumatori import lavanderia as L  # local import

        return [tuple(map(int, b)) for b in getattr(L, "SLOT_BANDS")]
    except Exception:
        # Fallback: 4 bande 6-24 in 4 blocchi (coerente con label UI)
        return [(6, 9), (9, 12), (12, 18), (18, 24)]


def matrix_to_dict(
    matrix_bools: List[List[bool]],
    slot_bands: Optional[List[Tup[int, int]]] = None,
) -> Dict[int, List[Tup[int, int]]]:
    """Convert a 7×N boolean matrix into the dict representation saved in JSON.

    Returns
    -------
    dict[int, list[tuple[int,int]]]
        Mapping: day_idx (0..6) -> list of (start_hour, end_hour)
    """
    bands = slot_bands or _default_slot_bands()
    out: Dict[int, List[Tup[int, int]]] = {}
    for i, row in enumerate(matrix_bools[:7]):
        out[i] = [bands[j] for j, chk in enumerate(row[: len(bands)]) if bool(chk)]
    return out


def dict_to_matrix(d: dict, slot_bands: Optional[List[Tup[int, int]]] = None) -> List[List[bool]]:
    """Convert stored JSON dict (day -> list of bands) to a 7×N boolean matrix."""
    bands = slot_bands or _default_slot_bands()
    m = [[False] * len(bands) for _ in range(7)]
    if not isinstance(d, dict):
        return m

    # Supporta chiavi int o string (JSON serializza chiavi dict come stringhe)
    for i in range(7):
        slots = d.get(i) or d.get(str(i)) or []
        for band in slots:
            try:
                h1, h2 = band
            except Exception:
                continue
            for j, b in enumerate(bands):
                if tuple(map(int, b)) == (int(h1), int(h2)):
                    m[i][j] = True
    return m


def count_weekly_slots_from_grid(
    grid: List[List[bool]], slot_bands: Optional[List[Tup[int, int]]] = None
) -> int:
    """Count selected weekly candidate slots in a 7×N boolean grid."""
    bands = slot_bands or _default_slot_bands()
    total = 0
    for day in range(min(7, len(grid))):
        row = grid[day]
        for j, chk in enumerate(row[: len(bands)]):
            if bool(chk):
                total += 1
    return total


def _default_slot_labels() -> List[str]:
    """Return canonical slot labels (UI header)."""
    try:
        from cer_core.consumatori import lavanderia as L

        return list(getattr(L, "SLOT_LABELS"))
    except Exception:
        return ["6-9", "9-12", "12-18", "18-24"]


def _default_day_names() -> List[str]:
    """Return canonical weekday labels (UI header)."""
    try:
        from cer_core.consumatori import lavanderia as L

        return list(getattr(L, "DAY_NAMES"))
    except Exception:
        return ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]


# Global esposti per compatibilità: alcune schede li referenziano direttamente
SLOT_LABELS: List[str] = _default_slot_labels()
DAY_NAMES: List[str] = _default_day_names()


def start_matrix_editor(key: str, init: dict | List[List[bool]] | None) -> List[List[bool]]:
    """Render a 7×N weekly slot editor and return a boolean matrix.

    Parametri
    ---------
    key:
        Prefisso per i widget Streamlit (evita collisioni).
    init:
        - dict (formato JSON day -> bands), oppure
        - matrice booleana 7×N, oppure
        - None.

    Returns
    -------
    list[list[bool]]
        Matrice 7×N (giorni × bande) con le selezioni utente.
    """
    import streamlit as st  # local import (no hard dependency at import time)

    slot_labels = _default_slot_labels()
    day_names = _default_day_names()

    # Manteniamo la compatibilità con la versione precedente:
    # appendiamo una colonna extra usata come checkbox "No".
    if isinstance(init, list):
        state = [list(map(bool, row[: len(slot_labels)])) + [False] for row in init[:7]]
    else:
        state = dict_to_matrix(init)
        state = [row + [False] for row in state]

    st.markdown("**Finestre di utilizzo (come nel sondaggio)**")
    cols = st.columns([1] + [1] * len(slot_labels) + [1])
    cols[0].markdown("**Giorno**")
    for j, lab in enumerate(slot_labels):
        cols[j + 1].markdown(f"**{lab}**")
    cols[-1].markdown("**No**")

    out: List[List[bool]] = []
    for i, day in enumerate(day_names[:7]):
        row: List[bool] = []
        rcols = st.columns([1] + [1] * len(slot_labels) + [1])
        rcols[0].write(day)

        for j in range(len(slot_labels)):
            row.append(rcols[j + 1].checkbox("", value=bool(state[i][j]), key=f"{key}_{i}_{j}"))

        none_chk = rcols[-1].checkbox("", value=bool(state[i][-1]), key=f"{key}_{i}_none")
        if none_chk:
            row = [False] * len(slot_labels)

        out.append(row)

    return out


# ---------------------------------------------------------------------
# Seed: persistenza sessione e derivazione deterministica
# ---------------------------------------------------------------------
def load_session_seed(session_dir: Path, default: int = 2025) -> int:
    """Load a persistent base seed for the current simulation session."""
    p = session_dir / "session_seed.txt"
    if p.exists():
        try:
            return int(p.read_text(encoding="utf-8").strip())
        except Exception:
            return default
    try:
        p.write_text(str(default), encoding="utf-8")
    except Exception:
        pass
    return default


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable seed from base_seed and arbitrary parts (32-bit)."""
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int(h.hexdigest()[:8], 16)


# ---------------------------------------------------------------------
# Timebase + meteo: INDEX 15-min + interpolazione temperatura
# ---------------------------------------------------------------------
def load_time_index_and_meteo(
    session_dir: Path,
) -> Tuple[pd.DatetimeIndex, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Load simulation index and meteo if present.

    Returns
    -------
    (INDEX_15min, t_air_daily, t_air_hourly, t_air_15min)

    - INDEX_15min: DateTimeIndex UTC a 15 minuti
    - t_air_hourly: serie oraria temperatura (UTC) se presente
    - t_air_15min: serie 15-min interpolata (UTC) se presente
    - t_air_daily: media giornaliera (UTC) se presente

    Fallback legacy
    --------------
    Se `meteo_hourly.csv` non esiste, prova a recuperare un indice leggendo
    un qualsiasi CSV già presente in cache. In quel caso l’indice ritornato
    può NON essere a 15 minuti (compatibilità con comportamento storico).
    """
    canon = session_dir / CANON_METEO_HOURLY_CSV
    if not canon.exists():
        cache_root = session_dir / CACHE_DIRNAME
        if cache_root.exists():
            for fp in cache_root.rglob("*.csv"):
                try:
                    df = pd.read_csv(fp, parse_dates=["timestamp"]).set_index("timestamp")
                    idx = pd.to_datetime(df.index, utc=True)
                    if len(idx) > 2:
                        idx15 = pd.DatetimeIndex(idx).sort_values()
                        return idx15, None, None, None
                except Exception:
                    continue
        return pd.DatetimeIndex([]), None, None, None

    dfm = pd.read_csv(canon, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    dfm = dfm.set_index("timestamp")
    dfm.index = pd.to_datetime(dfm.index, utc=True)

    # Supporto colonne legacy (robustezza)
    t_air_hourly = None
    for col in ["t_air", "temp_air", "temperature", "Tair", "T_air", "temp"]:
        if col in dfm.columns:
            t_air_hourly = dfm[col].astype(float)
            break

    t_air_daily = t_air_hourly.resample("1D").mean() if t_air_hourly is not None else None

    if len(dfm.index) == 0:
        return pd.DatetimeIndex([]), t_air_daily, t_air_hourly, None

    start = dfm.index.min()
    end = dfm.index.max()
    idx_15 = pd.date_range(start=start, end=end, freq="15min", tz="UTC")

    t_air_15 = None
    if t_air_hourly is not None:
        t_air_15 = (
            t_air_hourly.reindex(idx_15.union(t_air_hourly.index))
            .interpolate(method="time")
            .reindex(idx_15)
        )

    return pd.DatetimeIndex(idx_15), t_air_daily, t_air_hourly, t_air_15


# ---------------------------------------------------------------------
# Consumers registry + cache path helpers
# ---------------------------------------------------------------------
def load_consumers_json(session_dir: Path) -> list[dict]:
    p = session_dir / CONSUMERS_JSON
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_consumers_json(session_dir: Path, consumers: list[dict]) -> None:
    p = session_dir / CONSUMERS_JSON
    p.write_text(json.dumps(consumers, ensure_ascii=False, indent=2), encoding="utf-8")


def next_consumer_id(consumers: list[dict]) -> int:
    ids = [int(c.get("id")) for c in consumers if str(c.get("id", "")).isdigit()]
    return (max(ids) + 1) if ids else 1


def consumer_cache_dir(session_dir: Path, consumer_id: int) -> Path:
    d = session_dir / CACHE_DIRNAME / f"consumer_{int(consumer_id)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_device(consumer: dict, device: str, defaults: dict) -> dict:
    """Ensure a device config exists inside a consumer dict.

    Parameters
    ----------
    consumer:
        Dizionario consumatore (entry di consumers.json).
    device:
        Chiave del dispositivo dentro consumer["devices"].
    defaults:
        Default applicati se mancanti.

    Returns
    -------
    dict
        Config del device.
    """
    dev = (consumer.setdefault("devices", {})).setdefault(device, {})
    for k, v in (defaults or {}).items():
        dev.setdefault(k, v)
    return dev


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def prepare_curve_for_plot(curve: pd.Series, curve_view_mode: str, curve_view_month):
    """Prepare a curve for plotting based on view mode.

    - annuale: resample H mean (leggibilità)
    - mensile: filtro month mantenendo 15-min
    """
    if curve_view_mode == "annuale" or curve_view_month is None:
        return curve.resample("H").mean()
    year, month = curve_view_month
    mask_month = (curve.index.year == year) & (curve.index.month == month)
    return curve[mask_month]


# ---------------------------------------------------------------------
# Dataclass + JSON helpers
# ---------------------------------------------------------------------
def sanitize_for_dataclass(model_cls, data: dict) -> dict:
    """Filter a dict so it can be safely expanded into a dataclass."""
    names = {f.name for f in dc_fields(model_cls)}
    return {k: v for k, v in (data or {}).items() if k in names}


def normalize_nested(obj: Any):
    """Normalize nested structures to plain-Python types (JSON-friendly)."""
    try:
        import numpy as np  # optional
    except Exception:
        np = None

    import datetime as _dt

    if obj is None:
        return None

    if hasattr(obj, "to_pydatetime"):
        try:
            return obj.to_pydatetime()
        except Exception:
            pass

    if np is not None:
        try:
            if isinstance(obj, (np.generic,)):
                return obj.item()
        except Exception:
            pass

    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj

    if isinstance(obj, dict):
        return {str(k): normalize_nested(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [normalize_nested(v) for v in obj]

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    return obj
