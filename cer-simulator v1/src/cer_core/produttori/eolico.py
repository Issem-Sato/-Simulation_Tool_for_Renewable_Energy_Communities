"""cer_core.produttori.eolico

Calcolo della produzione eolica **oraria** a partire dal meteo di sessione.

Obiettivo
---------
Replicare, per l'eolico, il workflow già esistente per il fotovoltaico (PVGIS):

- definizione parametri in UI → persistenza in ``producers.json``
- calcolo produzione oraria → cache su disco per produttore
- curva oraria annuale e download

Source of truth
--------------
Il meteo di sessione (``data/sessions/<session>/meteo_hourly.csv``) è la *source of truth*.
Questo modulo **non** scarica meteo esterno: usa solo la serie oraria già salvata in sessione.

Pipeline tecnica (stile NREL SAM – wind)
---------------------------------------
1) Weather file → serie oraria velocità vento a quota di riferimento (es. 100 m)
2) Extrapolazione a quota mozzo (hub height) via shear:
   ``V_hub = V_ref * (H_hub / H_ref) ** alpha``
3) Power curve lookup + interpolazione lineare
4) Perdite globali → potenza netta
5) Aggregazione di N turbine (0..N) abilitate

Cache su disco (per producer)
-----------------------------
Directory: ``<session>/cache/producer_<id>/eolico/``

La cache è invalidata se:
- cambia il file ``meteo_hourly.csv`` (hash del contenuto)
- cambiano i parametri delle turbine (hash canonico JSON)
- cambia la versione dell'algoritmo
"""

from __future__ import annotations

# ==============================================================================
# Standard library
# ==============================================================================

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple

# ==============================================================================
# Third‑party
# ==============================================================================

import numpy as np
import pandas as pd

# ==============================================================================
# Project
# ==============================================================================

from cer_core.bilanciamento.fingerprint import sha256_file, sha256_json_canonical
from cer_core.produttori.produttori import producer_cache_dir


# ==============================================================================
# Wind turbine library (generic models)
# ==============================================================================

_LIBRARY_PATH = Path(__file__).with_name("wind_turbine_library.json")
_LIBRARY_CACHE: Optional[dict[str, Any]] = None
_LIBRARY_SHA256_CACHE: Optional[str] = None


def load_wind_turbine_library() -> dict[str, Any]:
    """Carica la libreria turbine dal JSON (cached in memoria).

    Returns
    -------
    dict
        Dict con chiavi:
        - version: str
        - models: list[dict]
    """
    global _LIBRARY_CACHE, _LIBRARY_SHA256_CACHE

    if _LIBRARY_CACHE is not None:
        return _LIBRARY_CACHE

    if not _LIBRARY_PATH.exists():
        raise FileNotFoundError(f"Libreria turbine non trovata: {_LIBRARY_PATH}")

    payload = json.loads(_LIBRARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "models" not in payload:
        raise RuntimeError("Libreria turbine: formato non valido (atteso oggetto con chiave 'models').")

    _LIBRARY_CACHE = payload
    _LIBRARY_SHA256_CACHE = sha256_file(_LIBRARY_PATH)
    return payload


def wind_turbine_library_sha256() -> str:
    """Hash della libreria turbine (per invalidare cache su aggiornamenti)."""
    global _LIBRARY_SHA256_CACHE
    if _LIBRARY_SHA256_CACHE is None:
        # questo imposta anche _LIBRARY_SHA256_CACHE
        load_wind_turbine_library()
    assert _LIBRARY_SHA256_CACHE is not None
    return _LIBRARY_SHA256_CACHE


def _resolve_power_curve_for_turbine(t: dict[str, Any]) -> Sequence[dict[str, Any]] | Sequence[Tuple[float, float]]:
    """Restituisce la power curve per una turbina.

    Priorità:
    1) power_curve_override (se presente)
    2) power_curve (legacy)
    3) model_id → libreria
    """
    if t.get("power_curve_override"):
        return t["power_curve_override"]
    if t.get("power_curve"):
        return t["power_curve"]

    model_id = t.get("model_id")
    if not model_id:
        raise ValueError("Turbina: manca 'model_id' e non è presente una curva esplicita.")

    lib = load_wind_turbine_library()
    models = lib.get("models") or []
    for m in models:
        if str(m.get("model_id")) == str(model_id):
            curve = m.get("power_curve") or []
            if not curve:
                raise RuntimeError(f"Modello '{model_id}': power_curve mancante nella libreria.")
            return curve

    raise KeyError(f"Modello turbina non trovato in libreria: {model_id!r}")
OutputUnit = Literal["kw", "kwh"]


@dataclass(frozen=True)
class WindCacheMeta:
    """Metadati scritti accanto al parquet di cache."""

    algo_version: str
    meteo_sha256: str
    library_sha256: str
    params_sha256: str
    output_unit: OutputUnit
    dt_hours: float
    n_turbines_enabled: int


def _median_step_hours(idx: pd.DatetimeIndex) -> float:
    """Durata media del passo temporale in ore (robusta)."""

    if idx is None or len(idx) < 2:
        return 1.0
    diffs = np.diff(idx.view("i8"))  # ns
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    median_ns = float(np.median(diffs))
    return max(1e-9, median_ns / 3_600_000_000_000.0)


def _read_meteo_hourly_csv(path: Path) -> pd.DataFrame:
    """Legge il meteo orario di sessione e normalizza indice temporale in UTC."""

    df = pd.read_csv(path)
    # Contratto del progetto: colonna timestamp. Best-effort su alternative.
    time_col = None
    for cand in ("timestamp", "time", "datetime", "date"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise RuntimeError(f"meteo_hourly.csv: colonna tempo non trovata. Colonne: {list(df.columns)}")

    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if t.isna().any():
        bad = df.loc[t.isna(), time_col].astype(str).head(1).tolist()
        raise RuntimeError(f"meteo_hourly.csv: timestamp non parseabile (esempio: {bad})")

    df = df.drop(columns=[time_col]).copy()
    df.index = pd.DatetimeIndex(t, name="time")
    df = df.sort_index()
    return df


def _normalize_power_curve(curve: Sequence[dict[str, Any]] | Sequence[Tuple[float, float]]):
    """Normalizza la curva di potenza in due array ordinati (v_ms, p_kw)."""

    pts: list[Tuple[float, float]] = []
    for it in curve:
        if isinstance(it, dict):
            v = float(it.get("v_ms"))
            p = float(it.get("p_kw"))
        else:
            v = float(it[0])
            p = float(it[1])
        if np.isfinite(v) and np.isfinite(p):
            pts.append((v, p))

    if len(pts) < 2:
        raise ValueError("power_curve deve contenere almeno 2 punti")

    pts.sort(key=lambda x: x[0])

    # Collassa duplicati su v_ms mantenendo l'ultimo valore
    v_list: list[float] = []
    p_list: list[float] = []
    for v, p in pts:
        if v_list and abs(v - v_list[-1]) < 1e-12:
            p_list[-1] = p
        else:
            v_list.append(v)
            p_list.append(p)

    v_arr = np.asarray(v_list, dtype=float)
    p_arr = np.asarray(p_list, dtype=float)

    if not np.all(np.diff(v_arr) > 0):
        raise ValueError("power_curve: v_ms deve essere strettamente crescente")

    return v_arr, p_arr


def _interp_power_kw(v_hub: np.ndarray, v_curve: np.ndarray, p_curve: np.ndarray) -> np.ndarray:
    """Lookup curva di potenza con interpolazione lineare e clamp a 0 fuori range."""

    vmin = float(v_curve[0])
    vmax = float(v_curve[-1])

    v_clip = np.clip(v_hub, vmin, vmax)
    p = np.interp(v_clip, v_curve, p_curve)

    # Fuori range → 0 (cut-in/cut-out gestibili inserendo punti, ma qui forziamo a 0)
    p = np.where((v_hub < vmin) | (v_hub > vmax) | ~np.isfinite(v_hub), 0.0, p)

    # Power non negativa per robustezza
    p = np.maximum(p, 0.0)
    return p


def _canonicalize_turbines_for_hash(turbines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rappresentazione stabile (JSON canonico) per hashing parametri.

    Nota: per le turbine basate su libreria, l'hash include `model_id` (non l'intera curva),
    così la cache cambia solo se:
    - cambia il meteo
    - cambiano i parametri / model_id
    - cambia la libreria (hash libreria incluso a parte nella cache key)
    """

    out: list[dict[str, Any]] = []
    for t in turbines:
        model_id = t.get("model_id")

        # Se l'utente ha un override esplicito, includi la curva normalizzata nell'hash.
        curve_override = t.get("power_curve_override") or t.get("power_curve")  # legacy compat
        curve_norm: list[dict[str, float]] | None = None
        if curve_override:
            try:
                v_arr, p_arr = _normalize_power_curve(curve_override)
                curve_norm = [{"v_ms": float(v), "p_kw": float(p)} for v, p in zip(v_arr.tolist(), p_arr.tolist())]
            except Exception:
                curve_norm = None

        out.append(
            {
                "id": str(t.get("id", "")),
                "name": str(t.get("name", "")),
                "enabled": bool(t.get("enabled", True)),
                "count": int(t.get("count", 1)),
                "model_id": str(model_id) if model_id is not None else "",
                "hub_height_m": float(t.get("hub_height_m", 100.0)),
                "ref_height_m": float(t.get("ref_height_m", 100.0)),
                "shear_alpha": float(t.get("shear_alpha", 0.14)),
                "loss_pct": float(t.get("loss_pct", 0.0)),
                "wind_speed_col": str(t.get("wind_speed_col", "wind_speed_100m")),
                # include curve only if override is present
                "power_curve_override": curve_norm,
            }
        )

    out.sort(key=lambda d: d.get("id", ""))
    return out


def _eolico_cache_dir(session_dir: str | Path, producer_id: int) -> Path:
    d = producer_cache_dir(session_dir, producer_id) / "eolico"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_eolico_cache_key(
    meteo_hourly_csv: str | Path,
    turbines: Sequence[dict[str, Any]],
    *,
    algo_version: str = "v2",
    output_unit: OutputUnit = "kwh",
) -> str:
    """Costruisce una key stabile basata su meteo+parametri+libreria+versione."""

    meteo_path = Path(meteo_hourly_csv)
    meteo_sha = sha256_file(meteo_path)

    lib_sha = wind_turbine_library_sha256()

    canon = _canonicalize_turbines_for_hash(list(turbines))
    params_sha = sha256_json_canonical({"turbines": canon, "output_unit": output_unit})

    key = sha256_json_canonical(
        {
            "algo_version": algo_version,
            "meteo": meteo_sha,
            "library": lib_sha,
            "params": params_sha,
        }
    )
    return key[:16]


def cache_eolico_save(
    session_dir: str | Path,
    producer_id: int,
    df: pd.DataFrame,
    key: str,
    *,
    meta: WindCacheMeta,
) -> Path:
    """Salva una serie eolica in cache (parquet) + meta JSON."""

    d = _eolico_cache_dir(session_dir, producer_id)
    p_parq = d / f"eolico_{key}.parquet"
    p_meta = d / f"eolico_{key}.json"

    df_to = df.copy()
    if df_to.index.name != "time":
        df_to.index.name = "time"
    df_to = df_to.reset_index()
    df_to["time"] = (
        pd.to_datetime(df_to["time"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    tmp = p_parq.with_suffix(p_parq.suffix + ".tmp")
    df_to.to_parquet(tmp, index=False)
    os.replace(tmp, p_parq)

    meta_payload = {
        "algo_version": meta.algo_version,
        "meteo_sha256": meta.meteo_sha256,
        "library_sha256": meta.library_sha256,
        "params_sha256": meta.params_sha256,
        "output_unit": meta.output_unit,
        "dt_hours": meta.dt_hours,
        "n_turbines_enabled": meta.n_turbines_enabled,
    }
    tmpj = p_meta.with_suffix(p_meta.suffix + ".tmp")
    tmpj.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmpj, p_meta)

    return p_parq


def cache_eolico_load(session_dir: str | Path, producer_id: int, key: str) -> Optional[pd.DataFrame]:
    """Carica una serie eolica dalla cache (parquet)."""

    d = _eolico_cache_dir(session_dir, producer_id)
    p = d / f"eolico_{key}.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)

    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.drop(columns=[c for c in ("time", "index") if c in df.columns])
        df.index = t
    elif "index" in df.columns:
        t = pd.to_datetime(df["index"], utc=True, errors="coerce")
        df = df.drop(columns=["index"])
        df.index = t
    else:
        try:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        except Exception:
            pass

    df.index.name = "time"
    return df.sort_index()


def compute_wind_hourly(
    df_meteo_hourly: pd.DataFrame,
    turbines: Sequence[dict[str, Any]],
    *,
    output_unit: OutputUnit = "kwh",
    include_disabled: bool = False,
) -> pd.DataFrame:
    """Calcola la produzione eolica su base oraria.

    Parameters
    ----------
    df_meteo_hourly:
        DataFrame indicizzato temporalmente (UTC) con colonne vento.
        Se non ha indice datetime, deve contenere una colonna ``timestamp``.
    turbines:
        Lista di turbine (0..N). Ogni turbina è un dict con chiavi tipiche:
        - enabled: bool
        - count: int
        - hub_height_m, ref_height_m, shear_alpha
        - wind_speed_col
        - power_curve: list[{v_ms,p_kw}, ...]
        - loss_pct
    output_unit:
        "kw"  → potenza netta (kW)
        "kwh" → energia per time step (kWh sul passo)

    Returns
    -------
    pandas.DataFrame
        Indicizzato UTC (``time``) con colonne: Turbine_<id> (kW o kWh) + ``Totale``.
    """

    df = df_meteo_hourly.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # tenta parsing da colonna timestamp
        if "timestamp" not in df.columns:
            raise RuntimeError("df_meteo_hourly deve avere DatetimeIndex o colonna 'timestamp'")
        t = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if t.isna().any():
            raise RuntimeError("df_meteo_hourly: timestamp non parseabile")
        df = df.drop(columns=["timestamp"]).copy()
        df.index = pd.DatetimeIndex(t, name="time")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "time"
    df = df.sort_index()

    dt_hours = _median_step_hours(df.index)

    enabled = [t for t in turbines if (include_disabled or bool(t.get("enabled", True)))]
    out_cols: dict[str, pd.Series] = {}

    for t in enabled:
        name = str(t.get("name") or f"Turbina_{t.get('id', '')}")
        tid = str(t.get("id") or name)
        colname = f"Turbine_{tid}"

        wind_col = str(t.get("wind_speed_col", "wind_speed_100m"))
        if wind_col not in df.columns:
            raise KeyError(f"Colonna vento '{wind_col}' non trovata nel meteo. Colonne: {list(df.columns)}")

        v_ref = pd.to_numeric(df[wind_col], errors="coerce").to_numpy(dtype=float)

        hub_h = float(t.get("hub_height_m", 100.0))
        ref_h = float(t.get("ref_height_m", 100.0))
        alpha = float(t.get("shear_alpha", 0.14))
        if ref_h <= 0 or hub_h <= 0:
            raise ValueError("hub_height_m e ref_height_m devono essere > 0")

        # shear
        v_hub = v_ref * (hub_h / ref_h) ** alpha

        # power curve
        v_curve, p_curve = _normalize_power_curve(_resolve_power_curve_for_turbine(t))
        p_gross_kw = _interp_power_kw(v_hub, v_curve, p_curve)

        # losses
        loss_pct = float(t.get("loss_pct", 0.0))
        p_net_kw = p_gross_kw * max(0.0, 1.0 - loss_pct / 100.0)

        # multiplicity
        count = int(t.get("count", 1))
        count = max(0, count)
        p_net_kw = p_net_kw * float(count)

        if output_unit == "kwh":
            y = p_net_kw * float(dt_hours)
        else:
            y = p_net_kw

        out_cols[colname] = pd.Series(y, index=df.index, name=colname)

    if not out_cols:
        # ritorna df vuoto ma con indice (utile per merge) e Totale=0
        out = pd.DataFrame(index=df.index)
        out["Totale"] = 0.0
        out.index.name = "time"
        return out

    out = pd.concat([s.to_frame() for s in out_cols.values()], axis=1).sort_index()
    out["Totale"] = out.sum(axis=1)
    out.index.name = "time"
    return out


def get_or_compute_eolico_hourly(
    session_dir: str | Path,
    producer_id: int,
    meteo_hourly_csv: str | Path,
    turbines: Sequence[dict[str, Any]],
    *,
    output_unit: OutputUnit = "kwh",
    algo_version: str = "v2",
    force: bool = False,
) -> tuple[pd.DataFrame, str, bool]:
    """Carica dalla cache o calcola e salva in cache.

    Returns
    -------
    df, key, from_cache
    """

    meteo_path = Path(meteo_hourly_csv)
    meteo_sha = sha256_file(meteo_path)
    lib_sha = wind_turbine_library_sha256()
    canon = _canonicalize_turbines_for_hash(list(turbines))
    params_sha = sha256_json_canonical({"turbines": canon, "output_unit": output_unit, "library_sha256": lib_sha})
    key = build_eolico_cache_key(meteo_path, turbines, algo_version=algo_version, output_unit=output_unit)

    if not force:
        df_cached = cache_eolico_load(session_dir, producer_id, key)
        if df_cached is not None:
            return df_cached, key, True

    df_meteo = _read_meteo_hourly_csv(meteo_path)
    df_out = compute_wind_hourly(df_meteo, turbines, output_unit=output_unit)
    dt_hours = _median_step_hours(df_out.index)
    n_enabled = sum(1 for t in turbines if bool(t.get("enabled", True)))
    meta = WindCacheMeta(
        algo_version=algo_version,
        meteo_sha256=meteo_sha,
        library_sha256=lib_sha,
        params_sha256=params_sha,
        output_unit=output_unit,
        dt_hours=float(dt_hours),
        n_turbines_enabled=int(n_enabled),
    )
    cache_eolico_save(session_dir, producer_id, df_out, key, meta=meta)
    return df_out, key, False
