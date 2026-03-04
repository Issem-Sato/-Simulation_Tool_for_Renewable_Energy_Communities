"""cer_app/app.py — Home (condizioni generali + meteo)

Questa pagina Streamlit è l'entry-point del simulatore CER.

Responsabilità (layer *UI/orchestrazione*):
1) Gestione sessione
   - selezione della sessione attiva (cartella sotto ``data/sessions/<session>/``)
   - creazione di una nuova sessione (sanitizzazione del nome tramite ``cer_app.session_paths``)

2) Metadati di simulazione (persistenza in ``session_meta.json``)
   - query testuale del comune (autocomplete)
   - *selezione esplicita* di un candidato (lat/lon) tramite Open-Meteo Geocoding API
   - intervallo temporale (start/end) e seed globale

3) Generazione dei dataset meteo canonici della sessione (source per le pagine successive)
   - ``meteo_hourly.csv``: serie oraria meteo (UTC) con temperatura e vento
   - ``meteo_daily.csv``: aggregazione giornaliera (mean/min/max) + classificazione ``condition``

Contratto dati (importante per la tesi):
- **Timebase master del progetto: UTC**.
- Le altre pagine (es. Consumatori) consumano i CSV meteo dalla sessione senza
  richiamare Open-Meteo e senza introdurre conversioni di fuso orario.

Dipendenze principali:
- ``cer_app.session_paths``: convenzioni di persistenza e selezione sessione
- ``cer_core.condizioni_generali.condizioni_climatiche``: integrazione Open‑Meteo
  (geocoding + Archive API + caching su disco)

Nota:
- Questa pagina NON esegue il bilanciamento energetico e NON calcola grandezze economiche.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict, NotRequired

import numpy as np
import pandas as pd
import streamlit as st

from cer_app.session_paths import (
    get_paths,
    get_current_session_name,
    list_sessions,
    set_current_session,
)
from cer_core.condizioni_generali.condizioni_climatiche import (
    OpenMeteoError,
    build_meteo_hourly_for_location_open_meteo,
    search_comuni_open_meteo,
)

# =============================================================================
# Streamlit config
# =============================================================================

st.set_page_config(page_title="CER - Home", page_icon="🏠", layout="wide")
st.title("CER — Home")

# =============================================================================
# Tipi (solo per chiarezza/documentazione; non vincolanti a runtime)
# =============================================================================


class LocationDict(TypedDict):
    """Location selezionata esplicitamente dall'utente (deterministica)."""

    name: str
    lat: float
    lon: float
    admin1: NotRequired[str | None]
    country_code: NotRequired[str | None]
    timezone: NotRequired[str | None]


class SessionMeta(TypedDict, total=False):
    """Metadati della sessione salvati in ``session_meta.json``."""

    comune_query: str
    comune: str  # label leggibile (nome comune scelto)
    start: str  # ISO date: YYYY-MM-DD
    end: str  # ISO date: YYYY-MM-DD
    seed: int
    location: LocationDict


# =============================================================================
# Costanti di progetto (assunzioni / scelte di modellazione)
# =============================================================================

# Classificazione giornaliera usata solo come *tag* informativo (non è un modello fisico).
DAILY_HOT_THRESHOLD_C = 25.0
DAILY_COLD_THRESHOLD_C = 5.0

# Cache dell'autocomplete (secondi)
GEOCODING_CACHE_TTL_S = 24 * 3600

# =============================================================================
# Paths sessione (convenzioni uniche del progetto)
# =============================================================================

PATHS = get_paths()
SESSION_DIR = PATHS.session_dir
META_PATH = PATHS.session_meta_json
METEO_HOURLY_PATH = PATHS.meteo_hourly_csv
METEO_DAILY_PATH = PATHS.meteo_daily_csv


# =============================================================================
# Utility: meta + meteo (I/O su disco)
# =============================================================================


def _default_meta() -> SessionMeta:
    """Metadati default (usati se ``session_meta.json`` non esiste o è invalido)."""
    return {
        "comune_query": "Roma",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "seed": 2025,
        # "location" verrà popolata dopo selezione del comune dall'elenco candidati
    }


def load_meta(meta_path: Path = META_PATH) -> SessionMeta:
    """Carica i metadati di sessione da JSON.

    Migrazione "soft" supportata:
    - se nei vecchi file esisteva solo la chiave ``comune``, viene copiata in ``comune_query``
      per precompilare l'autocomplete.

    In caso di errore/JSON malformato ritorna i default.
    """
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                if "comune_query" not in meta and "comune" in meta:
                    meta["comune_query"] = str(meta.get("comune") or "")
                return meta  # type: ignore[return-value]
        except Exception:
            # file corrotto o contenuto non-JSON: fallback ai default
            pass
    return _default_meta()


def save_meta(meta: SessionMeta, meta_path: Path = META_PATH) -> None:
    """Salva i metadati della sessione (source of truth per la Home)."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_daily_table(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Costruisce un DF giornaliero (UTC day boundary) a partire dal meteo orario.

    Contenuto:
    - temperatura: mean/min/max
    - vento (se presente):
        - velocità: mean/min/max
        - direzione: mean (media circolare in gradi)

    Nota: la media della direzione del vento è calcolata come *circular mean* per evitare
    errori su wrap-around (es. 350° e 10° non devono dare 180°).
    """

    df_idx = df_hourly.set_index("timestamp")

    # Temperatura (sempre presente)
    dailies = df_idx["temp"].resample("D").agg(["mean", "min", "max"])
    condition = np.where(
        dailies["mean"] > DAILY_HOT_THRESHOLD_C,
        "hot",
        np.where(dailies["mean"] < DAILY_COLD_THRESHOLD_C, "cold", "mild"),
    )

    df_daily = dailies.copy()

    # Velocità vento (opzionale)
    if "wind_speed_100m" in df_idx.columns:
        w = df_idx["wind_speed_100m"].resample("D").agg(["mean", "min", "max"])
        df_daily["wind_speed_mean"] = w["mean"]
        df_daily["wind_speed_min"] = w["min"]
        df_daily["wind_speed_max"] = w["max"]

    # Direzione vento (opzionale) — media circolare
    if "wind_direction_100m" in df_idx.columns:
        rad = np.deg2rad(df_idx["wind_direction_100m"].astype(float))
        sin_m = np.sin(rad).resample("D").mean()
        cos_m = np.cos(rad).resample("D").mean()
        mean_dir = (np.rad2deg(np.arctan2(sin_m, cos_m)) + 360.0) % 360.0
        df_daily["wind_direction_mean"] = mean_dir

    df_daily["condition"] = condition
    df_daily = df_daily.reset_index().rename(
        columns={"mean": "temp_mean", "min": "temp_min", "max": "temp_max"}
    )
    return df_daily


def _csv_columns(path: Path) -> list[str]:
    """Ritorna le colonne di un CSV senza leggerne i dati (best-effort)."""
    if not path.exists():
        return []
    try:
        # nrows=0 evita IO pesante su file grandi
        import pandas as _pd

        return list(_pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _csv_has_columns(path: Path, required: list[str]) -> bool:
    cols = _csv_columns(path)
    return all(c in cols for c in required)


def ensure_meteo(meta: SessionMeta, *, force: bool = False) -> bool:
    """Garantisce che i file meteo canonici esistano nella sessione corrente.

    Se i file sono già presenti e ``force=False``, la funzione non ricalcola nulla.

    Output su disco:
    - ``meteo_hourly.csv`` con colonne:
        - timestamp (UTC, tz-aware)
        - temp (°C)
        - wind_speed_100m (m/s)
        - wind_direction_100m (°)
    - ``meteo_daily.csv`` con colonne:
        - timestamp (UTC, inizio giorno)
        - temp_mean, temp_min, temp_max (°C)
        - wind_speed_mean, wind_speed_min, wind_speed_max (m/s) [se disponibili]
        - wind_direction_mean (°) [se disponibile]
        - condition ∈ {hot, mild, cold}

    Ritorna:
    - True se i file esistono (o sono stati rigenerati con successo)
    - False se mancano i prerequisiti (es. location non selezionata) o in caso di errore
    """
    if (not force) and METEO_HOURLY_PATH.exists() and METEO_DAILY_PATH.exists():
        # Migrazione automatica: se i CSV esistono ma sono legacy (senza vento), rigeneriamo.
        hourly_cols = _csv_columns(METEO_HOURLY_PATH)
        has_wind_hourly = ("wind_speed_100m" in hourly_cols) and ("wind_direction_100m" in hourly_cols)

        if not has_wind_hourly:
            # Forziamo rigenerazione sotto (download Open‑Meteo) così i dataset diventano la nuova source of truth.
            force = True
        else:
            # Se hourly è ok ma daily è legacy (senza aggregati vento), rigeneriamo solo il daily localmente.
            daily_cols = _csv_columns(METEO_DAILY_PATH)
            need_daily_wind = ("wind_speed_mean" not in daily_cols) or ("wind_direction_mean" not in daily_cols)
            if need_daily_wind:
                try:
                    dfh = pd.read_csv(METEO_HOURLY_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
                    dfh["timestamp"] = pd.to_datetime(dfh["timestamp"], utc=True)
                    dfd = _compute_daily_table(dfh)
                    dfd.to_csv(METEO_DAILY_PATH, index=False)
                except Exception:
                    # Best-effort: se fallisce, lasciamo i file esistenti
                    pass
            return True

    loc = meta.get("location")
    if not isinstance(loc, dict) or ("lat" not in loc) or ("lon" not in loc):
        st.error(
            "Per generare il meteo devi **selezionare un comune dall'elenco** (autocomplete) "
            "e poi premere **Salva sessione/meteo**."
        )
        return False

    # Validazione leggera del range date (evita richieste Open‑Meteo incoerenti)
    try:
        start_d = pd.to_datetime(str(meta["start"])).date()
        end_d = pd.to_datetime(str(meta["end"])).date()
    except Exception:
        st.error("Date di inizio/fine non valide (usa un formato tipo YYYY-MM-DD).")
        return False

    if start_d > end_d:
        st.error("Intervallo date non valido: la data di fine deve essere >= della data di inizio.")
        return False

    cache_dir = SESSION_DIR / "cache"

    try:
        df_core = build_meteo_hourly_for_location_open_meteo(
            latitude=float(loc["lat"]),
            longitude=float(loc["lon"]),
            start_date=str(start_d),
            end_date=str(end_d),
            cache_dir=cache_dir,
            timezone="UTC",
        )
    except (OpenMeteoError, ValueError) as e:
        # Robustezza: se NON stai forzando e hai un meteo precedente, lo mantieni
        if METEO_HOURLY_PATH.exists() and METEO_DAILY_PATH.exists() and (not force):
            st.warning("Open-Meteo non disponibile: uso il meteo già presente in sessione.")
            return True

        st.error(
            "Impossibile scaricare il meteo orario da Open‑Meteo. "
            "Verifica la selezione del comune o riprova più tardi.\n\n"
            f"Dettaglio: {e}"
        )
        return False

    # DF orario canonico (timestamp esplicito + sort)
    df_hourly = df_core.copy().reset_index().rename(columns={"index": "timestamp"})
    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)
    df_hourly = df_hourly.sort_values("timestamp").reset_index(drop=True)

    # Colonne: manteniamo un ordine stabile (utile per CSV e debugging)
    base_cols = ["timestamp", "temp"]
    wind_cols = [c for c in ["wind_speed_100m", "wind_direction_100m"] if c in df_hourly.columns]
    other_cols = [c for c in df_hourly.columns if c not in base_cols + wind_cols]
    df_hourly = df_hourly[base_cols + wind_cols + other_cols]

    # Scrittura su disco (CSV: semplice e interoperabile con Streamlit)
    df_hourly.to_csv(METEO_HOURLY_PATH, index=False)

    # Daily aggregation
    df_daily = _compute_daily_table(df_hourly)
    df_daily.to_csv(METEO_DAILY_PATH, index=False)

    return True


def load_meteo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica i CSV meteo canonici della sessione corrente."""
    dfh = pd.read_csv(METEO_HOURLY_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    dfd = pd.read_csv(METEO_DAILY_PATH, parse_dates=["timestamp"]).sort_values("timestamp")

    # Normalizzazione: garantiamo UTC tz-aware (evita bug silenziosi nelle pagine successive)
    dfh["timestamp"] = pd.to_datetime(dfh["timestamp"], utc=True)
    dfd["timestamp"] = pd.to_datetime(dfd["timestamp"], utc=True)
    return dfh, dfd


# =============================================================================
# Open‑Meteo Geocoding: cache (autocomplete)
# =============================================================================


@st.cache_data(show_spinner=False, ttl=GEOCODING_CACHE_TTL_S)
def cached_search_comuni(query: str) -> list[dict[str, Any]]:
    """Ricerca comuni con caching (ritorna tipi JSON-safe).

    Nota: Streamlit cache_data richiede oggetti serializzabili; per questo motivo
    convertiamo il dataclass ``OpenMeteoLocation`` in dict semplici.
    """
    out: list[dict[str, Any]] = []
    for loc in search_comuni_open_meteo(query, country_code="IT", language="it", count=10):
        out.append(
            {
                "name": loc.name,
                "admin1": loc.admin1,
                "country_code": loc.country_code,
                "lat": loc.latitude,
                "lon": loc.longitude,
                "timezone": loc.timezone,
            }
        )
    return out


# =============================================================================
# UI
# =============================================================================

meta = load_meta()

with st.sidebar:
    st.header("Sessione")

    # -------------------------------------------------------------------------
    # 1) Selezione / creazione sessione
    # -------------------------------------------------------------------------
    current_name = get_current_session_name()
    sessions = list_sessions()
    if current_name not in sessions:
        sessions = [current_name, *sessions]

    sel = st.selectbox("Sessione attiva", options=sessions, index=sessions.index(current_name))

    cols_sess = st.columns([1, 1])
    if cols_sess[0].button("🔁 Cambia", use_container_width=True):
        if sel != current_name:
            set_current_session(sel)
            # Evita che dati di sessione precedente rimangano in memoria in altre pagine.
            st.session_state.pop("producers", None)
            st.session_state.pop("consumers", None)
            st.session_state["_active_session_name"] = sel
            st.rerun()

    new_name = cols_sess[1].text_input(
        "Nuova",
        value="",
        placeholder="es. test_01",
        label_visibility="collapsed",
    )
    if st.button("➕ Crea nuova sessione", use_container_width=True):
        created_dir = set_current_session(new_name)
        st.session_state.pop("producers", None)
        st.session_state.pop("consumers", None)
        st.session_state["_active_session_name"] = created_dir.name
        st.success(f"Sessione creata e impostata: {created_dir.name}")
        st.rerun()

    st.write(f"Cartella sessione: **{SESSION_DIR.name}**")

    # -------------------------------------------------------------------------
    # 2) Selezione comune (autocomplete + scelta deterministica)
    # -------------------------------------------------------------------------
    comune_query = st.text_input(
        "Comune (scrivi per cercare)",
        value=str(meta.get("comune_query", "")),
    )

    candidates: list[dict[str, Any]] = []
    if len(comune_query.strip()) >= 2:
        try:
            candidates = cached_search_comuni(comune_query.strip())
        except OpenMeteoError as e:
            st.error(f"Geocoding Open‑Meteo fallito: {e}")

    if candidates:
        labels = [
            f"{c['name']} — {c.get('admin1') or ''} ({c.get('country_code') or '??'}) | "
            f"{float(c['lat']):.4f},{float(c['lon']):.4f}"
            for c in candidates
        ]

        # Se c'è una location già salvata, prova a pre-selezionarla (match su lat/lon).
        saved = meta.get("location") if isinstance(meta.get("location"), dict) else None
        default_idx = 0
        if saved:
            for i, c in enumerate(candidates):
                if (abs(float(c["lat"]) - float(saved.get("lat", 999))) < 1e-6) and (
                    abs(float(c["lon"]) - float(saved.get("lon", 999))) < 1e-6
                ):
                    default_idx = i
                    break

        sel_label = st.selectbox("Seleziona il comune corretto", options=labels, index=default_idx)
        sel_idx = labels.index(sel_label)
        selected_loc = candidates[sel_idx]

        st.caption(
            f"Selezione: {selected_loc['name']} ({selected_loc.get('admin1') or ''}) — "
            f"lat={selected_loc['lat']:.6f}, lon={selected_loc['lon']:.6f}"
        )
    else:
        st.selectbox("Seleziona il comune corretto", options=["(scrivi almeno 2 caratteri)"], disabled=True)
        selected_loc = None

    # -------------------------------------------------------------------------
    # 3) Periodo e seed
    # -------------------------------------------------------------------------
    col = st.columns(2)
    start = col[0].date_input("Inizio", pd.to_datetime(meta.get("start", "2025-01-01")).date())
    end = col[1].date_input("Fine", pd.to_datetime(meta.get("end", "2025-12-31")).date())
    seed = st.number_input("Seed simulazione", value=int(meta.get("seed", 2025)), step=1)

    if start > end:
        st.error("Intervallo date non valido: la data di fine deve essere >= della data di inizio.")

    # -------------------------------------------------------------------------
    # 4) Salvataggio + rigenerazione meteo (opzionale)
    # -------------------------------------------------------------------------
    if st.button("💾 Salva sessione/meteo"):
        if start > end:
            st.error("Correggi le date prima di salvare.")
        else:
            meta.update(
                {
                    "comune_query": comune_query.strip(),
                    "start": str(start),
                    "end": str(end),
                    "seed": int(seed),
                }
            )

            # Salva anche la location scelta (deterministica).
            if selected_loc is not None:
                meta["comune"] = selected_loc["name"]
                meta["location"] = {
                    "name": selected_loc["name"],
                    "admin1": selected_loc.get("admin1"),
                    "country_code": selected_loc.get("country_code"),
                    "lat": float(selected_loc["lat"]),
                    "lon": float(selected_loc["lon"]),
                    "timezone": selected_loc.get("timezone"),
                }
            else:
                # Senza selezione esplicita non possiamo garantire riproducibilità.
                meta.pop("location", None)

            save_meta(meta)

            ok = ensure_meteo(meta, force=True)
            if ok:
                st.success("Metadati salvati e meteo rigenerato per questa sessione.")
            else:
                st.error("Metadati salvati, ma meteo NON rigenerato (vedi errore sopra).")

# -------------------------------------------------------------------------
# Se manca il meteo (prima esecuzione), proviamo a generarlo; se manca la location, ci fermiamo.
# -------------------------------------------------------------------------
ok = ensure_meteo(meta)
if (not ok) or (not METEO_HOURLY_PATH.exists()) or (not METEO_DAILY_PATH.exists()):
    st.warning("Meteo non disponibile: seleziona un comune dall'elenco e salva la sessione.")
    st.stop()

df_hourly, df_daily = load_meteo()

# =============================================================================
# Output UI (riepilogo)
# =============================================================================

st.subheader("Riepilogo meteo")

c1, c2, c3 = st.columns(3)
c1.metric("Ore totali", f"{len(df_hourly):,}".replace(",", "."))
c2.metric("Periodo", f"{df_hourly['timestamp'].min().date()} → {df_hourly['timestamp'].max().date()}")
c3.metric("T media", f"{df_hourly['temp'].mean():.1f} °C")

# Grafico orario temperatura (come nella versione precedente):
st.markdown('**Temperatura oraria (°C)**')
st.line_chart(df_hourly.set_index("timestamp")["temp"])

# Preview tabelle:
st.markdown('**Meteo orario (prime 24 ore)**')
st.dataframe(df_hourly.head(24))

st.markdown('**Meteo giornaliero (prime 10 righe)**')
st.dataframe(df_daily.head(10))

st.info(
    "Le altre pagine (es. Consumatori) useranno **meteo_hourly.csv** e **meteo_daily.csv** "
    "in questa sessione senza ricalcolare."
)
