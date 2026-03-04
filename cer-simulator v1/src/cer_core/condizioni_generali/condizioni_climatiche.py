"""cer_core.condizioni_generali.condizioni_climatiche

Integrazione con Open‑Meteo (geocoding + serie meteo) per la generazione della
variabili meteo orarie (temperatura, vento) utilizzate dal simulatore CER.

Il modulo è collocato nel *core* perché fornisce dati di input condivisi da più
componenti (es. pagine Streamlit per i consumatori) ed incapsula:

1) Geocoding (autocomplete) per trasformare una query testuale ("comune") in una
   location deterministica (lat/lon).
2) Download della temperatura oraria da Open‑Meteo (Archive + Forecast).
3) Caching su disco (opzionale) per evitare chiamate ripetute e rendere la
   simulazione più reattiva.

API pubblica (funzioni "entry-point")
------------------------------------
- ``search_comuni_open_meteo``: ricerca candidati (lista) via Geocoding API.
- ``geocode_comune_open_meteo``: wrapper "legacy" che sceglie il primo candidato.
- ``build_meteo_hourly_for_location_open_meteo``: builder *source of truth* (multi-variabile, lat/lon già scelti).
- ``build_meteo_hourly_for_comune_open_meteo``: compatibilità (comune -> geocode -> builder).
- ``build_temperature_hourly_for_location_open_meteo``: compatibilità (solo temperatura).
- ``build_temperature_hourly_for_comune_open_meteo``: compatibilità (solo temperatura).

Contratto dati (fondamentale per la tesi)
----------------------------------------
Le funzioni che restituiscono una serie oraria (fetch/build) ritornano sempre un
``pandas.DataFrame`` con:

- **index**: ``DatetimeIndex`` tz-aware in **UTC**, con passo **1 ora**
- colonna **temp**: temperatura aria a 2 m (Open‑Meteo ``temperature_2m``) in **°C**
- colonna **wind_speed_100m**: velocità vento a 100 m (Open‑Meteo ``wind_speed_100m``) in **m/s**
- colonna **wind_direction_100m**: direzione vento a 100 m (Open‑Meteo ``wind_direction_100m``) in **°**

Nota: le colonne vento sono presenti quando richieste (nel builder multi-variabile sono incluse di default).

L'intervallo richiesto è *inclusivo* e copre:

- ``start_date`` alle **00:00 UTC**
- ``end_date`` alle **23:00 UTC**

Assunzione centrale: timebase master = UTC
-----------------------------------------
L'intero progetto usa **UTC** come fuso orario master per evitare ambiguità
(DST, offset locali, conversioni implicite tra pagine Streamlit).
Per questo motivo, se viene passato ``timezone != 'UTC'`` le funzioni sollevano
``ValueError``.

Strategia di acquisizione (Archive + Forecast)
---------------------------------------------
Open‑Meteo espone due endpoint principali per la temperatura oraria:

- **Archive API** (storico): ``https://archive-api.open-meteo.com/v1/archive``
- **Forecast API** (recente/futuro): ``https://api.open-meteo.com/v1/forecast``

Il builder ``build_temperature_hourly_for_location_open_meteo`` divide la
richiesta in due parti:

- parte storica → Archive fino a (oggi UTC − ``ARCHIVE_LAG_DAYS``)
- parte recente/futura → Forecast per la coda (tipicamente max ~16 giorni nel futuro)

Caching su disco
---------------
Se ``cache_dir`` è fornita, il builder salva:

- ``openmeteo_<id>.parquet``: il DataFrame orario
- ``openmeteo_<id>.json``: metadati (``cache_key``)

``<id>`` è calcolato come le prime 16 cifre dell'SHA256 del ``cache_key``
canonico (lat/lon arrotondati a 6 decimali + date + variabili richieste).

Failure modes
-------------
- ``OpenMeteoError`` in caso di:
  - HTTP error
  - payload inatteso (campi mancanti/disallineati)
  - serie incompleta sull'intervallo richiesto
- ``ValueError`` se ``timezone`` non è ``'UTC'``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests


# =============================================================================
# Costanti / endpoint
# =============================================================================

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# Nel pratico, l'Archive può non includere le ultimissime ore/giorni. Per evitare
# buchi, il builder sposta la "frontiera" sull'uso del Forecast indietro di
# qualche giorno.
ARCHIVE_LAG_DAYS = 2

# Limiti tipici della Forecast API (possono variare nel tempo; qui sono usati
# solo come guardrail per fornire messaggi d'errore più chiari).
FORECAST_MAX_DAYS_FWD = 16
FORECAST_MAX_DAYS_BACK = 92


# =============================================================================
# Errori e data model
# =============================================================================


class OpenMeteoError(RuntimeError):
    """Errore di acquisizione dati da Open‑Meteo (HTTP, payload o copertura)."""


@dataclass(frozen=True)
class OpenMeteoLocation:
    """Rappresenta una location restituita dal Geocoding API di Open‑Meteo."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    timezone: str | None = None
    country_code: str | None = None
    admin1: str | None = None


# =============================================================================
# Helper interni
# =============================================================================


def _require_utc(timezone: str) -> None:
    """Enforce della scelta progettuale: timebase master = UTC."""
    if (timezone or "").upper() != "UTC":
        raise ValueError(
            "Questa simulazione usa UTC come timebase master (timezone deve essere 'UTC')."
        )


def _sha256_hex(obj: Any) -> str:
    """SHA256 di un oggetto JSON-serializzabile (canonizzato con sort_keys)."""
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _request_json(
    url: str,
    *,
    params: dict[str, Any],
    timeout_s: float,
    err_prefix: str,
    check_payload_error: bool = True,
) -> dict[str, Any]:
    """Effettua una GET e ritorna il payload JSON.

    Solleva ``OpenMeteoError`` se:
    - status code != 200
    - payload contiene un flag di errore (``error: true``)
    """
    r = requests.get(url, params=params, timeout=timeout_s)
    if r.status_code != 200:
        raise OpenMeteoError(f"{err_prefix}: HTTP {r.status_code} - {r.text[:200]}")
    payload = r.json()
    # Open‑Meteo usa spesso {"error": true, "reason": "..."}. Per i dati meteo
    # (Archive/Forecast) trattiamo questo come errore. Per il geocoding, invece,
    # è utile mantenere un comportamento più permissivo (ritornare lista vuota).
    if check_payload_error and payload.get("error"):
        raise OpenMeteoError(f"{err_prefix}: {payload.get('reason')}")
    return payload



def _parse_hourly_payload(
    payload: dict[str, Any],
    *,
    var_map: dict[str, str],
    source: str,
) -> pd.DataFrame:
    """Estrae una o più variabili orarie da un payload Open‑Meteo.

    Parametri
    ---------
    payload:
        JSON di risposta Open‑Meteo (Archive o Forecast).
    var_map:
        Mapping ``{open_meteo_hourly_name -> output_column_name}``.
        Esempio: ``{"temperature_2m": "temp", "wind_speed_100m": "wind_speed_100m"}``
    source:
        Stringa usata nei messaggi d'errore (es. "Archive", "Forecast").

    Ritorna
    -------
    DataFrame con index ``DatetimeIndex`` tz-aware in UTC (step 1h) e colonne
    secondo ``var_map``.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise OpenMeteoError(f"Risposta {source} inattesa: hourly.time mancante o vuoto.")

    series: dict[str, Any] = {}
    for om_name, out_name in var_map.items():
        values = hourly.get(om_name)
        if values is None:
            raise OpenMeteoError(
                f"Risposta {source} inattesa: hourly.{om_name} mancante (richiesto)."
            )
        if len(values) != len(times):
            raise OpenMeteoError(
                f"Risposta {source} inattesa: hourly.time e hourly.{om_name} disallineati."
            )
        series[out_name] = pd.Series(values, dtype=float).values

    ts = pd.to_datetime(pd.Series(times), errors="raise")
    # Normalizzazione UTC (safety: alcune risposte potrebbero non riportare tz)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    df = pd.DataFrame(series, index=pd.DatetimeIndex(ts))
    df = df[~df.index.isna()].sort_index()
    return df


def _parse_hourly_temperature_payload(payload: dict[str, Any], *, source: str) -> pd.DataFrame:
    """Compatibilità: estrae (time, temperature_2m) da un payload Open‑Meteo.

    Ritorna un DataFrame con index UTC e colonna ``temp``.
    """
    return _parse_hourly_payload(payload, var_map={"temperature_2m": "temp"}, source=source)


# =============================================================================
# Geocoding
# =============================================================================


def search_comuni_open_meteo(
    query: str,
    *,
    country_code: str = "IT",
    language: str = "it",
    count: int = 10,
    timeout_s: float = 20.0,
) -> list[OpenMeteoLocation]:
    """Ricerca (autocomplete) tramite Geocoding API di Open‑Meteo.

    Parametri
    ---------
    query:
        Stringa di ricerca (min 2 caratteri). Tipicamente nome comune.
    country_code:
        Filtro ISO 3166‑1 alpha‑2 (default: IT).
    language:
        Lingua della risposta (default: it).
    count:
        Numero massimo di candidati (clippato a 1..20).
    timeout_s:
        Timeout HTTP in secondi.

    Ritorna
    -------
    Lista di ``OpenMeteoLocation``.

    Nota progettuale
    ---------------
    La funzione NON sceglie automaticamente un "migliore" candidato: la selezione
    deterministica è demandata alla UI (per riproducibilità della simulazione).
    """
    q = (query or "").strip()
    if len(q) < 2:
        return []

    params = {
        "name": q,
        "count": max(1, min(int(count), 20)),
        "language": (language or "").lower(),
        "format": "json",
        "countryCode": (country_code or "").upper(),
    }

    payload = _request_json(
        GEOCODING_API_URL,
        params=params,
        timeout_s=timeout_s,
        err_prefix="Geocoding fallito",
        check_payload_error=False,
    )

    results = payload.get("results") or []
    out: list[OpenMeteoLocation] = []
    for item in results:
        try:
            out.append(
                OpenMeteoLocation(
                    name=str(item.get("name") or q),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    elevation_m=float(item["elevation"]) if "elevation" in item else None,
                    timezone=str(item.get("timezone")) if item.get("timezone") else None,
                    country_code=str(item.get("country_code")) if item.get("country_code") else None,
                    admin1=str(item.get("admin1")) if item.get("admin1") else None,
                )
            )
        except Exception:
            # Se un record è malformato, lo ignoriamo.
            continue

    return out


def geocode_comune_open_meteo(
    comune: str,
    *,
    country_code: str = "IT",
    language: str = "it",
    timeout_s: float = 20.0,
) -> OpenMeteoLocation:
    """Geocoding deterministico "legacy": sceglie il primo risultato.

    Questa funzione è utile per compatibilità con codice storico, ma per una
    simulazione riproducibile è preferibile:
    - chiamare ``search_comuni_open_meteo``
    - far selezionare un candidato all'utente
    - salvare lat/lon nel ``session_meta.json`` (UI).
    """
    comune = (comune or "").strip()
    if not comune:
        raise OpenMeteoError("Comune vuoto.")

    results = search_comuni_open_meteo(
        comune,
        country_code=country_code,
        language=language,
        count=10,
        timeout_s=timeout_s,
    )
    if not results:
        raise OpenMeteoError(f"Nessun risultato geocoding per '{comune}' ({country_code}).")
    return results[0]


# =============================================================================
# Fetch: Archive e Forecast (temperatura oraria UTC)
# =============================================================================



# -----------------------------------------------------------------------------
# Fetch generico (Archive / Forecast) per più variabili orarie
# -----------------------------------------------------------------------------

# Mapping "standard" usato nel progetto (colonne di output).
DEFAULT_HOURLY_VAR_MAP: dict[str, str] = {
    "temperature_2m": "temp",
    "wind_speed_100m": "wind_speed_100m",
    "wind_direction_100m": "wind_direction_100m",
}


def fetch_open_meteo_hourly_utc(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
    timezone: str = "UTC",
    timeout_s: float = 30.0,
    wind_speed_unit: str = "ms",
    temperature_unit: str = "celsius",
) -> pd.DataFrame:
    """Archive API: serie oraria (UTC) per un intervallo di date e variabili.

    ``hourly_vars`` è una lista di variabili Open‑Meteo (es. ``["temperature_2m",
    "wind_speed_100m", "wind_direction_100m"]``).

    Ritorna un DataFrame con index UTC (step 1h) e colonne normalizzate secondo
    ``DEFAULT_HOURLY_VAR_MAP`` (o, se presenti variabili non note, con lo stesso
    nome Open‑Meteo).
    """
    _require_utc(timezone)

    hourly_vars = list(hourly_vars)
    if not hourly_vars:
        raise ValueError("hourly_vars non può essere vuoto.")

    # Costruzione mapping output colonne: per le variabili note usiamo nomi
    # stabili; altrimenti manteniamo il nome Open‑Meteo.
    var_map: dict[str, str] = {v: DEFAULT_HOURLY_VAR_MAP.get(v, v) for v in hourly_vars}

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "timeformat": "iso8601",
    }

    # Unità: le impostiamo solo se servono (per ridurre rischio di incompatibilità).
    if "temperature_2m" in hourly_vars:
        params["temperature_unit"] = temperature_unit
    if any(v.startswith("wind_speed_") for v in hourly_vars):
        params["wind_speed_unit"] = wind_speed_unit

    payload = _request_json(
        ARCHIVE_API_URL,
        params=params,
        timeout_s=timeout_s,
        err_prefix="Archive fallito",
    )

    df = _parse_hourly_payload(payload, var_map=var_map, source="Archive")

    # Validazione "soft": step orario regolare.
    if len(df.index) >= 2:
        deltas = df.index.to_series().diff().dropna().unique()
        if len(deltas) != 1 or deltas[0] != pd.Timedelta(hours=1):
            raise OpenMeteoError("Serie Archive non regolare (step != 1h).")

    return df


def fetch_open_meteo_forecast_hourly_utc(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
    timezone: str = "UTC",
    timeout_s: float = 30.0,
    wind_speed_unit: str = "ms",
    temperature_unit: str = "celsius",
) -> pd.DataFrame:
    """Forecast API: serie oraria (UTC) per variabili multiple.

    Usa ``past_days`` e ``forecast_days`` per coprire l'intervallo richiesto e
    poi effettua slice preciso sulle ore.
    """
    _require_utc(timezone)

    hourly_vars = list(hourly_vars)
    if not hourly_vars:
        raise ValueError("hourly_vars non può essere vuoto.")

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    today = pd.Timestamp.utcnow().date()

    # Limiti tipici: forecast_days ~16, past_days ~92
    days_back = min(FORECAST_MAX_DAYS_BACK, max(0, (today - start).days))
    days_fwd = min(FORECAST_MAX_DAYS_FWD, max(0, (end - today).days + 1))

    var_map: dict[str, str] = {v: DEFAULT_HOURLY_VAR_MAP.get(v, v) for v in hourly_vars}

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
        "timeformat": "iso8601",
        "past_days": days_back,
        "forecast_days": max(1, days_fwd),
    }

    if "temperature_2m" in hourly_vars:
        params["temperature_unit"] = temperature_unit
    if any(v.startswith("wind_speed_") for v in hourly_vars):
        params["wind_speed_unit"] = wind_speed_unit

    payload = _request_json(
        FORECAST_API_URL,
        params=params,
        timeout_s=timeout_s,
        err_prefix="Forecast fallito",
    )

    df = _parse_hourly_payload(payload, var_map=var_map, source="Forecast").sort_index()

    # Slice sull'intervallo richiesto (ore inclusive)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]


def fetch_open_meteo_temperature_hourly_utc(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Archive API: temperatura oraria (UTC) per un intervallo di date.

    Wrapper di compatibilità: usa ``fetch_open_meteo_hourly_utc`` con
    ``hourly_vars=["temperature_2m"]`` e ritorna una sola colonna ``temp``.
    """
    return fetch_open_meteo_hourly_utc(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_vars=["temperature_2m"],
        timezone=timezone,
        timeout_s=timeout_s,
    )


def fetch_open_meteo_forecast_temperature_hourly_utc(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    """Forecast API: temperatura oraria (UTC) per un intervallo di date.

    Wrapper di compatibilità: usa ``fetch_open_meteo_forecast_hourly_utc`` con
    ``hourly_vars=["temperature_2m"]``.
    """
    return fetch_open_meteo_forecast_hourly_utc(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_vars=["temperature_2m"],
        timezone=timezone,
        timeout_s=timeout_s,
    )


# =============================================================================
# Builder: serie oraria completa (Archive + Forecast) + caching
# =============================================================================



def build_meteo_hourly_for_location_open_meteo(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Entry-point (nuovo): genera meteo orario UTC per più variabili.

    Questo builder è pensato come *source of truth* meteo della simulazione:
    la stessa timebase alimenta consumatori e produttori (FV, eolico, ecc.).

    Default variabili
    -----------------
    Se ``hourly_vars`` è ``None`` vengono richieste:
    - ``temperature_2m`` → colonna ``temp`` (°C)
    - ``wind_speed_100m`` → colonna ``wind_speed_100m`` (m/s)
    - ``wind_direction_100m`` → colonna ``wind_direction_100m`` (°)

    Output
    ------
    DataFrame con index UTC (step 1h) e copertura completa
    ``start_date 00:00`` → ``end_date 23:00`` (inclusivo).

    Parametri
    ---------
    cache_dir:
        Se fornita, abilita caching su disco (parquet + json) per la specifica
        combinazione (lat/lon/date/variabili).
    """
    _require_utc(timezone)

    if hourly_vars is None:
        hourly_vars = ["temperature_2m", "wind_speed_100m", "wind_direction_100m"]
    hourly_vars = list(hourly_vars)

    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()

    # Round su lat/lon stabilizza la cache key evitando micro-variazioni.
    cache_key = {
        "provider": "open-meteo",
        "lat": round(float(latitude), 6),
        "lon": round(float(longitude), 6),
        "start_date": str(start_d),
        "end_date": str(end_d),
        "timezone": "UTC",
        "hourly": sorted(hourly_vars),
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }
    cache_id = _sha256_hex(cache_key)[:16]

    meta_path: Path | None = None
    data_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / f"openmeteo_{cache_id}.json"
        data_path = cache_dir / f"openmeteo_{cache_id}.parquet"
        if meta_path.exists() and data_path.exists():
            try:
                df_cached = pd.read_parquet(data_path)
                df_cached.index = pd.to_datetime(df_cached.index, utc=True)
                return df_cached.sort_index()
            except Exception:
                # Cache corrotta/non leggibile: rigenero.
                pass

    today = pd.Timestamp.utcnow().date()
    archive_cutoff = today - pd.Timedelta(days=ARCHIVE_LAG_DAYS)

    parts: list[pd.DataFrame] = []

    # Parte storica: Archive
    if start_d <= min(end_d, archive_cutoff):
        parts.append(
            fetch_open_meteo_hourly_utc(
                latitude=latitude,
                longitude=longitude,
                start_date=str(start_d),
                end_date=str(min(end_d, archive_cutoff)),
                hourly_vars=hourly_vars,
                timezone="UTC",
            )
        )

    # Coda recente/futura: Forecast
    if end_d > archive_cutoff:
        tail_start = max(start_d, (archive_cutoff + pd.Timedelta(days=1)))
        if (end_d - today).days > FORECAST_MAX_DAYS_FWD:
            raise OpenMeteoError(
                "Intervallo troppo nel futuro per Forecast API (max ~16 giorni). "
                "Riduci la data di fine."
            )
        parts.append(
            fetch_open_meteo_forecast_hourly_utc(
                latitude=latitude,
                longitude=longitude,
                start_date=str(tail_start),
                end_date=str(end_d),
                hourly_vars=hourly_vars,
                timezone="UTC",
            )
        )

    if not parts:
        raise OpenMeteoError("Intervallo non copribile da Open‑Meteo.")

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Reindex hard: garantisce TUTTE le ore richieste (e segnala eventuali buchi).
    start_ts = pd.Timestamp(start_d, tz="UTC")
    end_ts = pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(hours=23)
    expected_idx = pd.date_range(start_ts, end_ts, freq="1H", tz="UTC")
    df = df.reindex(expected_idx)

    # Verifica completezza su tutte le colonne richieste.
    missing_total = int(df.isna().sum().sum())
    if missing_total:
        # Conteggio per colonna per debug rapido.
        missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
        raise OpenMeteoError(
            "Meteo incompleto nell'intervallo richiesto. "
            f"Missing per colonna: {missing_by_col}"
        )

    df.attrs["open_meteo"] = {"cache_key": cache_key}

    if meta_path is not None and data_path is not None:
        try:
            meta_path.write_text(
                json.dumps(df.attrs["open_meteo"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            df.to_parquet(data_path)
        except Exception:
            # Il caching è best-effort: eventuali problemi di IO non devono bloccare la simulazione.
            pass

    return df


def build_meteo_hourly_for_comune_open_meteo(
    *,
    comune: str,
    start_date: str,
    end_date: str,
    hourly_vars: Optional[list[str]] = None,
    cache_dir: Optional[Path] = None,
    timezone: str = "UTC",
    language: str = "it",
    country_code: str = "IT",
) -> pd.DataFrame:
    """Wrapper: geocoding (primo risultato) + builder meteo multi-variabile."""
    loc = geocode_comune_open_meteo(comune, country_code=country_code, language=language)
    df = build_meteo_hourly_for_location_open_meteo(
        latitude=loc.latitude,
        longitude=loc.longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_vars=hourly_vars,
        cache_dir=cache_dir,
        timezone=timezone,
    )
    df.attrs["open_meteo"] = {
        "location": loc.__dict__,
        "cache_key": df.attrs.get("open_meteo", {}).get("cache_key"),
    }
    return df


def build_temperature_hourly_for_location_open_meteo(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Compatibilità: builder storico (solo temperatura).

    Internamente delega a ``build_meteo_hourly_for_location_open_meteo`` con
    ``hourly_vars=["temperature_2m"]`` per mantenere una sola implementazione.
    """
    df = build_meteo_hourly_for_location_open_meteo(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        hourly_vars=["temperature_2m"],
        cache_dir=cache_dir,
        timezone=timezone,
    )
    df_temp = df[["temp"]].copy()
    df_temp.attrs = dict(df.attrs)
    return df_temp


def build_temperature_hourly_for_comune_open_meteo(
    *,
    comune: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
    timezone: str = "UTC",
    language: str = "it",
    country_code: str = "IT",
) -> pd.DataFrame:
    """Wrapper di compatibilità: geocoding (primo risultato) + builder per coordinate.

    Per riproducibilità è preferibile usare direttamente:
    ``build_temperature_hourly_for_location_open_meteo`` con lat/lon selezionati.
    """
    loc = geocode_comune_open_meteo(comune, country_code=country_code, language=language)
    df = build_temperature_hourly_for_location_open_meteo(
        latitude=loc.latitude,
        longitude=loc.longitude,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        timezone=timezone,
    )
    df.attrs["open_meteo"] = {
        "location": loc.__dict__,
        "cache_key": df.attrs.get("open_meteo", {}).get("cache_key"),
    }
    return df
