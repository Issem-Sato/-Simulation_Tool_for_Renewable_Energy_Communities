"""cer_core.produttori.produttori

Questo modulo raccoglie le utilità *core* per la modellazione dei produttori FV (fotovoltaico)
utilizzate dalla UI Streamlit (pagina **Produttori**) e dai moduli geometrici correlati
(`laf_packing.py`, `roof_3d.py`).

L'obiettivo del modulo è fornire:

1) **Utility geografiche/proiezioni**
   - conversione automatica WGS84 (EPSG:4326) → UTM locale (metri) per misure robuste
   - calcolo area in m²

2) **Convenzioni di orientamento**
   - classi discrete (N, NE, E, ...)
   - conversione tra orientamento e angolo (con la convenzione del progetto: 0° = Sud)

3) **Supporto “satellite & maschere”**
   - download mosaico ESRI World_Imagery in una bbox
   - rendering di maschere PNG (id classe e RGB) a partire da poligoni e orientamenti

4) **Persistenza**
   - lettura/scrittura `producers.json` nella cartella di sessione

5) **PVGIS**
   - query alla API PVGIS (seriescalc) per ottenere la potenza oraria (W)
   - caching su disco per evitare chiamate ripetute

Contratti dati e invarianti
---------------------------
Geometrie
  - Le geometrie scambiate con la UI sono in **EPSG:4326** (lon/lat), in formato Shapely `Polygon`.
  - Ogni volta che serve una misura metrica (area, offset, packing, ecc.) si converte in **UTM locale**.

Orientamento e angoli
  - Convenzione del progetto (usata in UI e nei moduli 3D):
    - **0° = Sud**, 90° = Est, 180° = Nord, 270° = Ovest.
  - PVGIS usa una convenzione diversa per `aspect` (0° Sud, +90° Ovest, -90° Est).
    La conversione è gestita internamente in `pvgis_series_hourly`.

Tempo
  - Le serie orarie PVGIS vengono normalizzate in `DatetimeIndex` **UTC**.

Note di progetto
---------------
Questo file è volutamente “utility-oriented”: non contiene logiche Streamlit.
La UI (cer_app/pages/1_Producers.py) usa solo un sottoinsieme delle funzioni esportate.
Il modulo mantiene firme stabili per evitare regressioni.
"""

from __future__ import annotations

# ==============================================================================
# Standard library
# ==============================================================================

import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

# ==============================================================================
# Third‑party
# ==============================================================================

import pandas as pd
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

# ==============================================================================
# Typing helpers (documentazione)
# ==============================================================================


class ProducerPV(TypedDict, total=False):
    """Parametri FV globali a livello produttore.

    Questa struttura *non* è vincolante a runtime (è solo documentazione), ma aiuta a
    rendere esplicite le chiavi tipiche presenti in `producers.json`.
    """

    panel_w: float
    panel_h: float
    gap: float
    margin: float
    coverage_min_pct: float
    wpp: int
    system_loss: float
    tech: str
    selected_area_idxs: list[int]


class ProducerArea(TypedDict, total=False):
    """Area di tetto (GeoJSON + parametri geometrici) associata a un produttore."""

    geom: dict[str, Any]  # GeoJSON Polygon (EPSG:4326)
    orientation: str
    angle_deg: float
    tilt_deg: float

    n_panels: int
    kwp: float
    annual_kwh: Optional[float]

    # footprint pannelli (GeoJSON Polygons) [opzionale]
    panels_geojson: list[dict[str, Any]]


class Producer(TypedDict, total=False):
    """Record di un produttore come persistito in `producers.json`."""

    id: int
    name: str
    areas: list[ProducerArea]
    pv: ProducerPV
    annual_kwh: Optional[float]

# ==============================================================================
# Orientamenti e colori (shared con la UI)
# ==============================================================================

ORIENTATIONS: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Palette “fixed” per maschere e overlay.
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "N": (54, 75, 154),
    "NE": (74, 123, 183),
    "E": (110, 166, 205),
    "SE": (152, 202, 225),
    "S": (221, 221, 221),
    "SW": (253, 174, 107),
    "W": (244, 109, 67),
    "NW": (213, 62, 79),
}

# Class id (grayscale) per maschere.
ORIENT_TO_ID: dict[str, int] = {
    "N": 1,
    "NE": 2,
    "E": 3,
    "SE": 4,
    "S": 5,
    "SW": 6,
    "W": 7,
    "NW": 8,
}

# Centri delle classi per la quantizzazione angolare.
# Convenzione del progetto: 0=S, 90=E, 180=N, 270=W.
ORIENT_CENTERS: dict[str, float] = {
    "S": 0.0,
    "SE": 45.0,
    "E": 90.0,
    "NE": 135.0,
    "N": 180.0,
    "NW": 225.0,
    "W": 270.0,
    "SW": 315.0,
}


# ==============================================================================
# Geo / proiezioni
# ==============================================================================

def auto_utm_crs(lat: float, lon: float) -> int:
    """Calcola l'EPSG UTM più appropriato (WGS84) per una coordinata.

    Parameters
    ----------
    lat, lon:
        Coordinate WGS84 in gradi.

    Returns
    -------
    int
        Codice EPSG UTM:
        - 326xx per emisfero Nord
        - 327xx per emisfero Sud

    Notes
    -----
    La conversione in UTM è essenziale per misure in metri (area, offset, packing).
    """

    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return int(f"{326 if lat >= 0 else 327}{zone:02d}")


def _utm_transformers_for_lonlat(lon: float, lat: float):
    """Ritorna le trasformazioni (fwd, inv) WGS84↔UTM locale.

    Notes
    -----
    Si usa `always_xy=True` per imporre la convenzione (lon, lat) in input/output,
    evitando ambiguità tipiche di alcune librerie GIS che accettano (lat, lon).
    """

    utm_epsg = auto_utm_crs(lat, lon)
    fwd = Transformer.from_crs(4326, utm_epsg, always_xy=True).transform
    inv = Transformer.from_crs(utm_epsg, 4326, always_xy=True).transform
    return fwd, inv


def to_local(
    poly_ll: Polygon,
) -> tuple[
    Polygon,
    Callable[[float, float], tuple[float, float]],
    Callable[[float, float], tuple[float, float]],
]:
    """Proietta un poligono WGS84 (lon/lat) in UTM locale.

    Returns
    -------
    poly_utm, fwd, inv
        - `poly_utm`: shapely `Polygon` in metri
        - `fwd`: callable lon/lat → metri
        - `inv`: callable metri → lon/lat
    """

    rp = poly_ll.representative_point()
    lon, lat = float(rp.x), float(rp.y)
    fwd, inv = _utm_transformers_for_lonlat(lon, lat)
    return transform(fwd, poly_ll), fwd, inv


def area_m2(poly_ll: Polygon) -> float:
    """Area in m² di un poligono WGS84 (lon/lat), usando UTM locale."""

    poly_utm, *_ = to_local(poly_ll)
    return float(poly_utm.area)


# ==============================================================================
# Orientamento ↔ angolo (convenzione progetto)
# ==============================================================================

def angle_from_orientation(ori: str) -> float:
    """Ritorna l'angolo “reale” (0=S) associato alla classe di orientamento."""

    return float(ORIENT_CENTERS.get(ori, 0.0))


def angle_deg_from_line_lonlat(coords: list[tuple[float, float]]) -> float:
    """Stima l'angolo “reale” (0=S, 90=E, 180=N, 270=W) da una polyline disegnata in mappa.

    L'utente disegna tipicamente una freccia sulla mappa (WGS84). Per evitare distorsioni
    della metrica in gradi, l'angolo viene calcolato proiettando in UTM locale.

    Parameters
    ----------
    coords:
        Lista di punti (lon, lat). È sufficiente il primo e l'ultimo.
    """

    if len(coords) < 2:
        return 0.0

    (x1, y1), (x2, y2) = coords[0], coords[-1]
    lon_c, lat_c = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    fwd, _ = _utm_transformers_for_lonlat(lon_c, lat_c)

    X1, Y1 = fwd(x1, y1)
    X2, Y2 = fwd(x2, y2)
    dx, dy = (X2 - X1), (Y2 - Y1)

    # 0° = Est (matematica), +CCW
    theta_from_east = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    # Conversione alla convenzione progetto: 0° = Sud
    return (theta_from_east + 90.0) % 360.0


def orientation_from_angle(angle_deg: float) -> str:
    """Quantizza un angolo (0=S) nella classe di orientamento più vicina."""

    def _circ_dist(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    best = "S"
    best_d = 1e9
    for k, c in ORIENT_CENTERS.items():
        d = _circ_dist(float(angle_deg), float(c))
        if d < best_d:
            best, best_d = k, d
    return best


# ==============================================================================
# Satellite & maschere (ESRI World Imagery)
# ==============================================================================

ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def lonlat_to_tilexy(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Converte (lon,lat) in coordinate tile XY (WebMercator) per un dato zoom.

    Implementa la formula standard *Slippy Map* (WebMercator), restituisce coordinate
    frazionarie (non arrotondate) così che il chiamante possa gestire correttamente
    il cropping all'interno della tile.
    """

    n = 2**z
    xtile = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def fetch_satellite_image(
    bbox: tuple[float, float, float, float],
    zoom: int,
    max_tiles: int = 900,
    min_zoom: int = 15,
) -> tuple[Image.Image, tuple[float, float, float, float], int]:
    """Scarica un mosaico ESRI World_Imagery e lo ritaglia sulla bbox richiesta.

    Parameters
    ----------
    bbox:
        (min_lon, min_lat, max_lon, max_lat) in WGS84.
    zoom:
        Zoom richiesto. Se la bbox richiede troppe tile (> max_tiles), lo zoom viene abbassato.
    max_tiles:
        Guardrail per evitare download eccessivi (RAM/tempo).
    min_zoom:
        Limite inferiore di zoom: sotto questo livello si preferisce fallire o usare output grossolano.

    Returns
    -------
    image, bbox_eff, used_zoom
        - image: PIL.Image RGB
        - bbox_eff: bbox usata per il cropping (in pratica coincide con bbox input)
        - used_zoom: zoom effettivamente utilizzato

    Raises
    ------
    MemoryError
        Se la bbox è troppo ampia anche a zoom basso.
    """

    def _tiles_span(b: tuple[float, float, float, float], z: int):
        min_lon, min_lat, max_lon, max_lat = b
        tx_min, ty_max = lonlat_to_tilexy(min_lon, min_lat, z)
        tx_max, ty_min = lonlat_to_tilexy(max_lon, max_lat, z)
        x0, x1 = int(math.floor(tx_min)), int(math.floor(tx_max))
        y0, y1 = int(math.floor(ty_min)), int(math.floor(ty_max))
        return (x1 - x0 + 1), (y1 - y0 + 1), x0, x1, y0, y1

    z = int(zoom)
    while True:
        tiles_x, tiles_y, x0, x1, y0, y1 = _tiles_span(bbox, z)
        if tiles_x * tiles_y <= max_tiles or z <= min_zoom:
            break
        z -= 1

    if tiles_x * tiles_y > max_tiles:
        raise MemoryError(
            f"Bounding box troppo ampia ({tiles_x}x{tiles_y} tiles) anche a zoom {z}. "
            "Riduci l'area o usa un bbox più piccolo."
        )

    canvas = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            url = ESRI_URL.format(z=z, x=x, y=y)
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                tile = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                # Tile mancante/errore: usa un placeholder neutro.
                tile = Image.new("RGB", (256, 256), (200, 200, 200))
            canvas.paste(tile, box=((x - x0) * 256, (y - y0) * 256))

    def _ll_to_px(lon: float, lat: float) -> tuple[float, float]:
        xt, yt = lonlat_to_tilexy(lon, lat, z)
        return (xt - x0) * 256, (yt - y0) * 256

    min_lon, min_lat, max_lon, max_lat = bbox
    px_min, _py_max = _ll_to_px(min_lon, min_lat)
    px_max, _py_min = _ll_to_px(max_lon, max_lat)
    _px_min2, py_min = _ll_to_px(min_lon, max_lat)
    _px_max2, py_max = _ll_to_px(max_lon, min_lat)

    left, top = int(px_min), int(py_min)
    right, bottom = int(px_max), int(py_max)
    cropped = canvas.crop((left, top, right, bottom))
    return cropped, bbox, z


def render_mask_from_polys(
    polys_wgs84: list[Polygon],
    orients: list[str],
    bbox: tuple[float, float, float, float],
    size_wh: tuple[int, int],
) -> tuple[bytes, bytes]:
    """Rende due maschere PNG a partire da poligoni e orientamenti.

    Output
    ------
    mask_ids (L):
        PNG in scala di grigi con valori {0..8} (0=background, 1..8 = ORIENT_TO_ID).
    mask_rgb (RGB):
        PNG a colori (COLOR_MAP) per visualizzazione.
    """

    W, H = size_wh
    mask_ids = Image.new("L", (W, H), 0)
    mask_rgb = Image.new("RGB", (W, H), (0, 0, 0))
    draw_ids = ImageDraw.Draw(mask_ids)
    draw_rgb = ImageDraw.Draw(mask_rgb)

    min_lon, min_lat, max_lon, max_lat = bbox

    def ll_to_xy(lon: float, lat: float) -> tuple[int, int]:
        # mapping affino bbox→pixel (attenzione: y cresce verso il basso)
        x = int((lon - min_lon) / (max_lon - min_lon + 1e-9) * (W - 1))
        y = int((1 - (lat - min_lat) / (max_lat - min_lat + 1e-9)) * (H - 1))
        return x, y

    for poly, ori in zip(polys_wgs84, orients):
        pts = [ll_to_xy(x, y) for (x, y) in poly.exterior.coords]
        cls = ORIENT_TO_ID.get(ori, 0)
        col = COLOR_MAP.get(ori, (200, 200, 200))
        draw_ids.polygon(pts, fill=cls, outline=cls)
        draw_rgb.polygon(pts, fill=col, outline=(20, 20, 20))

    b1 = io.BytesIO()
    b2 = io.BytesIO()
    mask_ids.save(b1, format="PNG")
    mask_rgb.save(b2, format="PNG")
    return b1.getvalue(), b2.getvalue()


# ==============================================================================
# Persistenza produttori (session-scoped)
# ==============================================================================

def _producers_json_path(session_dir: str | Path) -> Path:
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "producers.json"


def load_producers(session_dir: str | Path) -> list[dict[str, Any]]:
    """Carica `producers.json` dalla directory di sessione.

    Ritorna una lista (potenzialmente vuota). In caso di file corrotto,
    ritorna lista vuota per non bloccare la UI.
    """

    p = _producers_json_path(session_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_producers(session_dir: str | Path, producers: list[dict[str, Any]]) -> None:
    """Scrive `producers.json` in modo atomico (tmp + replace)."""

    p = _producers_json_path(session_dir)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(producers, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ==============================================================================
# Cache per produttore (session-scoped)
# ==============================================================================

def producer_cache_dir(session_dir: str | Path, producer_id: int) -> Path:
    """Directory cache per produttore: `<session>/cache/producer_<id>/`."""

    d = Path(session_dir) / "cache" / f"producer_{producer_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


# ==============================================================================
# PVGIS (seriescalc) + cache
# ==============================================================================

PVGIS_BASE_V52 = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
PVGIS_BASE_V53 = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"


def _pvgis_base_for_db(raddb: str) -> str:
    """PVGIS-SARAH3 richiede v5_3; gli altri database sono compatibili con v5_2."""

    if raddb and "SARAH3" in raddb.upper():
        return PVGIS_BASE_V53
    return PVGIS_BASE_V52


def pvgis_aspect_from_project_azimuth(azimuth_deg: float) -> float:
    """Converte l'azimut interno (convenzione progetto) nel parametro `aspect` di PVGIS.

    Convenzione interna (UI / simulatore)
    ------------------------------------
    - 0° = Sud
    - 90° = Est
    - 180° = Nord
    - 270° = Ovest

    Convenzione PVGIS (`aspect`)
    ----------------------------
    - 0° = Sud
    - valori positivi = rotazione verso Ovest
    - valori negativi = rotazione verso Est
    - dominio tipico: [-180°, 180°]

    Returns
    -------
    float
        `aspect` in gradi nel dominio [-180, 180].
    """

    aspect = (-(float(azimuth_deg)) + 360.0) % 360.0
    if aspect > 180.0:
        aspect -= 360.0
    return float(aspect)


def pvgis_series_hourly(
    lat: float,
    lon: float,
    tilt_deg: float,
    azimuth_deg: float,
    peak_kwp: float,
    loss_pct: float,
    start_year: int = 2022,
    end_year: int = 2023,
    raddb: str = "PVGIS-SARAH3",
    tech: str = "crystSi",
) -> pd.DataFrame:
    """Richiede a PVGIS la **potenza AC oraria** (W) per un impianto FV.

    Returns
    -------
    pandas.DataFrame
        Indicizzato con `DatetimeIndex` UTC e colonna unica:
        - `P`: potenza in Watt.

    Notes
    -----
    - La UI usa `azimuth_deg` con convenzione 0=S, 90=E, 180=N, 270=W.
    - PVGIS usa `aspect`: 0=S, +90=W, -90=E. La conversione è gestita qui.

    Raises
    ------
    requests.HTTPError
        In caso di errore HTTP PVGIS.
    RuntimeError / pandas.errors.ParserError
        Se la risposta non è nel formato atteso.
    """

    # PVGIS: aspect = 0 (S), +90 (W), -90 (E)
    aspect = pvgis_aspect_from_project_azimuth(azimuth_deg)

    params = {
        "lat": f"{float(lat):.6f}",
        "lon": f"{float(lon):.6f}",
        "raddatabase": raddb,
        "startyear": str(int(start_year)),
        "endyear": str(int(end_year)),
        "mountingplace": "building",
        "mounting": "fixed",
        "angle": f"{float(tilt_deg):.2f}",
        "aspect": f"{float(aspect):.2f}",
        "hourlyvalues": "1",
        "pvcalculation": "1",
        "outputformat": "json",
        "pvtechchoice": tech,
        "peakpower": f"{float(peak_kwp):.4f}",
        "loss": f"{float(loss_pct):.2f}",
    }

    base_url = _pvgis_base_for_db(raddb)
    r = requests.get(base_url, params=params, timeout=60)

    if r.status_code >= 400:
        # Fallback storico: se SARAH3 fallisce, prova SARAH2 con v5_2.
        if raddb.upper() == "PVGIS-SARAH3":
            params_fb = params.copy()
            params_fb["raddatabase"] = "PVGIS-SARAH2"
            r2 = requests.get(PVGIS_BASE_V52, params=params_fb, timeout=60)
            if r2.status_code >= 400:
                try:
                    msg1 = r.json().get("message", "")
                except Exception:
                    msg1 = r.text[:300]
                try:
                    msg2 = r2.json().get("message", "")
                except Exception:
                    msg2 = r2.text[:300]
                raise requests.HTTPError(
                    "PVGIS error. First try "
                    f"({base_url}/{raddb}): {r.status_code} {r.reason} – {msg1}. "
                    f"Fallback (v5_2/SARAH2): {r2.status_code} {r2.reason} – {msg2}",
                    response=r2,
                )
            data = r2.json()
        else:
            try:
                msg = r.json().get("message", "")
            except Exception:
                msg = r.text[:300]
            raise requests.HTTPError(f"{r.status_code} {r.reason} – {msg}", response=r)
    else:
        data = r.json()

    try:
        points = data["outputs"]["hourly"]
    except Exception as e:
        raise RuntimeError(f"Risposta PVGIS inattesa: {str(e)} | {data.get('message', '')}")

    df = pd.DataFrame(points)

    # --- tempo: cerca la colonna giusta e normalizza ---
    time_col: Optional[str] = None
    for cand in ("time", "time(UTC)", "timestamp", "Date"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise RuntimeError(f"Colonna tempo non trovata. Colonne disponibili: {list(df.columns)}")

    s = df[time_col].astype(str)
    ts = pd.to_datetime(s, utc=True, errors="coerce")

    # Formato PVGIS noto: 20220101:0010 → parse custom
    mask = ts.isna() & s.str.match(r"^\d{8}:\d{4}$")
    if mask.any():
        ts2 = pd.to_datetime(
            s[mask].str.replace(":", " ", n=1),
            format="%Y%m%d %H%M",
            utc=True,
            errors="coerce",
        )
        ts.loc[mask] = ts2

    # Altro formato: 20220101 00:10
    mask2 = ts.isna() & s.str.match(r"^\d{8}\s\d{2}:\d{2}$")
    if mask2.any():
        ts3 = pd.to_datetime(s[mask2], format="%Y%m%d %H:%M", utc=True, errors="coerce")
        ts.loc[mask2] = ts3

    if ts.isna().any():
        bad = s[ts.isna()].iloc[0]
        raise pd.errors.ParserError(f"PVGIS: formato data/ora non riconosciuto (esempio: {bad})")

    df = df.set_index(ts).drop(columns=[time_col]).sort_index()

    # --- potenza: trova la colonna P (tollerante a varianti) ---
    power_col: Optional[str] = None
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for c, lc in cols_lower.items():
        if lc == "p":
            power_col = c
            break
    if power_col is None:
        for c, lc in cols_lower.items():
            if lc.startswith("p(") or lc in ("p_ac", "pac", "pout", "power", "p(ac)"):
                power_col = c
                break
    if power_col is None:
        cand = [c for c in df.columns if c.strip().lower().startswith("p")]
        if cand:
            power_col = cand[0]
    if power_col is None:
        raise RuntimeError(f"Colonna 'P' non presente. Colonne disponibili: {list(df.columns)}")

    P = pd.to_numeric(df[power_col], errors="coerce")
    return pd.DataFrame({"P": P})


def pvgis_hourly_for_area(
    poly_ll: Polygon,
    tilt_deg: float,
    azimuth_deg: float,
    peak_kwp: float,
    loss_pct: float,
    start_year: int,
    end_year: int,
    raddb: str = "PVGIS-SARAH3",
    tech: str = "crystSi",
) -> pd.DataFrame:
    """Wrapper: chiama PVGIS sul *representative point* dell'area."""

    rp = poly_ll.representative_point()
    lon, lat = float(rp.x), float(rp.y)
    return pvgis_series_hourly(
        lat,
        lon,
        tilt_deg,
        azimuth_deg,
        peak_kwp,
        loss_pct,
        start_year,
        end_year,
        raddb=raddb,
        tech=tech,
    )


def _pvgis_cache_dir(session_dir: str | Path, producer_id: int) -> Path:
    d = producer_cache_dir(session_dir, producer_id) / "pvgis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_pvgis_save(session_dir: str | Path, producer_id: int, df: pd.DataFrame, key: str) -> Path:
    """Salva una serie PVGIS su disco (Parquet) con normalizzazione esplicita del tempo."""

    d = _pvgis_cache_dir(session_dir, producer_id)
    path = d / f"{key}.parquet"

    df_to = df.copy()
    if df_to.index.name != "time":
        df_to.index.name = "time"
    df_to = df_to.reset_index()
    df_to["time"] = (
        pd.to_datetime(df_to["time"], utc=True, errors="coerce")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    df_to.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def cache_pvgis_load(session_dir: str | Path, producer_id: int, key: str) -> pd.DataFrame | None:
    """Carica una serie PVGIS dalla cache (parquet/feather/csv)."""

    d = _pvgis_cache_dir(session_dir, producer_id)

    path: Optional[Path] = None
    for ext in (".parquet", ".feather", ".csv"):
        p = d / f"{key}{ext}"
        if p.exists():
            path = p
            break
    if path is None:
        return None

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".feather":
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    # Ripristina indice temporale
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.drop(columns=[c for c in ("time", "index") if c in df.columns])
        df.index = t
    elif "index" in df.columns:
        t = pd.to_datetime(df["index"], utc=True, errors="coerce")
        df = df.drop(columns=["index"])
        df.index = t
    else:
        # best effort
        try:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        except Exception:
            pass

    df.index.name = "time"
    return df
