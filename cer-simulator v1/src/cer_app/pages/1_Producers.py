"""cer_app.pages.1_Producers

Pagina Streamlit (orchestratore UI) per la definizione dei *produttori* di energia
all'interno della simulazione di Comunità Energetica Rinnovabile (CER).

Scopo della pagina
------------------
Questa pagina consente di:

1. **Gestire l'anagrafica dei produttori** (CRUD):
   - creazione/modifica/eliminazione di un produttore;
   - salvataggio su file `producers.json` nello scope della sessione.

2. **Definire le aree disponibili sui tetti** tramite disegno su mappa (Folium + Draw):
   - ogni *area* è un poligono in coordinate geografiche (lon/lat, EPSG:4326);
   - ad ogni area si associano:
     - **orientamento** (classe discreta: N, NE, E, SE, S, SW, W, NW);
     - **angolo reale** in gradi secondo la convenzione *0=S, 90=E, 180=N, 270=W*;
     - **pendenza (tilt)** in gradi.

3. **Posizionare i pannelli FV** (packing) sulle aree selezionate:
   - il packing avviene in un sistema di riferimento metrico locale (UTM) per lavorare in metri;
   - i pannelli sono salvati come footprint poligonali (`panels_geojson`) dentro `producers.json`
     per garantire riproducibilità e facilità di rendering.

4. **Stimare la produzione** con PVGIS:
   - per ogni area selezionata si richiede la serie oraria (potenza) a PVGIS;
   - i risultati sono **cachati su disco** per evitare richieste ripetute.

5. (Opzionale) **Creare una maschera raster** (satellite ESRI + overlay) per validazione visiva:
   - utile per reportistica e per controlli qualitativi sul disegno e sulle classi di orientamento.

File e persistenza (scope sessione)
-----------------------------------
La pagina lavora esclusivamente dentro la cartella della sessione corrente
`data/sessions/<session_name>/` e produce/usa:

- `producers.json`
    Lista di produttori e relative aree.
    È il *source of truth* per la configurazione lato produttori.

- `cache/producer_<id>/mask/` (opzionale)
    `satellite.png`, `mask_ids.png`, `mask_rgb.png`, `meta.json`.

- `cache/producer_<id>/pvgis/` (gestito dal core)
    Cache delle richieste PVGIS (serie orarie).

Data model (semplificato)
-------------------------
Un produttore ha la forma:

- Producer:
    - id: int
    - name: str
    - areas: list[Area]
    - pv: dict (parametri FV globali del produttore: dimensioni pannello, Wp, loss, tech, ...)
    - annual_kwh: float | None  (derivato da PVGIS su anno intero)

- Area:
    - geom: GeoJSON Polygon (lon/lat)
    - orientation: str  (classe discreta)
    - angle_deg: float  (0=S, 90=E, 180=N, 270=W)
    - tilt_deg: float
    - n_panels: int
    - kwp: float (kWp installati sull'area)
    - panels_geojson: list[GeoJSON Polygon]  (footprint pannelli, lon/lat) [opzionale]

Assunzioni e invarianti
-----------------------
- Geometrie in `producers.json` sono in coordinate geografiche EPSG:4326.
- Le misure in metri (packing, offset maschera) usano una UTM locale (EPSG 326xx/327xx).
- Convenzione angoli:
    - 0° = Sud
    - 90° = Est
    - 180° = Nord
    - 270° = Ovest
- PVGIS restituisce tipicamente la potenza `P` (W) su base oraria: l'energia (kWh) si ottiene
  integrando su Δt (ore) e dividendo per 1000.

Note architetturali (Streamlit)
-------------------------------
- La pagina usa `st.session_state` come memoria della *state machine* UI (list ↔ edit).
- Quando cambia la sessione (selezionata in Home), gli stati che dipendono dalla sessione
  vengono resettati per evitare contaminazioni tra configurazioni diverse.

Dipendenze principali
---------------------
- `cer_app.session_paths`: risoluzione path della sessione corrente.
- `cer_core.produttori.laf_packing`: algoritmo di packing nel Local Area Frame (LAF).
- `cer_core.produttori.roof_3d`: mesh 3D del tetto e pannelli per visualizzazione.
- `cer_core.produttori.produttori`: utilità (orientamenti, area m², PVGIS, maschere, caching).

"""

from __future__ import annotations

# ==============================================================================
# Imports
# ==============================================================================

# stdlib
import calendar
import io
import json
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, TypedDict

# third-party
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import Draw
from PIL import Image
from shapely.geometry import Polygon, shape, mapping
from streamlit_folium import st_folium

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover (plotly potrebbe non essere installato in alcuni ambienti)
    go = None

# project
from cer_app.session_paths import get_paths

from cer_core.produttori.roof_3d import (
    roof_mesh_from_areas,
    panels_mesh_from_area,
    local_transformers_for_areas,
)
from cer_core.produttori.laf_packing import pack_panels_laf_lonlat
from cer_core.produttori.produttori import (
    load_producers,
    save_producers,
    ORIENTATIONS,
    COLOR_MAP,
    area_m2,
    angle_from_orientation,
    angle_deg_from_line_lonlat,
    orientation_from_angle,
    fetch_satellite_image,
    render_mask_from_polys,
    producer_cache_dir,
    cache_pvgis_save,
    cache_pvgis_load,
    pvgis_hourly_for_area,
    auto_utm_crs,
)

from cer_core.produttori.eolico import (
    build_eolico_cache_key,
    cache_eolico_load,
    get_or_compute_eolico_hourly,
    load_wind_turbine_library,
)

# ==============================================================================
# Typing helpers (solo documentazione; non impattano il runtime)
# ==============================================================================

Orientation = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PvgisTech = Literal["crystSi", "CIS", "CdTe"]


class ProducerPV(TypedDict, total=False):
    """Parametri FV globali per un produttore."""

    panel_w: float
    panel_h: float
    gap: float
    margin: float
    coverage_min_pct: float
    wpp: int
    system_loss: float
    tech: PvgisTech
    selected_area_idxs: list[int]


class ProducerArea(TypedDict, total=False):
    """Area di tetto definita dall'utente (geometria + parametri)."""

    geom: dict[str, Any]  # GeoJSON Polygon (EPSG:4326)
    orientation: Orientation
    angle_deg: float  # 0=S, 90=E, 180=N, 270=W
    tilt_deg: float

    n_panels: int
    kwp: float
    annual_kwh: Optional[float]

    # footprint pannelli (GeoJSON Polygons)
    panels_geojson: list[dict[str, Any]]

    # (opzionale) parametri riproducibilità packing per-area
    panel_w: float
    panel_h: float
    gap: float
    margin: float
    coverage_min_pct: float
    wp_panel: int
    stagger: bool


class Producer(TypedDict, total=False):
    id: int
    name: str
    areas: list[ProducerArea]
    pv: ProducerPV
    wind: dict[str, Any]
    annual_kwh: Optional[float]


# ==============================================================================
# Costanti UI / PVGIS
# ==============================================================================

PVGIS_TECH_CODES: list[PvgisTech] = ["crystSi", "CIS", "CdTe"]
PVGIS_TECH_LABEL: dict[str, str] = {
    "crystSi": "Crystalline Silicon (original)",
    "CIS": "CIS",
    "CdTe": "CdTe",
}

# Chiavi di session_state che dipendono dalla sessione corrente (da resettare su switch).
_SESSION_BOUND_KEYS: Sequence[str] = (
    "producers",
    "mode",
    "editing_obj",
    "draw_buffer",
    "draw_fc",
    "map_view",
    "last_line_coords",
    "map_nonce",
    "mask_off_e",
    "mask_off_n",
    # widget keys che possono contenere selezioni non più valide dopo lo switch
    "edit_pick",
    "del_pick",
    "edit_area_pick",
    "sel_panels_3d",
    "btn_place_3d",
)

# ==============================================================================
# Utility: serializzazione pannelli (GeoJSON <-> shapely)
# ==============================================================================

def _panels_geo_to_geojson_list(panels_geo: Optional[Sequence[Polygon]]) -> list[dict[str, Any]]:
    """Serializza una lista di poligoni shapely in una lista di dict GeoJSON.

    Parametri
    ---------
    panels_geo:
        Sequenza di poligoni shapely (footprint dei pannelli) in EPSG:4326.

    Ritorna
    -------
    list[dict]:
        Lista di dizionari GeoJSON JSON-serializzabili.
    """
    try:
        return [mapping(pg) for pg in (panels_geo or [])]
    except Exception:
        return []


def _panels_geo_from_geojson_list(gj_list: Optional[Sequence[dict[str, Any]]]) -> list[Polygon]:
    """Deserializza una lista di dict GeoJSON in poligoni shapely."""
    out: list[Polygon] = []
    if not gj_list:
        return out
    for gj in gj_list:
        try:
            out.append(shape(gj))
        except Exception:
            continue
    return out


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    """Converte una tripla RGB in stringa hex (#RRGGBB) per folium."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _median_step_hours(index: pd.DatetimeIndex) -> float:
    """Stima Δt (ore) da un indice temporale, usando la mediana delle differenze."""
    if len(index) <= 1:
        return 1.0
    try:
        dt_s = index.to_series().diff().dropna().dt.total_seconds().median()
        return float((dt_s or 3600.0) / 3600.0)
    except Exception:
        return 1.0


# ==============================================================================
# Session / State init
# ==============================================================================

PATHS = get_paths()
SESSION_DIR: Path = PATHS.session_dir
META: Path = PATHS.session_meta_json


def _project_center_default(meta_path: Path) -> tuple[float, float]:
    """Centro mappa di default.

    - usa la location della sessione (se presente in session_meta.json)
    - altrimenti fallback su Roma (41.9028, 12.4964)
    """
    lat, lon = 41.9028, 12.4964
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            loc = meta.get("location")
            if isinstance(loc, dict) and "lat" in loc and "lon" in loc:
                lat = float(loc["lat"])
                lon = float(loc["lon"])
    except Exception:
        pass
    return (lat, lon)


def _reset_state_if_session_changed(session_dir: Path) -> None:
    """Reset degli stati pagina quando cambia la sessione attiva.

    Motivo:
    - `producers.json` e la cache sono *session-scoped*; se l'utente cambia sessione
      dalla Home, questa pagina non deve riutilizzare stati vecchi (poligoni disegnati,
      editing_obj, etc.).
    """
    active = st.session_state.get("_active_session_name")
    if active == session_dir.name:
        return

    st.session_state["_active_session_name"] = session_dir.name
    for k in _SESSION_BOUND_KEYS:
        st.session_state.pop(k, None)


def _init_session_state() -> None:
    """Inizializza le chiavi minime di session_state usate dalla pagina."""
    if "producers" not in st.session_state:
        st.session_state["producers"] = load_producers(SESSION_DIR)

    st.session_state.setdefault("mode", "list")  # "list" | "edit"
    st.session_state.setdefault("editing_obj", None)  # Producer in modifica

    # Mappa: persistiamo disegni e viewport per evitare che spariscano al rerun.
    st.session_state.setdefault("draw_buffer", [])  # solo poligoni (draft)
    st.session_state.setdefault("draw_fc", {"type": "FeatureCollection", "features": []})
    st.session_state.setdefault("map_view", {"center": _project_center_default(META), "zoom": 20})
    st.session_state.setdefault("last_line_coords", None)  # coords polyline (freccia)

    # nonce per forzare remount di st_folium (utile quando cambia produttore o entra in edit)
    st.session_state.setdefault("map_nonce", 0)

    # Default FV globali (comodi tra produttori; non sono session-scoped)
    st.session_state.setdefault("panel_w_global", 1.10)
    st.session_state.setdefault("panel_h_global", 1.80)
    st.session_state.setdefault("gap_global", 0.20)
    st.session_state.setdefault("margin_global", 0.05)
    st.session_state.setdefault("coverage_min_pct_global", 95.0)
    st.session_state.setdefault("wp_panel_global", 420)
    st.session_state.setdefault("system_loss_global", 14.0)
    st.session_state.setdefault("pv_tech_global", "crystSi")


def _reset_map_component(clear_buffer: bool = True) -> None:
    """Forza il remount del componente Folium/Draw.

    È utile perché Folium + streamlit_folium possono conservare uno stato interno
    non coerente quando si cambia produttore o si passa da list → edit.

    Parametri
    ---------
    clear_buffer:
        Se True, pulisce i poligoni draft e l'ultima freccia.
    """
    st.session_state["map_nonce"] += 1
    if clear_buffer:
        st.session_state["draw_buffer"] = []
        st.session_state["draw_fc"] = {"type": "FeatureCollection", "features": []}
        st.session_state["last_line_coords"] = None




def _clear_transient_visual_state() -> None:
    """Pulisce stati UI *effimeri* (non salvati su JSON).

    Esempi:
    - offset della maschera (Est/Nord), che ha senso ripartire da 0 quando si passa
      da un produttore all'altro.
    """
    for k in ("mask_off_e", "mask_off_n"):
        st.session_state.pop(k, None)

# ==============================================================================
# PV form helpers (salvataggio parametri FV nel JSON)
# ==============================================================================

def _ensure_pv_defaults_in_editing(editing_obj: Producer) -> ProducerPV:
    """Garantisce la presenza di `editing_obj['pv']` con tutti i default.

    Questo è importante per retro-compatibilità: sessioni/JSON creati con versioni
    precedenti potrebbero non contenere il nodo `pv` o potrebbero avere campi mancanti.
    """
    pv = editing_obj.setdefault("pv", {})  # type: ignore[assignment]

    pv.setdefault("panel_w", float(st.session_state.get("panel_w_global", 1.10)))
    pv.setdefault("panel_h", float(st.session_state.get("panel_h_global", 1.80)))
    pv.setdefault("gap", float(st.session_state.get("gap_global", 0.20)))
    pv.setdefault("margin", float(st.session_state.get("margin_global", 0.05)))
    pv.setdefault("coverage_min_pct", float(st.session_state.get("coverage_min_pct_global", 95.0)))
    pv.setdefault("wpp", int(st.session_state.get("wp_panel_global", 420)))
    pv.setdefault("system_loss", float(st.session_state.get("system_loss_global", 14.0)))
    pv.setdefault("tech", str(st.session_state.get("pv_tech_global", "crystSi")))

    # sanificazione tecnologia
    if pv.get("tech") not in PVGIS_TECH_CODES:
        pv["tech"] = "crystSi"  # type: ignore[assignment]

    return pv  # type: ignore[return-value]


def _init_pv_form_from_editing(editing_obj: Producer) -> None:
    """Allinea i widget (session_state) ai valori salvati nel produttore."""
    pv = _ensure_pv_defaults_in_editing(editing_obj)
    st.session_state["pv_panel_w"] = float(pv.get("panel_w", 1.10))
    st.session_state["pv_panel_h"] = float(pv.get("panel_h", 1.80))
    st.session_state["pv_gap"] = float(pv.get("gap", 0.20))
    st.session_state["pv_margin"] = float(pv.get("margin", 0.05))
    st.session_state["pv_coverage_min_pct"] = float(pv.get("coverage_min_pct", 95.0))
    st.session_state["pv_wpp"] = int(pv.get("wpp", 420))
    st.session_state["pv_system_loss"] = float(pv.get("system_loss", 14.0))
    st.session_state["pv_tech"] = str(pv.get("tech", "crystSi"))
    if st.session_state["pv_tech"] not in PVGIS_TECH_CODES:
        st.session_state["pv_tech"] = "crystSi"


def _persist_pv_from_form_into_editing(editing_obj: Producer, *, update_globals: bool = True) -> None:
    """Copia i parametri FV dai widget a `editing_obj['pv']`.

    Parametri
    ---------
    update_globals:
        Se True, aggiorna anche i default globali (così il prossimo produttore parte
        dagli ultimi valori inseriti).
    """
    pv = editing_obj.setdefault("pv", {})  # type: ignore[assignment]

    pv["panel_w"] = float(st.session_state.get("pv_panel_w", 1.10))
    pv["panel_h"] = float(st.session_state.get("pv_panel_h", 1.80))
    pv["gap"] = float(st.session_state.get("pv_gap", 0.20))
    pv["margin"] = float(st.session_state.get("pv_margin", 0.05))
    pv["coverage_min_pct"] = float(st.session_state.get("pv_coverage_min_pct", 95.0))
    pv["wpp"] = int(st.session_state.get("pv_wpp", 420))
    pv["system_loss"] = float(st.session_state.get("pv_system_loss", 14.0))
    pv["tech"] = str(st.session_state.get("pv_tech", "crystSi"))
    if pv["tech"] not in PVGIS_TECH_CODES:
        pv["tech"] = "crystSi"

    if update_globals:
        st.session_state["panel_w_global"] = pv["panel_w"]
        st.session_state["panel_h_global"] = pv["panel_h"]
        st.session_state["gap_global"] = pv["gap"]
        st.session_state["margin_global"] = pv["margin"]
        st.session_state["coverage_min_pct_global"] = pv["coverage_min_pct"]
        st.session_state["wp_panel_global"] = pv["wpp"]
        st.session_state["system_loss_global"] = pv["system_loss"]
        st.session_state["pv_tech_global"] = pv["tech"]


# ==============================================================================
# UI helpers
# ==============================================================================

def _producer_table_rows(producers: Sequence[Producer]) -> pd.DataFrame:
    """Costruisce una tabella riassuntiva per la lista produttori."""
    wp_panel_global = float(st.session_state.get("wp_panel_global", 0.0))
    rows: list[dict[str, Any]] = []

    for p in producers:
        pv_p: dict[str, Any] = p.get("pv", {}) or {}
        wp_panel = float(pv_p.get("wpp", wp_panel_global) or wp_panel_global)

        areas = p.get("areas", []) or []
        m2_tot = sum(area_m2(shape(a["geom"])) for a in areas) if areas else 0.0

        selected_idxs = list((pv_p.get("selected_area_idxs") or []))
        if not selected_idxs:
            selected_idxs = list(range(len(areas)))

        kwp_tot = 0.0
        for ii in selected_idxs:
            if ii < 0 or ii >= len(areas):
                continue
            a = areas[ii]
            n_pan = int(a.get("n_panels", 0) or 0)
            kwp_a = float(a.get("kwp", 0.0) or 0.0)
            if kwp_a <= 0 and wp_panel > 0:
                kwp_a = (n_pan * wp_panel) / 1000.0
            kwp_tot += kwp_a

        rows.append(
            {
                "ID": p["id"],
                "Nome": p.get("name", f"Produttore {p['id']}"),
                "Aree": len(areas),
                "m² aree": round(m2_tot, 1),
                "kWp installati": round(kwp_tot, 3),
                "kWh/anno": (
                    round(float(p.get("annual_kwh", 0.0) or 0.0), 1)
                    if (p.get("annual_kwh") is not None)
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def _center_from_areas(areas: Sequence[ProducerArea]) -> Optional[tuple[float, float]]:
    """Calcola un centro (lat, lon) approssimato dai vertici dei poligoni."""
    if not areas:
        return None
    try:
        coords = np.vstack([np.asarray(shape(a["geom"]).exterior.coords) for a in areas])
        lat = float(coords[:, 1].mean())
        lon = float(coords[:, 0].mean())
        return (lat, lon)
    except Exception:
        return None


def _enter_edit_mode_for_producer(prod: Producer) -> None:
    """Imposta session_state per entrare in modalità 'edit' su un produttore esistente."""
    _reset_map_component()
    _clear_transient_visual_state()
    st.session_state["mode"] = "edit"
    # deep copy via json per evitare mutazioni sulla lista prima del salvataggio
    st.session_state["editing_obj"] = json.loads(json.dumps(prod))
    _init_pv_form_from_editing(st.session_state["editing_obj"])

    areas = prod.get("areas", []) or []
    c = _center_from_areas(areas)
    if c:
        st.session_state["map_view"]["center"] = c
        st.session_state["map_view"]["zoom"] = 20


def _enter_edit_mode_new_producer(new_id: int) -> None:
    """Crea un nuovo produttore (in memoria) e passa alla modalità edit."""
    _reset_map_component()
    _clear_transient_visual_state()
    st.session_state["mode"] = "edit"
    st.session_state["editing_obj"] = {"id": new_id, "name": f"Produttore {new_id}", "areas": [], "pv": {}}
    _init_pv_form_from_editing(st.session_state["editing_obj"])
    st.session_state["map_view"]["center"] = _project_center_default(META)


def _require_editing() -> Producer:
    """Ritorna il produttore in editing oppure interrompe la pagina."""
    editing = st.session_state.get("editing_obj")
    if st.session_state.get("mode") != "edit" or editing is None:
        st.info("Seleziona un produttore o clicca **Aggiungi nuovo**.")
        st.stop()
    return editing


# ==============================================================================
# Sezione: Area editor (mappa + controlli)
# ==============================================================================

def _render_area_editor(editing: Producer) -> None:
    st.subheader("Area")

    # assicura pv defaults anche per produttori creati con versioni precedenti
    _ensure_pv_defaults_in_editing(editing)

    editing["name"] = st.text_input("Nome produttore", value=editing.get("name", ""))

    col_map, col_ctrl = st.columns([2.4, 1.2])

    # ----------------------
    # Mappa + draw tools
    # ----------------------
    with col_map:
        center = st.session_state["map_view"]["center"]
        zoom = st.session_state["map_view"]["zoom"]

        m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, max_zoom=24)
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                 'contributors &copy; <a href="https://carto.com/">CARTO</a>',
            name="Carto Light",
            subdomains="abcd",
            max_zoom=24,
        ).add_to(m)

        # Aree salvate nel produttore (colorate per orientamento)
        for idx, a in enumerate(editing.get("areas", []), start=1):
            try:
                g = shape(a["geom"])
                col = COLOR_MAP.get(a.get("orientation", "S"), (200, 200, 200))
                folium.GeoJson(
                    mapping(g),
                    name=f"Area #{idx}",
                    style_function=lambda _x, c=col: {
                        "color": _rgb_hex(c),
                        "weight": 2,
                        "fillColor": _rgb_hex(c),
                        "fillOpacity": 0.25,
                    },
                ).add_to(m)
            except Exception:
                # geometria corrotta → semplicemente non renderizziamo
                pass

        # Disegni draft (persistiti in draw_fc per non perderli al rerun)
        for f in (st.session_state.get("draw_fc", {}) or {}).get("features", []):
            try:
                g = f.get("geometry", {})
                if g and g.get("type") == "Polygon":
                    gb = shape(g)
                    folium.GeoJson(
                        mapping(gb),
                        name="(draft) disegno",
                        style_function=lambda _x: {
                            "color": "#444444",
                            "weight": 2,
                            "dashArray": "6,4",
                            "fillColor": "#888888",
                            "fillOpacity": 0.15,
                        },
                    ).add_to(m)
            except Exception:
                pass

        Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": True,   # usata come "freccia" per derivare angle_deg
                "rectangle": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polygon": True,    # area di tetto
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        folium.LayerControl(position="topleft", collapsed=False).add_to(m)

        map_state = st_folium(
            m,
            width=None,
            height=560,
            returned_objects=["last_active_drawing", "all_drawings", "center", "zoom"],
            key=f"draw_map_osm_editor_{editing['id']}_{st.session_state['map_nonce']}",
        )

        # Persist viewport (così non "salta" ad ogni rerun)
        if map_state:
            c = map_state.get("center")
            z = map_state.get("zoom")
            if c:
                st.session_state["map_view"]["center"] = (c["lat"], c["lng"])
            if z:
                st.session_state["map_view"]["zoom"] = int(z)

        # Cattura l'ultima polyline disegnata (LineString) come "freccia" per angolo
        if map_state and map_state.get("last_active_drawing"):
            g = map_state["last_active_drawing"].get("geometry", {})
            if g.get("type") == "LineString":
                st.session_state["last_line_coords"] = g.get("coordinates")

        # Persisti la FeatureCollection completa dei disegni
        if map_state and map_state.get("all_drawings"):
            fc = map_state["all_drawings"]
            if isinstance(fc, list):
                fc = {"type": "FeatureCollection", "features": fc}
            if isinstance(fc, dict) and fc.get("type") == "FeatureCollection":
                st.session_state["draw_fc"] = fc

        # Deriva un buffer di soli poligoni (draft) per i pulsanti di aggiunta/sostituzione
        st.session_state["draw_buffer"] = []
        for f in (st.session_state.get("draw_fc", {}) or {}).get("features", []):
            try:
                g = f.get("geometry", {})
                if g and g.get("type") == "Polygon":
                    st.session_state["draw_buffer"].append(f)
            except Exception:
                pass

        st.caption(f"Poligoni disegnati (draft): {len(st.session_state.get('draw_buffer', []))}")

    # ----------------------
    # Controlli area
    # ----------------------
    with col_ctrl:
        st.session_state.setdefault("ori_new", "S")
        st.session_state.setdefault("angle_new", int(angle_from_orientation(st.session_state["ori_new"])))

        if st.button("↗️ Angolo da freccia", disabled=(st.session_state.get("last_line_coords") is None)):
            coords = st.session_state.get("last_line_coords") or []
            coords = [(float(x), float(y)) for x, y in coords]
            ang = int(round(angle_deg_from_line_lonlat(coords)))
            st.session_state["ori_new"] = orientation_from_angle(ang)
            st.session_state["angle_new"] = ang
            st.rerun()

        new_ori_val = st.selectbox(
            "Orientamento (classe colore)",
            ORIENTATIONS,
            index=ORIENTATIONS.index(st.session_state["ori_new"]),
        )
        if new_ori_val != st.session_state["ori_new"]:
            st.session_state["ori_new"] = new_ori_val
            st.session_state["angle_new"] = int(angle_from_orientation(new_ori_val))
            st.rerun()

        tilt = st.number_input("Pendenza (°)", 0, 60, 30, key="tilt_new")

        st.number_input(
            "Angolo reale (°) [0=S, 90=E, 180=N, 270=W]",
            min_value=0,
            max_value=359,
            key="angle_new",
        )

        c1, c2 = st.columns(2)
        if c1.button("➕ Aggiungi area", disabled=(len(st.session_state["draw_buffer"]) == 0)):
            feat = st.session_state["draw_buffer"][-1]
            poly = shape(feat["geometry"])
            editing.setdefault("areas", []).append(
                {
                    "geom": mapping(poly),
                    "orientation": st.session_state["ori_new"],
                    "angle_deg": float(st.session_state["angle_new"]),
                    "tilt_deg": float(tilt),
                    "n_panels": 0,
                    "kwp": 0.0,
                    "annual_kwh": None,
                }
            )

            # dopo l'aggiunta consumiamo il draft
            st.session_state["draw_buffer"].clear()
            st.session_state["draw_fc"] = {"type": "FeatureCollection", "features": []}
            st.success("Area aggiunta.")

        if editing.get("areas"):
            st.divider()
            idx_sel = st.selectbox(
                "Area da modificare/eliminare",
                options=list(range(len(editing["areas"]))),
                format_func=lambda i: f"Area #{i+1}",
                index=0,
                key="edit_area_pick",
            )
            a = editing["areas"][idx_sel]

            c3, c4 = st.columns(2)
            if c3.button("Aggiorna parametri area selezionata"):
                a["orientation"] = st.session_state["ori_new"]
                a["angle_deg"] = float(st.session_state["angle_new"])
                a["tilt_deg"] = float(tilt)
                st.success("Parametri aggiornati.")

            if c4.button(
                "Sostituisci geometria con ultimo disegno",
                disabled=(len(st.session_state["draw_buffer"]) == 0),
            ):
                feat = st.session_state["draw_buffer"][-1]
                a["geom"] = mapping(shape(feat["geometry"]))

                st.session_state["draw_buffer"].clear()
                st.session_state["draw_fc"] = {"type": "FeatureCollection", "features": []}
                st.success("Geometria aggiornata.")

            if st.button("🗑️ Rimuovi questa area", type="secondary"):
                editing["areas"].pop(idx_sel)
                st.warning("Area rimossa.")


# ==============================================================================
# Sezione: Render 3D tetto
# ==============================================================================

def _render_roof_3d(editing: Producer) -> None:
    st.subheader("Render 3D tetto (per aree con pendenza/orientamento)")

    areas_for_3d = editing.get("areas_norm") or editing.get("areas") or []
    if not areas_for_3d:
        st.info("Aggiungi almeno un'area con pendenza e orientamento.")
        return

    if go is None:
        st.warning("Plotly non è disponibile: la visualizzazione 3D è disabilitata.")
        return

    # Opzionale: altezza pareti/base. (Al momento non modifica roof_mesh_from_areas: resta qui per UI future.)
    _ = st.number_input(
        "Altezza base/pareti (m) – opzionale",
        0.0,
        20.0,
        0.0,
        0.1,
        key="wall_h_3d",
        help="Se >0, aggiunge un basamento estruso; il tetto resta inclinato.",
    )

    res = roof_mesh_from_areas(areas_for_3d)
    meshes = res.get("meshes", [])

    # Recentra coordinate (stabilizza WebGL: evita distorsioni con coordinate grandi)
    ox = oy = 0.0
    for m0 in meshes:
        if m0.get("x") and m0.get("y"):
            ox, oy = float(m0["x"][0]), float(m0["y"][0])
            break

    fig = go.Figure()
    for m in meshes:
        if not (m.get("x") and m.get("faces")):
            continue
        i = [f[0] for f in m["faces"]]
        j = [f[1] for f in m["faces"]]
        k = [f[2] for f in m["faces"]]
        fig.add_trace(
            go.Mesh3d(
                x=[vx - ox for vx in m["x"]],
                y=[vy - oy for vy in m["y"]],
                z=m["z"],
                i=i,
                j=j,
                k=k,
                opacity=0.6,
                flatshading=True,
                name="tetto",
            )
        )

    fig.update_layout(scene=dict(aspectmode="data"), height=520, margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# Sezione: Maschera (satellite / overlay)
# ==============================================================================

def _render_mask_section(editing: Producer) -> None:
    st.subheader("Maschera (satellite / mask / overlay)")

    if not editing.get("areas"):
        st.info("Aggiungi almeno un'area per generare la maschera.")
        return

    cM1, cM2, cM3 = st.columns([1, 1, 1])
    area_scope = cM1.selectbox(
        "Scopo",
        ["Tutte le aree"] + [f"Area #{i+1}" for i in range(len(editing["areas"]))],
    )
    zoom_req = cM2.slider("Zoom", 18, 23, value=st.session_state["map_view"]["zoom"])
    pad_pct = cM3.slider("Padding bbox (%)", 0, 60, 15)

    if area_scope == "Tutte le aree":
        geoms = [shape(a["geom"]) for a in editing["areas"]]
        oris = [a.get("orientation", "S") for a in editing["areas"]]
    else:
        i = int(area_scope.split("#")[1]) - 1
        geoms = [shape(editing["areas"][i]["geom"])]
        oris = [editing["areas"][i].get("orientation", "S")]

    # bounding box dei poligoni (lon/lat) + padding percentuale
    all_coords = np.vstack([np.asarray(g.exterior.coords) for g in geoms])
    min_lon, min_lat = all_coords.min(axis=0)
    max_lon, max_lat = all_coords.max(axis=0)
    lon_pad = (max_lon - min_lon) * (pad_pct / 100.0)
    lat_pad = (max_lat - min_lat) * (pad_pct / 100.0)
    bbox = (float(min_lon - lon_pad), float(min_lat - lat_pad), float(max_lon + lon_pad), float(max_lat + lat_pad))

    st.session_state.setdefault("mask_off_e", 0.0)
    st.session_state.setdefault("mask_off_n", 0.0)

    cO1, cO2, cO3 = st.columns([1, 1, 1])
    st.session_state["mask_off_e"] = cO1.number_input(
        "Offset Est (m)",
        -100.0,
        100.0,
        float(st.session_state["mask_off_e"]),
        step=0.5,
    )
    st.session_state["mask_off_n"] = cO2.number_input(
        "Offset Nord (m)",
        -100.0,
        100.0,
        float(st.session_state["mask_off_n"]),
        step=0.5,
    )
    alpha_mask = cO3.slider("Trasparenza overlay", 0, 255, 150)

    colB1, colB2 = st.columns(2)
    do_preview = colB1.button("👀 Anteprima")
    do_save = colB2.button("💾 Salva maschera")

    if not (do_preview or do_save):
        return

    with st.spinner("Genero satellite e maschera…"):
        sat_img, bbox_eff, used_zoom = fetch_satellite_image(bbox, zoom_req)
        mask_ids, mask_rgb = render_mask_from_polys(geoms, oris, bbox_eff, sat_img.size)

    min_lon, min_lat, max_lon, max_lat = bbox_eff
    lon_c, lat_c = (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0

    # Offset in metri -> shift in pixel:
    # (1) convertiamo il centro bbox in UTM
    # (2) applichiamo offset E/N (m)
    # (3) riconvertiamo in lon/lat e trasformiamo in pixel shift
    from pyproj import Transformer  # import locale: dipendenza solo per questa sezione

    utm_epsg = auto_utm_crs(lat_c, lon_c)
    fwd = Transformer.from_crs(4326, utm_epsg, always_xy=True)
    inv = Transformer.from_crs(utm_epsg, 4326, always_xy=True)

    Xc, Yc = fwd.transform(lon_c, lat_c)
    Xn, Yn = Xc + float(st.session_state["mask_off_e"]), Yc + float(st.session_state["mask_off_n"])
    lon2, lat2 = inv.transform(Xn, Yn)

    dlon = lon2 - lon_c
    dlat = lat2 - lat_c

    W, H = sat_img.size
    px_per_lon = (W - 1) / max(1e-9, (max_lon - min_lon))
    px_per_lat = (H - 1) / max(1e-9, (max_lat - min_lat))

    shift_x = int(dlon * px_per_lon)
    shift_y = int(-dlat * px_per_lat)  # nord positivo → y negativo

    sat_rgba = sat_img.convert("RGBA")
    mask_rgb_img = Image.open(io.BytesIO(mask_rgb)).convert("RGBA")
    alpha = Image.new("L", mask_rgb_img.size, alpha_mask)
    mask_rgb_img.putalpha(alpha)

    canvas = Image.new("RGBA", sat_rgba.size, (0, 0, 0, 0))
    canvas.paste(mask_rgb_img, box=(shift_x, shift_y), mask=mask_rgb_img)
    comp = Image.alpha_composite(sat_rgba, canvas)

    cA, cB, cC = st.columns(3)
    cA.image(sat_img, caption=f"Satellite (zoom {used_zoom})", use_container_width=True)
    cB.image(mask_rgb_img.convert("RGB"), caption="Mask (color, traslata)", use_container_width=True)
    cC.image(comp, caption="Satellite + Mask", use_container_width=True)

    if do_save:
        out_dir = producer_cache_dir(SESSION_DIR, editing["id"]) / "mask"
        out_dir.mkdir(parents=True, exist_ok=True)

        b_sat = io.BytesIO()
        sat_img.save(b_sat, format="PNG")

        ids_img = Image.open(io.BytesIO(mask_ids)).convert("RGBA")
        canvas_ids = Image.new("RGBA", sat_rgba.size, (0, 0, 0, 0))
        canvas_ids.paste(ids_img, box=(shift_x, shift_y), mask=ids_img)
        b_ids = io.BytesIO()
        canvas_ids.convert("RGB").save(b_ids, format="PNG")

        b_rgb = io.BytesIO()
        canvas.convert("RGB").save(b_rgb, format="PNG")

        (out_dir / "satellite.png").write_bytes(b_sat.getvalue())
        (out_dir / "mask_ids.png").write_bytes(b_ids.getvalue())
        (out_dir / "mask_rgb.png").write_bytes(b_rgb.getvalue())

        meta = {
            "bbox": bbox_eff,
            "zoom": used_zoom,
            "offset_m": [float(st.session_state["mask_off_e"]), float(st.session_state["mask_off_n"])],
            "alpha": int(alpha_mask),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        st.success("Maschera salvata.")
        st.download_button(
            "⬇️ satellite.png",
            b_sat.getvalue(),
            file_name=f"{editing['name'].replace(' ', '_')}_satellite.png",
        )
        st.download_button(
            "⬇️ mask_ids.png",
            b_ids.getvalue(),
            file_name=f"{editing['name'].replace(' ', '_')}_mask_ids.png",
        )
        st.download_button(
            "⬇️ mask_rgb.png",
            b_rgb.getvalue(),
            file_name=f"{editing['name'].replace(' ', '_')}_mask_rgb.png",
        )


# ==============================================================================
# Sezione: Posizionamento pannelli (render 3D)
# ==============================================================================

def _render_panel_placement_3d(editing: Producer) -> None:
    st.subheader("Posizionamento pannelli (render 3D)")

    areas_for_pack = editing.get("areas_norm") or editing.get("areas") or []
    if not areas_for_pack:
        st.info("Aggiungi almeno un'area con pendenza e orientamento.")
        return

    if go is None:
        st.warning("Plotly non è disponibile: la sezione di posizionamento 3D è disabilitata.")
        return

    pv = _ensure_pv_defaults_in_editing(editing)

    # default widget state (se non inizializzato) dai valori del produttore
    st.session_state.setdefault("pv_panel_w", float(pv.get("panel_w", 1.10)))
    st.session_state.setdefault("pv_panel_h", float(pv.get("panel_h", 1.80)))
    st.session_state.setdefault("pv_gap", float(pv.get("gap", 0.20)))
    st.session_state.setdefault("pv_margin", float(pv.get("margin", 0.05)))
    st.session_state.setdefault("pv_coverage_min_pct", float(pv.get("coverage_min_pct", 95.0)))
    st.session_state.setdefault("pv_wpp", int(pv.get("wpp", 420)))
    st.session_state.setdefault("pv_system_loss", float(pv.get("system_loss", 14.0)))
    st.session_state.setdefault("pv_tech", str(pv.get("tech", "crystSi")))

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    panel_w = c1.number_input("Larghezza pannello (m)", 0.20, 3.0, float(st.session_state["pv_panel_w"]), 0.01, key="pv_panel_w")
    panel_h = c2.number_input("Altezza pannello (m)", 0.20, 3.0, float(st.session_state["pv_panel_h"]), 0.01, key="pv_panel_h")
    gap = c3.number_input("Gap tra pannelli (m)", 0.00, 1.00, float(st.session_state["pv_gap"]), 0.01, key="pv_gap")
    margin = c4.number_input("Margine bordo (m)", 0.00, 0.50, float(st.session_state["pv_margin"]), 0.01, key="pv_margin")
    cov_pct = c5.number_input("Coverage minimo (%)", 50.0, 100.0, float(st.session_state["pv_coverage_min_pct"]), 0.5, key="pv_coverage_min_pct")
    coverage = float(cov_pct) / 100.0
    wpp = c6.number_input("Potenza per pannello (Wp)", 50, 1000, int(st.session_state["pv_wpp"]), 10, key="pv_wpp")
    _ = c7.number_input("System loss (%)", 0.0, 40.0, float(st.session_state["pv_system_loss"]), 0.5, key="pv_system_loss")

    _ = st.selectbox(
        "Tecnologia FV (PVGIS)",
        options=PVGIS_TECH_CODES,
        index=PVGIS_TECH_CODES.index(st.session_state.get("pv_tech", "crystSi"))
        if st.session_state.get("pv_tech", "crystSi") in PVGIS_TECH_CODES
        else 0,
        format_func=lambda c: PVGIS_TECH_LABEL.get(c, c),
        key="pv_tech",
        help="PVGIS supporta: Crystalline Silicon (original), CIS, CdTe.",
    )

    stagger = st.checkbox("Sfalsa file (stagger)", value=False)
    sel = st.multiselect(
        "Aree",
        options=list(range(len(areas_for_pack))),
        format_func=lambda i: f"Area #{i+1}",
        default=list(range(len(areas_for_pack))),
        key="sel_panels_3d",
    )

    place = st.button(
        "📐 Posiziona pannelli sul render 3D",
        type="primary",
        use_container_width=True,
        key="btn_place_3d",
    )
    st.markdown("---")

    # usa un unico CRS/trasformazione per roof render
    fwd, inv = local_transformers_for_areas(areas_for_pack)

    fig = go.Figure()
    res_roof = roof_mesh_from_areas([areas_for_pack[i] for i in (sel or [])] or areas_for_pack, fwd=fwd, inv=inv)

    # Recentra coordinate per stabilità WebGL
    ox = oy = 0.0
    for m0 in res_roof.get("meshes", []):
        if m0.get("x") and m0.get("y"):
            ox, oy = float(m0["x"][0]), float(m0["y"][0])
            break

    for m in res_roof.get("meshes", []):
        if m.get("x") and m.get("faces"):
            i_arr = [f[0] for f in m["faces"]]
            j_arr = [f[1] for f in m["faces"]]
            k_arr = [f[2] for f in m["faces"]]
            fig.add_trace(
                go.Mesh3d(
                    x=[vx - ox for vx in m["x"]],
                    y=[vy - oy for vy in m["y"]],
                    z=m["z"],
                    i=i_arr,
                    j=j_arr,
                    k=k_arr,
                    opacity=0.35,
                    name="tetto",
                    flatshading=True,
                )
            )

    # Render pannelli già posizionati (persistiti nel JSON)
    sel_idxs = sel or (editing.get("pv", {}) or {}).get("selected_area_idxs") or []
    for i in sel_idxs:
        try:
            a = areas_for_pack[i]
        except Exception:
            continue
        gj_pan = a.get("panels_geojson")
        if not gj_pan:
            continue
        panels_geo = _panels_geo_from_geojson_list(gj_pan)
        if not panels_geo:
            continue

        pm = panels_mesh_from_area(a, panels_geo, fwd=fwd, inv=inv)
        if pm.get("x"):
            fig.add_trace(
                go.Mesh3d(
                    x=[vx - ox for vx in pm["x"]],
                    y=[vy - oy for vy in pm["y"]],
                    z=pm["z"],
                    i=pm["i"],
                    j=pm["j"],
                    k=pm["k"],
                    opacity=0.85,
                    name=f"Pannelli area #{i+1}",
                )
            )

    total_kwp = 0.0
    if place and sel:
        # salva i parametri FV nel produttore (e aggiorna i default globali)
        _persist_pv_from_form_into_editing(editing)

        # se cambia selezione aree: azzera aree non selezionate
        sel_set = set(sel)
        for jj, aa in enumerate(areas_for_pack):
            if jj not in sel_set:
                aa["n_panels"] = 0
                aa["kwp"] = 0.0
                aa.pop("panels_geojson", None)

        for i in sel:
            a = areas_for_pack[i]
            poly_geo = shape(a["geom"])

            # Packing Local Area Frame: lavora in metri ma salva pannelli in lon/lat
            panels_geo, _aux = pack_panels_laf_lonlat(
                poly_geo,
                float(a.get("angle_deg", 0.0)),
                float(panel_w),
                float(panel_h),
                float(gap),
                float(margin),
                float(coverage),
                bool(stagger),
            )
            a["panels_geojson"] = _panels_geo_to_geojson_list(panels_geo)

            n_pan = len(panels_geo)
            a["n_panels"] = n_pan
            a["kwp"] = (n_pan * wpp) / 1000.0

            # persist parametri FV per-area (riproducibilità)
            a["panel_w"] = float(panel_w)
            a["panel_h"] = float(panel_h)
            a["gap"] = float(gap)
            a["margin"] = float(margin)
            a["coverage_min_pct"] = float(cov_pct)
            a["wp_panel"] = int(wpp)
            a["stagger"] = bool(stagger)

            total_kwp += (n_pan * wpp) / 1000.0

            pm = panels_mesh_from_area(a, panels_geo, fwd=fwd, inv=inv)
            if pm.get("x"):
                fig.add_trace(
                    go.Mesh3d(
                        x=[vx - ox for vx in pm["x"]],
                        y=[vy - oy for vy in pm["y"]],
                        z=pm["z"],
                        i=pm["i"],
                        j=pm["j"],
                        k=pm["k"],
                        opacity=0.85,
                        name=f"Pannelli area #{i+1}",
                    )
                )

        # persist indici aree selezionate (usati anche da PVGIS di default)
        editing.setdefault("pv", {})["selected_area_idxs"] = list(sel)  # type: ignore[index]

        st.success(f"Pannelli posizionati. kWp installati (aree selezionate): **{total_kwp:.3f} kWp**")

    fig.update_layout(scene=dict(aspectmode="data"), height=620, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# Sezione: PVGIS
# ==============================================================================

def _render_pvgis(editing: Producer) -> None:
    st.subheader("Produzione fotovoltaica (PVGIS) — anni fissi, grafico con aggregazione")

    if not editing.get("areas"):
        st.info("Aggiungi almeno un'area per simulare la produzione.")
        return

    # allinea editing['pv'] ai widget correnti (tecnologia/loss inclusi)
    _persist_pv_from_form_into_editing(editing, update_globals=False)
    pv = _ensure_pv_defaults_in_editing(editing)

    year = st.radio("Anno di simulazione", options=[2022, 2023], index=1, horizontal=True)
    agg_choice = st.selectbox("Risoluzione grafico", ["Oraria", "Giornaliera", "Settimanale", "Mensile"], index=0)
    rule_map = {"Oraria": "H", "Giornaliera": "D", "Settimanale": "W", "Mensile": "M"}
    rule = rule_map[agg_choice]

    month: Optional[int] = None
    if agg_choice == "Oraria":
        month = st.selectbox(
            "Mese (solo per grafico orario)",
            options=list(range(1, 13)),
            format_func=lambda m: calendar.month_name[m],
            index=0,
            help="Limita l’intervallo orario a un singolo mese.",
        )

    areas = editing["areas"]
    idxs = list(range(len(areas)))

    # default: usa le aree selezionate nel posizionamento pannelli (se presenti)
    pick = list((editing.get("pv", {}) or {}).get("selected_area_idxs") or [])
    if not pick:
        pick = idxs

    st.info(
        "Aree usate per PVGIS (auto): "
        + (", ".join([f"Area #{i+1}" for i in pick]) if pick else "Nessuna area selezionata.")
    )
    with st.expander("Opzionale: override manuale aree PVGIS", expanded=False):
        pick = st.multiselect("Aree da simulare", options=idxs, format_func=lambda i: f"Area #{i+1}", default=pick)


    # ----------------------------
    # Button: fetch/cache PVGIS
    # ----------------------------
    if st.button("⚡ Calcola/Carica produzione (oraria)"):
        total_kwh_year = 0.0
        any_ok = False

        for i in pick:
            a = areas[i]
            g = shape(a["geom"])
            tilt = float(a.get("tilt_deg", 30))
            az = float(a.get("angle_deg", angle_from_orientation(a.get("orientation", "S"))))

            kwp = float(a.get("kwp", 0.0))
            if kwp <= 0:
                kwp = float(a.get("n_panels", 0)) * float(pv.get("wpp", st.session_state["wp_panel_global"])) / 1000.0

            loss = float(pv.get("system_loss", st.session_state["system_loss_global"]))
            tech = str(pv.get("tech", "crystSi"))

            # cache key: include tutti i parametri che influenzano PVGIS
            key = f"a{i}_y{year}-{year}_kwp{kwp:.3f}_loss{loss:.1f}_tilt{tilt:.1f}_az{az:.1f}_tech{tech}"
            df = cache_pvgis_load(SESSION_DIR, editing["id"], key)

            if df is None:
                try:
                    df = pvgis_hourly_for_area(g, tilt, az, kwp, loss, int(year), int(year), tech=tech)
                except Exception as e:
                    st.error(f"PVGIS error Area #{i+1}: {e}")
                    continue
                cache_pvgis_save(SESSION_DIR, editing["id"], df, key)

            # Calcolo energia annua dall'output PVGIS:
            # - df['P'] è in W
            # - energia = sum(P * dt) / 1000
            try:
                if df is not None and (not df.empty) and ("P" in df.columns):
                    df_full = df[df.index.year == year]
                    if not df_full.empty:
                        step_hours = _median_step_hours(df_full.index)
                        kwh_area = float((df_full["P"] * step_hours / 1000.0).sum())
                        total_kwh_year += kwh_area
                        any_ok = True
            except Exception:
                pass

        if any_ok:
            editing["annual_kwh"] = float(total_kwh_year)

            # Propaga anche alla lista in memoria per aggiornare la tabella in alto
            for idx, pr in enumerate(st.session_state.get("producers", [])):
                if pr.get("id") == editing.get("id"):
                    st.session_state["producers"][idx]["annual_kwh"] = float(total_kwh_year)
                    break

            st.success("Produzione PVGIS aggiornata in cache (kWh/anno aggiornati).")
        else:
            st.warning(
                "Produzione PVGIS aggiornata in cache, ma non ho potuto calcolare i kWh/anno "
                "(dati incompleti o errore PVGIS)."
            )

    # ----------------------------
    # Dati per grafico e download (da cache)
    # ----------------------------
    dfs_plot: list[pd.DataFrame] = []
    dfs_download: list[pd.DataFrame] = []

    for i in pick:
        a = areas[i]
        tilt = float(a.get("tilt_deg", 30))
        az = float(a.get("angle_deg", angle_from_orientation(a.get("orientation", "S"))))

        kwp = float(a.get("kwp", 0.0))
        if kwp <= 0:
            kwp = float(a.get("n_panels", 0)) * float(pv.get("wpp", st.session_state["wp_panel_global"])) / 1000.0

        loss = float(pv.get("system_loss", st.session_state["system_loss_global"]))
        tech = str(pv.get("tech", "crystSi"))

        key = f"a{i}_y{year}-{year}_kwp{kwp:.3f}_loss{loss:.1f}_tilt{tilt:.1f}_az{az:.1f}_tech{tech}"
        df = cache_pvgis_load(SESSION_DIR, editing["id"], key)

        if df is None or df.empty or "P" not in df.columns:
            continue

        df_full = df[df.index.year == year]
        if df_full.empty:
            continue

        step_hours = _median_step_hours(df_full.index)

        # anno intero (base per download)
        e_kwh_full = (df_full["P"] * step_hours / 1000.0).rename(f"Area_{i+1}")
        series_full = e_kwh_full.resample("H").sum(min_count=1) if rule == "H" else e_kwh_full.resample(rule).sum(min_count=1)
        dfs_download.append(series_full.to_frame())

        # dataset per grafico: se orario, puoi limitare al mese selezionato
        df_plot = df_full
        if rule == "H" and month is not None:
            df_plot = df_full[df_full.index.month == month]

        if not df_plot.empty:
            e_kwh_plot = (df_plot["P"] * step_hours / 1000.0).rename(f"Area_{i+1}")
            series_plot = e_kwh_plot.resample("H").sum(min_count=1) if rule == "H" else e_kwh_plot.resample(rule).sum(min_count=1)
            dfs_plot.append(series_plot.to_frame())

    # --- GRAFICO (mese se orario) ---
    if dfs_plot:
        dfAgg_plot = pd.concat(dfs_plot, axis=1).sort_index()
        pv_total_plot = dfAgg_plot.sum(axis=1)
        dfAgg_plot["Totale"] = pv_total_plot

        if go is not None:
            fig = go.Figure()
            for col in [c for c in dfAgg_plot.columns if c != "Totale"]:
                fig.add_trace(go.Scattergl(x=dfAgg_plot.index, y=dfAgg_plot[col], name=col, mode="lines"))
            fig.add_trace(go.Scattergl(x=dfAgg_plot.index, y=dfAgg_plot["Totale"], name="Totale", mode="lines"))

            if rule == "H" and month is not None:
                titolo = f"Energia {agg_choice.lower()} (kWh) — {calendar.month_name[month]} {year}"
            else:
                titolo = f"Energia {agg_choice.lower()} (kWh) — {year}"

            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                title=titolo,
                xaxis_title="Data",
                yaxis_title="kWh",
                legend=dict(orientation="h", y=1.1),
            )
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_traces(line_simplify=True, selector=dict(type="scatter"))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
        else:
            st.line_chart(dfAgg_plot)

    # --- DOWNLOAD (sempre anno intero, anche se il grafico mostra un mese) ---
    if dfs_download:
        dfAgg_dl = pd.concat(dfs_download, axis=1).sort_index()
        pv_total_dl = dfAgg_dl.sum(axis=1)
        dfAgg_dl["Totale"] = pv_total_dl

        # salva kWh annui totali nel produttore (per tabella lista produttori)
        try:
            editing["annual_kwh"] = float(dfAgg_dl["Totale"].sum())
        except Exception:
            pass

        csv = dfAgg_dl.to_csv(index=True).encode("utf-8")

        suffix = " — anno intero" if (rule == "H" and month is not None) else ""
        st.download_button(
            f"⬇️ Produzione {agg_choice.lower()} per area (CSV, wide){suffix}",
            data=csv,
            file_name=f"{editing['name'].replace(' ', '_')}_pvgis_{rule}_wide_{year}_FULLYEAR.csv",
            mime="text/csv",
        )


# ==============================================================================
# Sezione: Eolico
# ==============================================================================

def _default_power_curve() -> list[dict[str, float]]:
    """Curva di potenza di default (placeholder)."""

    return [
        {"v_ms": 0.0, "p_kw": 0.0},
        {"v_ms": 3.0, "p_kw": 0.0},  # cut-in tipico
        {"v_ms": 4.0, "p_kw": 25.0},
        {"v_ms": 6.0, "p_kw": 90.0},
        {"v_ms": 8.0, "p_kw": 170.0},
        {"v_ms": 10.0, "p_kw": 250.0},
        {"v_ms": 12.0, "p_kw": 300.0},  # rated
        {"v_ms": 25.0, "p_kw": 300.0},
        {"v_ms": 26.0, "p_kw": 0.0},  # cut-out
    ]


def _ensure_wind_defaults_in_editing(editing: Producer) -> dict[str, Any]:
    """Inizializza la struttura `editing['wind']` in modo retro-compatibile."""

    w = editing.setdefault("wind", {})  # type: ignore[assignment]
    if not isinstance(w, dict):
        w = {}
        editing["wind"] = w  # type: ignore[index]
    w.setdefault("turbines", [])
    return w


def _render_eolico(editing: Producer) -> None:
    st.subheader("Produzione eolica — da meteo di sessione (cache su disco)")

    w = _ensure_wind_defaults_in_editing(editing)
    turbines: list[dict[str, Any]] = list(w.get("turbines") or [])

    if not PATHS.meteo_hourly_csv.exists():
        st.warning(
            "Meteo di sessione non trovato: manca meteo_hourly.csv. "
            "Torna in Home e genera il meteo prima di calcolare l'eolico."
        )
        return

    # Leggi solo intestazioni per opzioni colonna vento (best-effort)
    wind_cols: list[str] = ["wind_speed_100m"]
    try:
        df_head = pd.read_csv(PATHS.meteo_hourly_csv, nrows=5)
        wind_cols = [c for c in df_head.columns if "wind_speed" in c] or wind_cols
    except Exception:
        pass

    # Libreria turbine (generic)
    try:
        lib = load_wind_turbine_library()
        lib_models: list[dict[str, Any]] = list(lib.get("models") or [])
    except Exception as e:
        st.error(f"Errore caricamento libreria turbine: {e}")
        return

    if not lib_models:
        st.error("Libreria turbine vuota: nessun modello disponibile.")
        return

    model_by_id: dict[str, dict[str, Any]] = {str(m.get("model_id")): m for m in lib_models if m.get("model_id")}
    model_ids: list[str] = list(model_by_id.keys())

    def _fmt_model(mid: str) -> str:
        m = model_by_id.get(mid) or {}
        name = str(m.get("name", mid))
        rated = m.get("rated_power_kw")
        try:
            rated_s = f" — {float(rated):.0f} kW" if rated is not None else ""
        except Exception:
            rated_s = ""
        return name + rated_s

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("➕ Aggiungi turbina", key=f"wind_add_{editing['id']}"):
            new_idx = 1
            existing_ids = {str(t.get("id")) for t in turbines if t.get("id") is not None}
            while str(new_idx) in existing_ids:
                new_idx += 1
            turbines.append(
                {
                    "id": str(new_idx),
                    "name": f"Turbina {new_idx}",
                    "enabled": True,
                    "count": 1,
                    "model_id": model_ids[0],
                    "hub_height_m": float((model_by_id.get(model_ids[0]) or {}).get("default_hub_height_m", 80.0)),
                    "ref_height_m": 100.0,
                    "shear_alpha": 0.14,
                    "loss_pct": 12.0,
                    "wind_speed_col": wind_cols[0] if wind_cols else "wind_speed_100m",
                }
            )
            w["turbines"] = turbines
            st.rerun()

    # ---- editor turbine ----
    if not turbines:
        st.info("Nessuna turbina definita. Aggiungi almeno una turbina per calcolare l'eolico.")
        return

    st.caption("Parametri: shear a quota mozzo, curva di potenza con interpolazione lineare, perdite globali.")

    dirty_any = False

    for idx, t in enumerate(turbines):
        tid = str(t.get("id", idx + 1))
        exp_label = f"{t.get('name', f'Turbina {tid}')} (id={tid})"

        # Widget keys (stabili per producer + turbina)
        kbase = f"wind_{editing['id']}_{tid}"
        k_init = f"{kbase}__init"

        def _init_if_needed() -> None:
            """Inizializza i widget con i valori 'committati' (turbines) una sola volta."""
            if not st.session_state.get(k_init, False):
                st.session_state[f"{kbase}_enabled"] = bool(t.get("enabled", True))
                st.session_state[f"{kbase}_count"] = int(t.get("count", 1))
                st.session_state[f"{kbase}_name"] = str(t.get("name", f"Turbina {tid}"))
                cur_mid = str(t.get("model_id") or model_ids[0])
                if cur_mid not in model_by_id:
                    cur_mid = model_ids[0]
                st.session_state[f"{kbase}_model_id"] = cur_mid
                st.session_state[f"{kbase}_hub"] = float(t.get("hub_height_m", float((model_by_id.get(cur_mid) or {}).get("default_hub_height_m", 80.0))))
                st.session_state[f"{kbase}_ref"] = float(t.get("ref_height_m", 100.0))
                st.session_state[f"{kbase}_alpha"] = float(t.get("shear_alpha", 0.14))
                st.session_state[f"{kbase}_loss"] = float(t.get("loss_pct", 12.0))
                # wind column
                cur_col = str(t.get("wind_speed_col", wind_cols[0] if wind_cols else "wind_speed_100m"))
                if wind_cols and cur_col not in wind_cols:
                    cur_col = wind_cols[0]
                st.session_state[f"{kbase}_windcol"] = cur_col
                st.session_state[k_init] = True

        def _is_dirty() -> bool:
            """True se i valori nei widget differiscono dai valori committati nel dict turbina."""
            try:
                committed_mid = str(t.get("model_id") or model_ids[0])
                committed_col = str(t.get("wind_speed_col", wind_cols[0] if wind_cols else "wind_speed_100m"))
                return any(
                    [
                        bool(t.get("enabled", True)) != bool(st.session_state.get(f"{kbase}_enabled")),
                        int(t.get("count", 1)) != int(st.session_state.get(f"{kbase}_count")),
                        str(t.get("name", f"Turbina {tid}")) != str(st.session_state.get(f"{kbase}_name")),
                        committed_mid != str(st.session_state.get(f"{kbase}_model_id")),
                        float(t.get("hub_height_m", 80.0)) != float(st.session_state.get(f"{kbase}_hub")),
                        float(t.get("ref_height_m", 100.0)) != float(st.session_state.get(f"{kbase}_ref")),
                        float(t.get("shear_alpha", 0.14)) != float(st.session_state.get(f"{kbase}_alpha")),
                        float(t.get("loss_pct", 12.0)) != float(st.session_state.get(f"{kbase}_loss")),
                        committed_col != str(st.session_state.get(f"{kbase}_windcol")),
                    ]
                )
            except Exception:
                return False

        def _persist_current_editing() -> None:
            """Persisti l'editing su producers.json (solo se la lista producers è disponibile)."""
            if "producers" not in st.session_state or not isinstance(st.session_state["producers"], list):
                return
            try:
                pid = int(editing["id"])
                for pidx, p in enumerate(st.session_state["producers"]):
                    if int(p.get("id", -1)) == pid:
                        st.session_state["producers"][pidx] = editing
                        break
                save_producers(SESSION_DIR, st.session_state["producers"])
            except Exception:
                # best-effort: non bloccare la UI
                pass

        _init_if_needed()

        dirty_now = _is_dirty()
        dirty_any = dirty_any or dirty_now

        with st.expander(exp_label, expanded=(idx == 0)):
            # Stato "dirty" (modifiche non salvate)
            if dirty_now:
                st.warning("Modifiche non salvate: premi **Salva turbina** per applicarle.", icon="⚠️")

            # Eliminazione immediata (committata)
            c_top1, c_top2, c_top3 = st.columns([1, 1, 1])
            with c_top3:
                if st.button("🗑️ Elimina", key=f"wind_del_{editing['id']}_{tid}"):
                    turbines = [x for x in turbines if str(x.get("id")) != tid]
                    w["turbines"] = turbines
                    _persist_current_editing()
                    st.rerun()

            # Form di editing (draft) -> Salva
            with st.form(key=f"wind_form_{editing['id']}_{tid}", clear_on_submit=False):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.checkbox("Abilitata", key=f"{kbase}_enabled")
                with c2:
                    st.number_input("Numero turbine", min_value=0, step=1, key=f"{kbase}_count")
                with c3:
                    st.text_input("Nome", key=f"{kbase}_name")

                st.selectbox(
                    "Modello turbina (libreria)",
                    options=model_ids,
                    format_func=_fmt_model,
                    key=f"{kbase}_model_id",
                )
                msel = model_by_id.get(str(st.session_state.get(f"{kbase}_model_id"))) or {}
                try:
                    st.caption(
                        f"Potenza nominale: {float(msel.get('rated_power_kw', 0.0)):.0f} kW • "
                        f"Cut-in: {float(msel.get('cut_in_ms', 0.0)):.1f} m/s • "
                        f"Rated: {float(msel.get('rated_speed_ms', 0.0)):.1f} m/s • "
                        f"Cut-out: {float(msel.get('cut_out_ms', 0.0)):.1f} m/s"
                    )
                except Exception:
                    pass

                with st.expander("Anteprima curva di potenza (read-only)", expanded=False):
                    try:
                        df_curve_ro = pd.DataFrame(msel.get("power_curve") or [])
                        if not df_curve_ro.empty:
                            st.dataframe(df_curve_ro, use_container_width=True, hide_index=True)
                        else:
                            st.info("Curva di potenza non disponibile per questo modello.")
                    except Exception:
                        st.info("Curva di potenza non disponibile per questo modello.")

                cH1, cH2, cH3, cH4 = st.columns(4)
                with cH1:
                    st.number_input("Hub height (m)", min_value=1.0, step=1.0, key=f"{kbase}_hub")
                with cH2:
                    st.number_input("Ref height (m)", min_value=1.0, step=1.0, key=f"{kbase}_ref")
                with cH3:
                    st.number_input("Shear α", min_value=0.0, max_value=1.0, step=0.01, key=f"{kbase}_alpha")
                with cH4:
                    st.number_input("Perdite (%)", min_value=0.0, max_value=100.0, step=0.5, key=f"{kbase}_loss")

                st.selectbox("Colonna vento (m/s)", options=wind_cols, key=f"{kbase}_windcol")

                cS1, cS2 = st.columns([1, 1])
                with cS1:
                    save_clicked = st.form_submit_button("💾 Salva turbina")
                with cS2:
                    reset_clicked = st.form_submit_button("↩️ Annulla modifiche")

                if reset_clicked:
                    # reset draft ai valori committati
                    st.session_state[k_init] = False
                    st.rerun()

                if save_clicked:
                    # Commit: sostituisci il dict turbina con una nuova copia (no in-place)
                    t_new = dict(t)
                    t_new["enabled"] = bool(st.session_state.get(f"{kbase}_enabled", True))
                    t_new["count"] = int(st.session_state.get(f"{kbase}_count", 1))
                    t_new["name"] = str(st.session_state.get(f"{kbase}_name", f"Turbina {tid}"))
                    t_new["model_id"] = str(st.session_state.get(f"{kbase}_model_id", model_ids[0]))
                    t_new["hub_height_m"] = float(st.session_state.get(f"{kbase}_hub", 80.0))
                    t_new["ref_height_m"] = float(st.session_state.get(f"{kbase}_ref", 100.0))
                    t_new["shear_alpha"] = float(st.session_state.get(f"{kbase}_alpha", 0.14))
                    t_new["loss_pct"] = float(st.session_state.get(f"{kbase}_loss", 12.0))
                    t_new["wind_speed_col"] = str(st.session_state.get(f"{kbase}_windcol", wind_cols[0] if wind_cols else "wind_speed_100m"))

                    # replace in list
                    new_list: list[dict[str, Any]] = []
                    for x in turbines:
                        if str(x.get("id")) == tid:
                            new_list.append(t_new)
                        else:
                            new_list.append(x)
                    turbines = new_list
                    w["turbines"] = turbines
                    _persist_current_editing()
                    st.success("Impostazioni turbina salvate.")
                    st.rerun()

    st.markdown("---")

    if dirty_any:
        st.warning("Ci sono modifiche non salvate nelle turbine. Premi **Salva turbina** prima di calcolare per applicarle.", icon="⚠️")

    # ---- Calcolo / Cache ----
    cC1, cC2 = st.columns([1, 2])
    with cC1:
        force = st.checkbox("Ricalcola (ignora cache)", value=False, key=f"wind_force_{editing['id']}")
    with cC2:
        if st.button("🌬️ Calcola/Carica produzione eolica (oraria)", key=f"wind_run_{editing['id']}"):
            try:
                df_w, key, from_cache = get_or_compute_eolico_hourly(
                    SESSION_DIR,
                    int(editing["id"]),
                    PATHS.meteo_hourly_csv,
                    turbines,
                    output_unit="kwh",
                    force=bool(force),
                )
                w["last_cache_key"] = key
                w["annual_kwh"] = float(df_w["Totale"].sum()) if ("Totale" in df_w.columns) else 0.0
                st.success(
                    ("Produzione eolica caricata da cache." if from_cache else "Produzione eolica calcolata e salvata in cache.")
                    + f"  (key={key})"
                )
            except Exception as e:
                st.error(f"Errore calcolo eolico: {e}")

    # ---- Visualizzazione + download (da cache) ----
    try:
        wkey = build_eolico_cache_key(PATHS.meteo_hourly_csv, turbines, output_unit="kwh")
        df_cached = cache_eolico_load(SESSION_DIR, int(editing["id"]), wkey)
    except Exception:
        df_cached = None

    if df_cached is None or df_cached.empty:
        st.info("Nessun risultato eolico in cache per i parametri attuali. Premi 'Calcola/Carica'.")
        return

    years = sorted({int(y) for y in df_cached.index.year})
    year = st.selectbox(
        "Anno (dal meteo di sessione)",
        options=years,
        index=len(years) - 1,
        key=f"wind_year_{editing['id']}",
    )
    agg_choice = st.selectbox(
        "Risoluzione grafico eolico",
        ["Oraria", "Giornaliera", "Settimanale", "Mensile"],
        index=0,
        key=f"wind_agg_{editing['id']}",
    )
    rule_map = {"Oraria": "H", "Giornaliera": "D", "Settimanale": "W", "Mensile": "M"}
    rule = rule_map[agg_choice]

    df_y = df_cached[df_cached.index.year == int(year)]
    if df_y.empty:
        st.warning("Nessun dato eolico per l'anno selezionato.")
        return

    # Aggrega
    if rule == "H":
        df_plot = df_y.resample("H").sum(min_count=1)
    else:
        df_plot = df_y.resample(rule).sum(min_count=1)

    if go is not None:
        fig = go.Figure()
        for col in [c for c in df_plot.columns if c != "Totale"]:
            fig.add_trace(go.Scattergl(x=df_plot.index, y=df_plot[col], name=col, mode="lines"))
        fig.add_trace(go.Scattergl(x=df_plot.index, y=df_plot["Totale"], name="Totale", mode="lines"))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"Energia {agg_choice.lower()} eolica (kWh) — {year}",
            xaxis_title="Data",
            yaxis_title="kWh",
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    else:
        st.line_chart(df_plot)

    csv = df_y.to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Produzione eolica oraria (CSV, wide)",
        data=csv,
        file_name=f"{editing['name'].replace(' ', '_')}_eolico_H_wide_{year}_FULLYEAR.csv",
        mime="text/csv",
        key=f"wind_dl_{editing['id']}",
    )



# ==============================================================================
# Sezione: Aggregazione FV + Eolico
# ==============================================================================

def _render_aggregazione_pv_eolico(editing: Producer) -> None:
    st.subheader("Aggregazione produzione — fotovoltaico + eolico")

    if not editing.get("areas"):
        st.info("Aggiungi almeno un'area FV per poter aggregare.")
        return

    if not PATHS.meteo_hourly_csv.exists():
        st.warning("Meteo di sessione non trovato: manca meteo_hourly.csv.")
        return

    # Allinea parametri FV
    _persist_pv_from_form_into_editing(editing, update_globals=False)
    pv = _ensure_pv_defaults_in_editing(editing)

    year = st.radio(
        "Anno di aggregazione",
        options=[2022, 2023],
        index=1,
        horizontal=True,
        key=f"agg_year_{editing['id']}",
    )
    agg_choice = st.selectbox(
        "Risoluzione grafico aggregazione",
        ["Oraria", "Giornaliera", "Settimanale", "Mensile"],
        index=0,
        key=f"agg_res_{editing['id']}",
    )
    rule_map = {"Oraria": "H", "Giornaliera": "D", "Settimanale": "W", "Mensile": "M"}
    rule = rule_map[agg_choice]

    month: Optional[int] = None
    if agg_choice == "Oraria":
        month = st.selectbox(
            "Mese (solo per grafico orario)",
            options=list(range(1, 13)),
            format_func=lambda m: calendar.month_name[m],
            index=0,
            key=f"agg_month_{editing['id']}",
        )

    areas = editing["areas"]
    idxs = list(range(len(areas)))
    pick = list((editing.get("pv", {}) or {}).get("selected_area_idxs") or [])
    if not pick:
        pick = idxs

    with st.expander("Override manuale aree FV per aggregazione", expanded=False):
        pick = st.multiselect(
            "Aree FV da includere",
            options=idxs,
            format_func=lambda i: f"Area #{i+1}",
            default=pick,
            key=f"agg_pick_{editing['id']}",
        )

    # ----------------------------
    # FV: carica da cache PVGIS
    # ----------------------------
    dfs_plot: list[pd.DataFrame] = []
    dfs_download: list[pd.DataFrame] = []

    for i in pick:
        a = areas[i]
        tilt = float(a.get("tilt_deg", 30))
        az = float(a.get("angle_deg", angle_from_orientation(a.get("orientation", "S"))))

        kwp = float(a.get("kwp", 0.0))
        if kwp <= 0:
            kwp = float(a.get("n_panels", 0)) * float(pv.get("wpp", st.session_state["wp_panel_global"])) / 1000.0

        loss = float(pv.get("system_loss", st.session_state["system_loss_global"]))
        tech = str(pv.get("tech", "crystSi"))

        key = f"a{i}_y{year}-{year}_kwp{kwp:.3f}_loss{loss:.1f}_tilt{tilt:.1f}_az{az:.1f}_tech{tech}"
        df = cache_pvgis_load(SESSION_DIR, editing["id"], key)

        if df is None or df.empty or "P" not in df.columns:
            continue

        df_full = df[df.index.year == year]
        if df_full.empty:
            continue

        step_hours = _median_step_hours(df_full.index)

        e_kwh_full = (df_full["P"] * step_hours / 1000.0).rename(f"Area_{i+1}")
        series_full = e_kwh_full.resample("H").sum(min_count=1) if rule == "H" else e_kwh_full.resample(rule).sum(min_count=1)
        dfs_download.append(series_full.to_frame())

        df_plot = df_full
        if rule == "H" and month is not None:
            df_plot = df_full[df_full.index.month == month]
        if not df_plot.empty:
            e_kwh_plot = (df_plot["P"] * step_hours / 1000.0).rename(f"Area_{i+1}")
            series_plot = e_kwh_plot.resample("H").sum(min_count=1) if rule == "H" else e_kwh_plot.resample(rule).sum(min_count=1)
            dfs_plot.append(series_plot.to_frame())

    if not dfs_download:
        st.warning("Nessun dato FV in cache per l'anno selezionato. Calcola prima la produzione FV nella sezione PVGIS.")
        pv_dl = None
        pv_plot = None
    else:
        pv_dl = pd.concat(dfs_download, axis=1).sort_index()
        pv_dl_total = pv_dl.sum(axis=1)

        pv_plot = None
        if dfs_plot:
            pv_plot = pd.concat(dfs_plot, axis=1).sort_index()
        else:
            pv_plot = pv_dl.copy()
        pv_plot_total = pv_plot.sum(axis=1)

    # ----------------------------
    # EOLICO: carica da cache
    # ----------------------------
    turbines = list(((editing.get("wind", {}) or {}).get("turbines") or []))
    wind_dl_total = None
    wind_plot_total = None
    if turbines:
        try:
            wkey = build_eolico_cache_key(PATHS.meteo_hourly_csv, turbines, output_unit="kwh")
            df_w = cache_eolico_load(SESSION_DIR, int(editing["id"]), wkey)
            if df_w is not None and (not df_w.empty) and ("Totale" in df_w.columns):
                df_wy = df_w[df_w.index.year == year]
                if not df_wy.empty:
                    wind_dl_total = df_wy["Totale"].resample("H").sum(min_count=1) if rule == "H" else df_wy["Totale"].resample(rule).sum(min_count=1)

                    df_wplot = df_wy
                    if rule == "H" and month is not None:
                        df_wplot = df_wy[df_wy.index.month == month]
                    if not df_wplot.empty:
                        wind_plot_total = df_wplot["Totale"].resample("H").sum(min_count=1) if rule == "H" else df_wplot["Totale"].resample(rule).sum(min_count=1)
        except Exception:
            wind_dl_total = None
            wind_plot_total = None

    if turbines and wind_dl_total is None:
        st.info("Nessun dato eolico in cache per i parametri attuali. Calcola prima l'eolico nella sezione dedicata.")

    # ----------------------------
    # Aggregazione (grafico + download)
    # ----------------------------
    if pv_plot is not None:
        # Manteniamo un grafico "pulito": solo FV aggregato, Eolico aggregato e Totale.
        fv_plot_total = pv_plot.sum(axis=1)

        if wind_plot_total is not None:
            wind_plot = wind_plot_total.reindex(pv_plot.index).fillna(0.0)
        else:
            wind_plot = pd.Series(0.0, index=pv_plot.index)

        dfAgg_plot = pd.DataFrame(index=pv_plot.index)
        # Naming richiesto per compatibilità a valle:
        # - Area_1 = FV aggregato
        # - Area_2 = Eolico aggregato
        dfAgg_plot["Area_1"] = fv_plot_total
        dfAgg_plot["Area_2"] = wind_plot
        dfAgg_plot["Totale"] = dfAgg_plot["Area_1"] + dfAgg_plot["Area_2"]

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=dfAgg_plot.index, y=dfAgg_plot["Area_1"], name="Area_1", mode="lines"))
            fig.add_trace(go.Scattergl(x=dfAgg_plot.index, y=dfAgg_plot["Area_2"], name="Area_2", mode="lines"))
            fig.add_trace(go.Scattergl(x=dfAgg_plot.index, y=dfAgg_plot["Totale"], name="Totale", mode="lines"))

            if rule == "H" and month is not None:
                titolo = f"Aggregazione {agg_choice.lower()} (kWh) — {calendar.month_name[month]} {year}"
            else:
                titolo = f"Aggregazione {agg_choice.lower()} (kWh) — {year}"

            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                title=titolo,
                xaxis_title="Data",
                yaxis_title="kWh",
                legend=dict(orientation="h", y=1.1),
            )
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
        else:
            st.line_chart(dfAgg_plot[["Area_1", "Area_2", "Totale"]])

    if pv_dl is not None:
        # Output richiesto:
        # - time (index)
        # - Area_1 = FV aggregato
        # - Area_2 = Eolico aggregato
        # - Totale = somma
        fv_dl_total = pv_dl.sum(axis=1)

        if wind_dl_total is not None:
            wind_dl = wind_dl_total.reindex(pv_dl.index).fillna(0.0)
        else:
            wind_dl = pd.Series(0.0, index=pv_dl.index)

        dfAgg_dl = pd.DataFrame(index=pv_dl.index)
        dfAgg_dl.index.name = "time"
        dfAgg_dl["Area_1"] = fv_dl_total
        dfAgg_dl["Area_2"] = wind_dl
        dfAgg_dl["Totale"] = dfAgg_dl["Area_1"] + dfAgg_dl["Area_2"]

        # aggiorna kWh annui totali del produttore (per tabella lista produttori)
        try:
            editing["annual_kwh"] = float(dfAgg_dl["Totale"].sum())
            for idx, pr in enumerate(st.session_state.get("producers", [])):
                if pr.get("id") == editing.get("id"):
                    st.session_state["producers"][idx]["annual_kwh"] = float(editing["annual_kwh"])
                    break
        except Exception:
            pass

        csv = dfAgg_dl.to_csv(index=True).encode("utf-8")
        st.download_button(
            f"⬇️ Aggregazione {agg_choice.lower()} (CSV, wide) — Area_1(FV) + Area_2(Eolico) + Totale",
            data=csv,
            file_name=f"{editing['name'].replace(' ', '_')}_AGG_{rule}_wide_{year}_FULLYEAR.csv",
            mime="text/csv",
            key=f"agg_dl_{editing['id']}_{rule}_{year}",
        )



# ==============================================================================
# Sezione: Save / Cancel
# ==============================================================================

def _render_save_cancel(editing: Producer) -> None:
    cS1, cS2 = st.columns(2)

    if cS1.button("💾 Salva produttore"):
        # assicurati che i parametri FV (dimensioni, Wp, loss, tecnologia) siano nel JSON
        _persist_pv_from_form_into_editing(editing)

        exists = False
        for idx, pr in enumerate(st.session_state["producers"]):
            if pr["id"] == editing["id"]:
                st.session_state["producers"][idx] = editing
                exists = True
                break

        if not exists:
            st.session_state["producers"].append(editing)

        save_producers(SESSION_DIR, st.session_state["producers"])

        st.session_state["mode"] = "list"
        st.session_state["editing_obj"] = None
        st.success("Produttore salvato in locale.")
        st.rerun()

    if cS2.button("↩️ Annulla"):
        st.session_state["mode"] = "list"
        st.session_state["editing_obj"] = None
        st.rerun()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    # 1) reset on session change
    _reset_state_if_session_changed(SESSION_DIR)

    # 2) init session_state defaults
    _init_session_state()

    st.title("🔆 Produttori")

    # ---- LISTA + AZIONI ----
    col_table, col_actions = st.columns([3.2, 1.1])

    with col_table:
        prods: list[Producer] = st.session_state.get("producers", [])
        if not prods:
            st.info("Nessun produttore. Usa i pulsanti a destra per crearne uno.")
        else:
            df = _producer_table_rows(prods)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with col_actions:
        if st.button("➕ Aggiungi nuovo", use_container_width=True):
            new_id = (max([p["id"] for p in st.session_state["producers"]] or [0]) + 1)
            _enter_edit_mode_new_producer(new_id)
            st.rerun()

        ids = [p["id"] for p in st.session_state["producers"]]
        if ids:
            sel_id = st.selectbox("Modifica ID", options=ids, index=0, key="edit_pick")
            if st.button("✏️ Modifica", use_container_width=True):
                for p in st.session_state["producers"]:
                    if p["id"] == sel_id:
                        _enter_edit_mode_for_producer(p)
                        break
                st.rerun()

            del_id = st.selectbox("Elimina ID", options=ids, index=0, key="del_pick")
            if st.button("🗑️ Elimina", type="secondary", use_container_width=True):
                st.session_state["producers"] = [p for p in st.session_state["producers"] if p["id"] != del_id]
                save_producers(SESSION_DIR, st.session_state["producers"])
                _clear_transient_visual_state()
                st.success(f"Produttore {del_id} eliminato.")
                st.rerun()

    st.markdown("---")

    # ---- EDIT MODE ----
    editing = _require_editing()

    _render_area_editor(editing)
    st.markdown("---")

    _render_roof_3d(editing)
    st.markdown("---")

    _render_mask_section(editing)
    st.markdown("---")

    _render_panel_placement_3d(editing)
    st.markdown("---")

    _render_pvgis(editing)
    st.markdown("---")

    _render_eolico(editing)
    st.markdown("---")

    _render_aggregazione_pv_eolico(editing)
    st.markdown("---")

    _render_save_cancel(editing)


# Streamlit entry-point
main()
