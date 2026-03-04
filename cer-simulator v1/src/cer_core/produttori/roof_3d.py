"""
cer_core.produttori.roof_3d
==========================

Costruzione di mesh 3D (tetto e pannelli) per la visualizzazione in Plotly.

Questo modulo **non** calcola la produzione energetica né il posizionamento dei
pannelli (packing): il suo compito è convertire le aree di tetto (poligoni) e
l'insieme di pannelli (rettangoli) in una rappresentazione triangolata 3D
utilizzabile da `plotly.graph_objects.Mesh3d`.

Input e convenzioni
-------------------
- Le geometrie in ingresso sono in EPSG:4326 (lon/lat) e arrivano tipicamente da
  `producers.json` (campo `areas[*].geom`) o dai risultati del packing
  (`areas[*].panels_geojson`).
- Le operazioni geometriche metriche (triangolazione, calcolo pendenze) vengono
  svolte in un sistema UTM locale (metri) derivato dal centro delle aree tramite
  `auto_utm_crs(lat, lon)`.

Modello 3D (semplificazione)
----------------------------
Ogni area di tetto viene modellata come **falda piana**: la quota z cresce in
modo lineare lungo una direzione orizzontale `u`, determinata dall'angolo
`angle_deg`. La pendenza è pari a `tan(tilt_deg)`.

Convenzione angolare del progetto
---------------------------------
La convenzione (coerente con il resto del progetto UI/core) è:

- 0°  = Sud
- 90° = Est
- 180°= Nord
- 270°= Ovest

API pubblica
------------
- `local_transformers_for_areas(areas_ll) -> (fwd, inv)`
- `roof_mesh_from_areas(areas, fwd=None, inv=None) -> {"meshes": [...] }`
- `panels_mesh_from_area(area, panels_geo, fwd=None, inv=None) -> {x,y,z,i,j,k}`

Nota
----
Le mesh prodotte sono pensate per **rendering**: l'obiettivo è avere una
rappresentazione visiva stabile e coerente, non un modello strutturale
ingegneristico del tetto.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Tuple, TypedDict, Optional

from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import triangulate

from .produttori import auto_utm_crs

# ---------------------------------------------------------------------------
# Tipi (solo per chiarezza/documentazione; non vincolanti a runtime)
# ---------------------------------------------------------------------------

TransformFn = Callable[[float, float], Tuple[float, float]]


class ProducerArea(TypedDict, total=False):
    """Struttura minima attesa per un'area di tetto (subset di producers.json)."""

    geom: Dict[str, Any]  # GeoJSON Polygon in EPSG:4326
    angle_deg: float  # convenzione progetto: 0=S, 90=E, 180=N, 270=W
    tilt_deg: float  # inclinazione falda in gradi


# ---------------------------------------------------------------------------
# Trasformazioni locali (UTM)
# ---------------------------------------------------------------------------

def _local_crs_from_areas(areas_ll: Sequence[ProducerArea]) -> Tuple[TransformFn, TransformFn]:
    """
    Calcola le trasformazioni lon/lat ↔ UTM locale per un insieme di aree.

    Strategia:
    - Si calcola un punto rappresentativo per ciascuna area (Shapely
      `representative_point()`).
    - Si media (lon, lat) dei representative point per ottenere un "centro"
      robusto anche per aree concave.
    - Si ricava l'EPSG UTM coerente con quel centro (`auto_utm_crs`).

    Parametri
    ---------
    areas_ll:
        Sequenza di aree con `geom` in EPSG:4326.

    Ritorna
    -------
    (fwd, inv):
        - `fwd(lon, lat) -> (x_m, y_m)`  in UTM (metri)
        - `inv(x_m, y_m) -> (lon, lat)`  in EPSG:4326
    """
    xs: List[float] = []
    ys: List[float] = []

    for a in areas_ll:
        g = shape(a["geom"])
        rp = g.representative_point()
        xs.append(float(rp.x))
        ys.append(float(rp.y))

    lon = sum(xs) / max(1, len(xs))
    lat = sum(ys) / max(1, len(ys))

    epsg = auto_utm_crs(lat, lon)
    fwd: TransformFn = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv: TransformFn = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    return fwd, inv


def local_transformers_for_areas(areas_ll: Sequence[ProducerArea]) -> Tuple[TransformFn, TransformFn]:
    """
    Alias pubblico di `_local_crs_from_areas`.

    È la funzione usata dalla UI (pagina Produttori) per ottenere una proiezione
    metrica stabile con cui costruire la mesh.
    """
    return _local_crs_from_areas(areas_ll)


# ---------------------------------------------------------------------------
# Geometria della falda (direzione e quota)
# ---------------------------------------------------------------------------

def _dir_unit_from_angle_deg(angle_deg: float) -> Tuple[float, float]:
    """
    Converte `angle_deg` (convenzione progetto) in un versore 2D (ux, uy) in metri.

    Convenzione angoli:
    - 0° = Sud  -> u = (0, +1) in coordinate UTM (y verso Nord, quindi Sud significa
                 che la quota cresce verso Sud: qui si applica la convenzione
                 storica del progetto; il segno è gestito coerentemente.)
    - 90° = Est -> u = (-1, 0)
    - 180° = Nord -> u = (0, -1)
    - 270° = Ovest -> u = (+1, 0)

    Nota: la trasformazione adottata replica esattamente il comportamento
    pre-esistente (sin/cos con segni specifici).
    """
    a = math.radians(angle_deg % 360.0)
    ux = -math.sin(a)
    uy = math.cos(a)
    n = (ux * ux + uy * uy) ** 0.5
    return (ux / n, uy / n) if n > 0 else (1.0, 0.0)


def _plane_reference_projection(poly_m: Polygon, u: Tuple[float, float]) -> float:
    """
    Ritorna p0, la proiezione minima dei vertici lungo `u`.

    p0 viene usato come riferimento per avere z=0 sul bordo "più basso" della falda.
    """
    return min((x * u[0] + y * u[1]) for (x, y) in poly_m.exterior.coords)


# ---------------------------------------------------------------------------
# Mesh tetto
# ---------------------------------------------------------------------------

def roof_mesh_from_areas(
    areas: Sequence[ProducerArea],
    fwd: Optional[TransformFn] = None,
    inv: Optional[TransformFn] = None,
) -> Dict[str, Any]:
    """
    Costruisce una mesh 3D (triangolata) per ciascuna area di tetto.

    Parametri
    ---------
    areas:
        Sequenza di aree con almeno:
        - `geom` GeoJSON Polygon in EPSG:4326
        - `angle_deg` (opzionale; default 0.0)
        - `tilt_deg`  (opzionale; default 0.0)
    fwd, inv:
        Trasformazioni lon/lat ↔ UTM (metri). Se non fornite vengono calcolate
        automaticamente dalle aree.

    Ritorna
    -------
    dict:
        {"meshes": [mesh_area_0, mesh_area_1, ...]}

        dove ciascun `mesh_area` è:
        - x, y, z: liste di coordinate dei vertici (in metri UTM)
        - faces: lista di triangoli come tuple di indici (i, j, k)

    Note implementative
    -------------------
    - La triangolazione usa `shapely.ops.triangulate(poly)`.
    - Le quote z sono assegnate dopo la deduplicazione dei vertici, per garantire
      coerenza tra triangoli adiacenti.
    - Per deduplicare si arrotonda (x, y) a 6 decimali: scelta pensata per
      ridurre vertici quasi coincidenti a causa di floating point, e per
      stabilizzare la mesh in Plotly.
    """
    if not areas:
        return {"meshes": []}

    if fwd is None or inv is None:
        fwd, inv = _local_crs_from_areas(areas)

    meshes: List[Dict[str, Any]] = []

    for a in areas:
        poly_ll = shape(a["geom"])
        poly_m = Polygon([fwd(x, y) for (x, y) in poly_ll.exterior.coords])

        # Triangolazione in metri.
        tris = triangulate(poly_m)

        ang = float(a.get("angle_deg", 0.0))
        tilt = float(a.get("tilt_deg", 0.0))
        u = _dir_unit_from_angle_deg(ang)
        slope = math.tan(math.radians(tilt))
        p0 = _plane_reference_projection(poly_m, u)

        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        faces: List[Tuple[int, int, int]] = []
        vidx: Dict[Tuple[float, float], int] = {}

        def add_v(x: float, y: float) -> int:
            key = (round(x, 6), round(y, 6))
            if key in vidx:
                return vidx[key]
            i = len(xs)
            xs.append(x)
            ys.append(y)
            zs.append(0.0)  # valorizzato successivamente
            vidx[key] = i
            return i

        for t in tris:
            coords = list(t.exterior.coords)[:-1]
            if len(coords) != 3:
                continue
            idx = [add_v(float(x), float(y)) for (x, y) in coords]
            faces.append((idx[0], idx[1], idx[2]))

        # Assegna quota ai vertici: z = slope * (proj - p0)
        for i in range(len(xs)):
            proj = xs[i] * u[0] + ys[i] * u[1]
            zs[i] = slope * (proj - p0)

        meshes.append({"x": xs, "y": ys, "z": zs, "faces": faces})

    return {"meshes": meshes}


# ---------------------------------------------------------------------------
# Mesh pannelli
# ---------------------------------------------------------------------------

def panels_mesh_from_area(
    area: ProducerArea,
    panels_geo: Sequence[Polygon],
    fwd: Optional[TransformFn] = None,
    inv: Optional[TransformFn] = None,
) -> Dict[str, Any]:
    """
    Costruisce la mesh 3D dei pannelli di una singola area.

    Parametri
    ---------
    area:
        Area che contiene `geom` + (opzionale) `angle_deg`, `tilt_deg`.
    panels_geo:
        Sequenza di Poligoni Shapely (rettangoli) in EPSG:4326.
        Tipicamente ottenuti dal packing (`panels_geojson` convertito a Shapely).
    fwd, inv:
        Trasformazioni lon/lat ↔ UTM. Se non fornite vengono calcolate dalla sola
        area (UTM locale centrata sull'area stessa).

    Ritorna
    -------
    dict compatibile con Plotly Mesh3d:
        {"x": X, "y": Y, "z": Z, "i": I, "j": J, "k": K}

    dove (i, j, k) definiscono i triangoli; ogni pannello viene triangolato in
    due facce: (0,1,2) e (0,2,3).
    """
    poly_ll = shape(area["geom"])
    ang = float(area.get("angle_deg", 0.0))
    tilt = float(area.get("tilt_deg", 0.0))

    if fwd is None or inv is None:
        fwd, inv = _local_crs_from_areas([area])

    poly_m = Polygon([fwd(x, y) for (x, y) in poly_ll.exterior.coords])

    u = _dir_unit_from_angle_deg(ang)
    slope = math.tan(math.radians(tilt))
    p0 = _plane_reference_projection(poly_m, u)

    X: List[float] = []
    Y: List[float] = []
    Z: List[float] = []
    I: List[int] = []
    J: List[int] = []
    K: List[int] = []
    vidx: Dict[Tuple[float, float], int] = {}

    def add_v(x: float, y: float) -> int:
        key = (round(x, 6), round(y, 6))
        if key in vidx:
            return vidx[key]
        i = len(X)
        X.append(x)
        Y.append(y)
        Z.append(slope * ((x * u[0] + y * u[1]) - p0))
        vidx[key] = i
        return i

    for pg in panels_geo:
        # Nota: i pannelli sono rettangoli; l'exterior include il punto di chiusura,
        # qui utilizziamo i primi 4 vertici.
        coords_m = [fwd(x, y) for (x, y) in list(pg.exterior.coords)]
        if len(coords_m) < 4:
            continue
        idx = [add_v(float(coords_m[k][0]), float(coords_m[k][1])) for k in range(4)]

        # Due triangoli: (0,1,2) e (0,2,3)
        I += [idx[0], idx[0]]
        J += [idx[1], idx[2]]
        K += [idx[2], idx[3]]

    return {"x": X, "y": Y, "z": Z, "i": I, "j": J, "k": K}
