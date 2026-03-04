"""cer_core.produttori.laf_packing

Packing di pannelli fotovoltaici su poligoni di tetto tramite **Local Area Frame (LAF)**.

Contesto e obiettivo
--------------------
La pagina Streamlit dei produttori consente all'utente di disegnare (in mappa) una o più
aree di tetto come poligoni in coordinate geografiche (WGS84 / EPSG:4326). Per stimare
un numero realistico di pannelli installabili, è necessario eseguire un posizionamento
geometrico in **metri**, evitando di lavorare direttamente in gradi (lon/lat).

Questo modulo implementa un algoritmo semplice ma robusto:

1) Proietta il poligono lon/lat in un sistema metrico locale (UTM calcolato automaticamente).
2) Costruisce un riferimento locale 2D (LAF) con origine vicino al poligono e assi ruotati
   in funzione dell'angolo scelto dall'utente (convenzione progetto: 0=S, 90=E, 180=N, 270=W).
3) Esegue un packing su griglia di rettangoli (pannelli) **allineati agli assi del frame**,
   con passo determinato da dimensioni pannello + gap.
4) Applica vincoli geometrici:
   - margine interno al bordo (buffer negativo)
   - soglia di "coverage" (percentuale minima di area del pannello che deve cadere dentro l'area utile)
5) Converte i pannelli risultanti di nuovo in lon/lat (EPSG:4326) per persistenza/visualizzazione.

Nota sulle convenzioni d'angolo
-------------------------------
L'angolo `angle_deg` segue la convenzione del progetto (0°=Sud, 90°=Est, 180°=Nord, 270°=Ovest).
Nel LAF l'angolo è utilizzato per definire **l'asse** a cui allineare la griglia; la direzione
del vettore è ininfluente fino a un segno (u e -u definiscono lo stesso asse).

API pubblica (usata dalla UI)
-----------------------------
- `pack_panels_laf_lonlat(...)`: wrapper completo (lon/lat -> LAF -> packing -> lon/lat)
- `build_laf(...)`: costruzione del frame (utile per debug/estensioni)
- `pack_local_grid(...)`: packing nel frame locale (metri)

Invarianti
----------
- Input poligoni: EPSG:4326 (lon, lat) con ordine (x=lon, y=lat).
- Packing: eseguito in metri in UTM locale.
- Output pannelli: lista di `shapely.geometry.Polygon` in EPSG:4326.

"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

from pyproj import Transformer
from shapely.geometry import Polygon, box

from .produttori import auto_utm_crs

# --- Type aliases (documentativi) -------------------------------------------------

Vec2 = Tuple[float, float]
Rot2 = Tuple[Tuple[float, float], Tuple[float, float]]
TransformFn = Callable[[float, float], Tuple[float, float]]


# --- Algebra 2D minimale ----------------------------------------------------------

def _axis_unit_from_angle(angle_deg: float) -> Vec2:
    """Ritorna un versore nel piano (E,N) associato a `angle_deg`.

    `angle_deg` usa la convenzione del progetto: 0=S, 90=E, 180=N, 270=W.

    Nota: il packing richiede l'allineamento della griglia ad un asse; il verso del versore
    è quindi irrilevante (u e -u definiscono lo stesso asse). L'implementazione restituisce
    un versore deterministico coerente con la pipeline esistente.

    Parameters
    ----------
    angle_deg:
        Angolo in gradi secondo convenzione progetto.

    Returns
    -------
    Vec2:
        Versore (ux, uy) nel sistema metrico (x=Est, y=Nord).
    """

    a = math.radians(angle_deg % 360.0)
    # Convenzione interna storica: mantiene piena compatibilità con i risultati esistenti.
    return (-math.sin(a), math.cos(a))


def _rotation(theta: float) -> Rot2:
    """Matrice di rotazione 2D (theta in radianti, CCW)."""
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s), (s, c))


def _mat_mul_vec(R: Rot2, v: Vec2) -> Vec2:
    """Prodotto R @ v per matrice 2x2 e vettore 2D."""
    return (R[0][0] * v[0] + R[0][1] * v[1], R[1][0] * v[0] + R[1][1] * v[1])


def _vec_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _vec_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


# --- Local Area Frame -------------------------------------------------------------

def build_laf(poly_ll: Polygon, angle_deg: float):
    """Costruisce un Local Area Frame (LAF) esplicito per un poligono lon/lat.

    Il LAF è un riferimento locale 2D in metri (UTM) con:
    - origine C: punto di ancoraggio vicino al poligono (media dei vertici dell'anello esterno)
    - asse +X allineato all'asse associato all'angolo `angle_deg`

    La funzione ritorna anche trasformazioni (callable) per passare:
    - lon/lat -> UTM (fwd)
    - UTM -> lon/lat (inv)
    - world(UTM) -> local(LAF) (to_local)
    - local(LAF) -> world(UTM) (to_world)

    Parameters
    ----------
    poly_ll:
        Poligono `shapely` in EPSG:4326 (lon/lat).
    angle_deg:
        Angolo in gradi secondo convenzione progetto (0=S, 90=E, 180=N, 270=W).

    Returns
    -------
    tuple:
        (fwd, inv, C, R, Rt, poly_m, poly_local, to_local, to_world)

        - fwd / inv : transformer callable
        - C : (Cx, Cy) in metri (UTM)
        - R : matrice world->local
        - Rt: matrice local->world (trasposta/inversa per rotazioni pure)
        - poly_m: poligono in metri (UTM)
        - poly_local: poligono in coordinate locali (LAF)
        - to_local / to_world: callable di conversione punti

    Note
    ----
    La scelta di C come media dei vertici (anziché `poly_m.centroid`) è mantenuta per
    compatibilità con i risultati esistenti e per ridurre dipendenze da dettagli geometrici
    (es. concavità) che possono influenzare il posizionamento della griglia.
    """

    # UTM locale scelto usando un punto garantito interno (representative point)
    rp = poly_ll.representative_point()
    epsg = auto_utm_crs(rp.y, rp.x)

    fwd: TransformFn = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv: TransformFn = Transformer.from_crs(epsg, 4326, always_xy=True).transform

    # Proiezione dell'anello esterno in metri
    coords_m: List[Vec2] = [fwd(x, y) for (x, y) in poly_ll.exterior.coords]
    poly_m = Polygon(coords_m)

    # Punto di ancoraggio: media dei vertici (escluso il punto di chiusura)
    Cx = sum(p[0] for p in coords_m[:-1]) / max(1, len(coords_m) - 1)
    Cy = sum(p[1] for p in coords_m[:-1]) / max(1, len(coords_m) - 1)
    C: Vec2 = (Cx, Cy)

    # Asse unitario e rotazione: allinea l'asse a +X del frame locale
    ux, uy = _axis_unit_from_angle(angle_deg)
    theta = math.atan2(uy, ux)   # angolo (da +X) dell'asse nel sistema metrico UTM
    R = _rotation(-theta)        # world -> local
    Rt = _rotation(theta)        # local -> world (inversa per rotazioni pure)

    # Conversioni world<->local
    def to_local(pt_world: Vec2) -> Vec2:
        # (x', y') = R @ ( (x, y) - C )
        return _mat_mul_vec(R, _vec_sub(pt_world, C))

    def to_world(pt_local: Vec2) -> Vec2:
        # (x, y) = Rt @ (x', y') + C
        return _vec_add(_mat_mul_vec(Rt, pt_local), C)

    poly_local = Polygon([to_local(p) for p in coords_m])
    return fwd, inv, C, R, Rt, poly_m, poly_local, to_local, to_world


# --- Packing 2D nel frame locale --------------------------------------------------

def pack_local_grid(
    poly_local: Polygon,
    panel_w: float,
    panel_h: float,
    gap: float,
    margin: float,
    coverage: float,
    stagger: bool,
) -> List[Polygon]:
    """Posizionamento di pannelli come rettangoli axis-aligned in coordinate locali.

    Il poligono `poly_local` è già in metri e già ruotato nel LAF; il packing si riduce quindi
    ad un problema 2D di inserimento rettangoli su griglia regolare.

    Algoritmo (deterministico):
    - (opzionale) applica un margine interno con `buffer(-margin)`
    - scansiona una griglia (x, y) dal bounding box inferiore sinistro
    - ad ogni passo valuta il candidato rettangolo
    - accetta il pannello se la sua intersezione con l'area utile copre almeno
      `coverage * area_pannello`
    - (opzionale) staggering: alterna l'offset in x per righe dispari

    Parameters
    ----------
    poly_local:
        Area utile in coordinate locali (metri).
    panel_w, panel_h:
        Dimensioni del pannello in metri.
    gap:
        Distanza minima tra pannelli (metri).
    margin:
        Margine interno dal bordo del poligono (metri). Se > 0, applica un buffer negativo.
    coverage:
        Soglia [0..1]. 1.0 richiede pannello completamente interno. Valori tipici 0.95.
    stagger:
        Se True, sfalsa le righe dispari di mezzo pannello in x.

    Returns
    -------
    list[Polygon]:
        Lista di rettangoli (in coordinate locali) corrispondenti ai pannelli posati.
    """

    # Margine interno: attenzione, buffer può produrre geometrie vuote o non-poligonali;
    # la pipeline esistente assume un risultato compatibile con bounds/intersection.
    if margin and margin > 1e-9:
        poly_local = poly_local.buffer(-margin)

    step_x = float(panel_w) + float(gap)
    step_y = float(panel_h) + float(gap)

    minx, miny, maxx, maxy = poly_local.bounds

    panels: List[Polygon] = []
    row = 0
    y = miny
    while y + panel_h <= maxy + 1e-6:
        x_offset = (panel_w / 2.0) if (stagger and (row % 2 == 1)) else 0.0
        x = minx + x_offset

        while x + panel_w <= maxx + 1e-6:
            cand = box(x, y, x + panel_w, y + panel_h)

            inter = cand.intersection(poly_local)
            if not inter.is_empty and inter.area >= coverage * cand.area:
                panels.append(cand)

            x += step_x

        y += step_y
        row += 1

    return panels


def pack_panels_laf_lonlat(
    poly_ll: Polygon,
    angle_deg: float,
    panel_w: float,
    panel_h: float,
    gap: float,
    margin: float = 0.0,
    coverage: float = 0.95,
    stagger: bool = False,
) -> Tuple[list, dict]:
    """Wrapper completo: LAF + packing + riconversione in lon/lat.

    Questa è la funzione utilizzata dalla pagina Streamlit `1_Producers.py`.

    Strategia:
    - costruisce un LAF (lon/lat -> UTM -> local)
    - esegue il packing su griglia nel frame locale
    - prova entrambe le orientazioni del pannello (w×h e h×w) e sceglie la migliore
      (massimizza il numero di pannelli)
    - riconverte i pannelli in EPSG:4326

    Parameters
    ----------
    poly_ll:
        Poligono area tetto in EPSG:4326 (lon/lat).
    angle_deg:
        Angolo in gradi secondo convenzione progetto (0=S, 90=E, 180=N, 270=W).
    panel_w, panel_h:
        Dimensioni del pannello in metri.
    gap:
        Distanza minima tra pannelli in metri.
    margin:
        Margine interno dal bordo del poligono in metri.
    coverage:
        Frazione minima di area del pannello che deve ricadere nell'area utile.
    stagger:
        Se True, sfalsa le righe.

    Returns
    -------
    (panels_ll, aux):
        - panels_ll: list[Polygon] in EPSG:4326 (lon/lat)
        - aux: dizionario con trasformazioni e geometrie intermedie (utile per debug/3D)
    """

    fwd, inv, C, R, Rt, poly_m, poly_local, to_local, to_world = build_laf(poly_ll, angle_deg)

    # Prova entrambe le orientazioni e mantiene quella con più pannelli
    A = pack_local_grid(poly_local, panel_w, panel_h, gap, margin, coverage, stagger)
    B = pack_local_grid(poly_local, panel_h, panel_w, gap, margin, coverage, stagger)
    panels_local = A if len(A) >= len(B) else B

    # Riconversione: local -> world(UTM) -> lon/lat
    panels_ll: List[Polygon] = []
    for p in panels_local:
        coords_world = [to_world((x, y)) for (x, y) in p.exterior.coords]
        coords_ll = [inv(x, y) for (x, y) in coords_world]
        panels_ll.append(Polygon(coords_ll))

    aux = {
        "fwd": fwd,
        "inv": inv,
        "C": C,
        "R": R,
        "Rt": Rt,
        "poly_m": poly_m,
        "poly_local": poly_local,
        "to_local": to_local,
        "to_world": to_world,
    }

    return panels_ll, aux
