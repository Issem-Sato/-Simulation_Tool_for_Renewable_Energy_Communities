from __future__ import annotations

"""
lavanderia.py – Modello lavatrice + asciugatrice per il simulatore CER.

Questo modulo è pensato come “motore ufficiale” per i carichi di lavanderia.
È auto-contenuto e può essere richiamato sia dalla UI Streamlit sia da script
offline.

Concetto chiave:
- L'utente definisce:
    * finestre temporali di utilizzo (LaundryWindows)
    * configurazione lavatrice (WasherConfig)
    * configurazione asciugatrice (DryerConfig)
  riunite in un oggetto LaundryConfig.

- A partire da questi parametri, vengono generati eventi di utilizzo
  (timestamp + modalità) e le relative curve di potenza (kW) su un DatetimeIndex (tipicamente a 15 minuti in UTC, ma compatibile anche con griglie orarie).

- Il modulo espone anche dei wrapper di comodo:
    simulate_washer(...)
    simulate_dryer(...)
    simulate("washer" / "dryer", ...)


Assunzioni e contratti (importanti per la compatibilità con l'interfaccia):
- Timebase: l'indice temporale è fornito dal chiamante ed è trattato come equispaziato.
  Nel simulatore CER l'indice è normalmente a 15 minuti e timezone-aware in UTC.
- Unità: le serie restituite sono potenze istantanee/medie per step in **kW**.
  L'energia (kWh) si ottiene a valle con: sum(P_kW) * dt_hours.
- Riproducibilità: tutte le scelte casuali dipendono esclusivamente dal parametro `seed`.
- Gestione settimane: i cicli sono campionati su settimane *complete* di 7 giorni contigue
  a partire da `index.min().normalize()` (non è richiesto l'allineamento al calendario).
  Le settimane parziali a inizio/fine simulazione non generano cicli.
- Failure modes: se le finestre disponibili non consentono di allocare `cycles_per_week`
  (dopo arrotondamento a intero), viene sollevato `ValueError` per segnalare una configurazione incoerente.
  compatibili con l'uso nella pagina Streamlit 2_Consumatori.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Optional, Literal, Tuple
import numpy as np
import pandas as pd

DayIndex = int                # 0=Monday ... 6=Sunday
SlotLabel = str               # e.g. "18-20"
ModeName = str                # "eco" | "standard" | "intensivo"

# ---------------------------------------------------------------------------
# Costanti condivise con la UI
# ---------------------------------------------------------------------------

DAY_NAMES: List[str] = [
    "Lunedì",
    "Martedì",
    "Mercoledì",
    "Giovedì",
    "Venerdì",
    "Sabato",
    "Domenica",
]

# Etichette fascia oraria e banda [h_start, h_end)
SLOT_LABELS: List[SlotLabel] = [
    "06-08",
    "08-10",
    "10-12",
    "12-14",
    "14-16",
    "16-18",
    "18-20",
    "20-22",
    "22-24",
    "00-06",
]

SLOT_BANDS: List[Tuple[int, int]] = [
    (6, 8),
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 22),
    (22, 24),
    (0, 6),
]

ENERGY_CLASS_FACTOR = {
    None: 1.00,
    "A": 0.85,
    "B": 0.95,
    "C": 1.05,
    "D": 1.15,
}

# Modalità di lavaggio considerate nel modello
LAUNDRY_MODES: List[ModeName] = ["eco", "standard", "intensivo", "rapido"]

# Opzioni per la selezione in UI (stesse semantiche di cer_appliances)
SEASONALITY_OPTIONS: List[str] = ["tutto_anno", "estate", "inverno"]
ENERGY_CLASS_OPTIONS: List[Optional[str]] = [None, "A", "B", "C", "D"]
WASHER_MODE_OPTIONS: List[str] = list(LAUNDRY_MODES)

# ---------------------------------------------------------------------------
# Funzione di utilità: normalizzazione griglia fasce orarie
# (equivalente a cer_appliances.parse_start_matrix)
# ---------------------------------------------------------------------------

def parse_start_matrix(
    matrix: Dict[str, List[str]] | List[List[bool]] | None
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Normalizza la rappresentazione delle finestre di utilizzo.

    Input accettati:
    - None -> nessuna finestra (tutti i giorni vuoti); la UI dovrebbe fornire una griglia valida
    - lista 7x10 di bool (griglia checkbox UI)
    - dict:
        * chiavi "0".."6" o int 0..6
        * oppure "lun","mar","...","feriali","festivi"
        * valori: lista di bande (h1,h2) o stringhe "18-20"

    Output:
    - dict: day_index (0..6) -> lista di bande (h_start, h_end)
    """
    out: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(7)}

    # Caso: lista 7x10 di bool
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        for i in range(min(7, len(matrix))):
            row = matrix[i][: len(SLOT_BANDS)]
            day_slots: List[Tuple[int, int]] = []
            for j, checked in enumerate(row):
                if bool(checked):
                    day_slots.append(SLOT_BANDS[j])
            out[i] = day_slots
        return out

    # Caso: dict
    if isinstance(matrix, dict):

        def _day_to_idx(k):
            # 1) int 0..6
            if isinstance(k, int):
                return min(max(k, 0), 6)
            sk = str(k).strip()
            # 2) numerico "0".."6"
            if sk.isdigit():
                di = int(sk)
                return min(max(di, 0), 6)
            # 3) abbreviazioni e feriali/festivi
            kk = sk.lower()[:3]
            m = {"lun": 0, "mar": 1, "mer": 2, "gio": 3, "ven": 4, "sab": 5, "dom": 6}
            if kk in ("fer",):
                return -1
            if kk in ("fes",):
                return -2
            return m.get(kk, 0)

        tmp: Dict[int, List[Tuple[int, int]]] = {}
        for k, v in matrix.items():
            di = _day_to_idx(k)
            bands: List[Tuple[int, int]] = []
            if isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        bands.append((int(item[0]), int(item[1])))
                    else:
                        s = str(item).strip()
                        if "-" in s:
                            h1, h2 = s.split("-")
                        elif "," in s:
                            s2 = s.strip("()[]{} ")
                            parts = s2.split(",")
                            if len(parts) != 2:
                                continue
                            h1, h2 = parts
                        else:
                            continue
                        bands.append((int(h1), int(h2)))
            tmp[di] = bands

        for i in range(7):
            out[i] = tmp.get(i, [])
        # feriali
        if -1 in tmp:
            for i in range(5):
                out[i] = tmp[-1]
        # festivi
        if -2 in tmp:
            for i in (5, 6):
                out[i] = tmp[-2]

        return out

    # fallback totale

    return out


# ---------------------------------------------------------------------------
# Dataclasses di configurazione
# ---------------------------------------------------------------------------

@dataclass
class LaundryWindows:
    """Finestre giornaliere in cui l'utente *potrebbe* usare la lavanderia.

    mapping:
        day_index (0=Mon) -> lista di SLOT_LABELS selezionate.
    """

    windows: Dict[DayIndex, List[SlotLabel]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "LaundryWindows":
        return cls({i: [] for i in range(7)})

    @classmethod
    def from_dict(cls, data: Dict[str, List[SlotLabel]]) -> "LaundryWindows":
        # chiavi stringhe -> int
        out: Dict[int, List[SlotLabel]] = {}
        for k, v in data.items():
            try:
                di = int(k)
            except (TypeError, ValueError):
                continue
            out[di] = [s for s in v if s in SLOT_LABELS]
        return cls(out)

    def to_bands(self) -> Dict[int, List[Tuple[int, int]]]:
        """Converte le etichette in bande orarie (h_start, h_end)."""
        band_by_label: Dict[SlotLabel, Tuple[int, int]] = dict(
            zip(SLOT_LABELS, SLOT_BANDS)
        )
        out: Dict[int, List[Tuple[int, int]]] = {}
        for di in range(7):
            labels = self.windows.get(di, [])
            bands: List[Tuple[int, int]] = []
            for lab in labels:
                b = band_by_label.get(lab)
                if b:
                    bands.append(b)
            out[di] = bands
        return out


@dataclass
class WasherConfig:
    has_washer: bool = True
    cycles_per_week: float = 3.0
    power_kw: Optional[float] = None          # override utente; se None si stima
    mode_weights: Dict[ModeName, float] = field(
        default_factory=lambda: {"eco": 0.6, "standard": 0.4}
    )

    def normalised_mode_probs(self) -> Dict[ModeName, float]:
        w = {m: max(self.mode_weights.get(m, 0.0), 0.0) for m in LAUNDRY_MODES}
        s = sum(w.values())
        if s <= 0:
            w = {m: 1.0 for m in LAUNDRY_MODES}
            s = sum(w.values())
        return {m: w[m] / s for m in LAUNDRY_MODES}


DryerLink = Literal["after_wash", "same_day", "independent"]


@dataclass
class DryerConfig:
    has_dryer: bool = False
    type: str = "pompa_di_calore"                 # solo a scopo descrittivo
    power_kw: Optional[float] = None
    cycles_per_week: float = 0.0
    seasonality: str = "tutto_anno"
    ratio_dry_per_wash: float = 1.0               # numero di asciugature per lavaggio
    link_to_washer: DryerLink = "after_wash"      # come si aggancia ai lavaggi
    winter_only: bool = True
    mode_weights: Dict[ModeName, float] = field(
        default_factory=lambda: {"eco": 0.5, "standard": 0.5}
    )

    def normalised_mode_probs(self) -> Dict[ModeName, float]:
        w = {m: max(self.mode_weights.get(m, 0.0), 0.0) for m in LAUNDRY_MODES}
        s = sum(w.values())
        if s <= 0:
            w = {m: 1.0 for m in LAUNDRY_MODES}
            s = sum(w.values())
        return {m: w[m] / s for m in LAUNDRY_MODES}




@dataclass
class LaundryConfig:
    washer_windows: LaundryWindows = field(default_factory=LaundryWindows.empty)
    dryer_windows: LaundryWindows = field(default_factory=LaundryWindows.empty)
    washer: WasherConfig = field(default_factory=WasherConfig)
    dryer: DryerConfig = field(default_factory=DryerConfig)


# ---------------------------------------------------------------------------
# Logica core: generazione eventi e profili di potenza
# ---------------------------------------------------------------------------

def _mode_profile_kw(mode: ModeName, device: str, power_kw: float) -> List[float]:
    """
    Restituisce una lista di valori di POTENZA (kW) a passo 1h per un ciclo.
    La durata è deterministica per modalità.
    L'energia per ciclo è deterministica (kWh) per modalità, scalata in modo
    semplice rispetto al power_kw (che incorpora anche la classe energetica via UI).
    """

    # Durate (ore) deterministiche
    DUR_H = {
        "washer": {"eco": 3, "standard": 2, "intensivo": 2, "rapido": 1},
        "dryer":  {"eco": 3, "standard": 2, "intensivo": 2, "rapido": 1},
    }

    # Energie base (kWh) deterministiche per modalità (valori indicativi ma stabili)
    # NOTA: sono "base" e poi vengono scalate con power_kw per mantenere coerenza con input utente.
    E_BASE = {
        "washer": {"eco": 0.7, "standard": 0.9, "intensivo": 1.2, "rapido": 0.6},
        "dryer":  {"eco": 1.8, "standard": 2.2, "intensivo": 2.6, "rapido": 1.6},
    }

    dev = "dryer" if device == "dryer" else "washer"
    mode = mode if mode in LAUNDRY_MODES else "standard"

    dur = int(DUR_H[dev][mode])  # ore intere
    if dur <= 0:
        dur = 1

    # scala energia in modo semplice rispetto al power_kw:
    # - washer baseline 1.0 kW, dryer baseline 0.8 kW (coerente con i default nel file)
    base_kw = 0.8 if dev == "dryer" else 1.0
    scale = (float(power_kw) / base_kw) if power_kw and power_kw > 0 else 1.0

    e_kwh = float(E_BASE[dev][mode]) * scale

    # profilo orario: potenza costante tale che energia totale = e_kwh
    p_kw = e_kwh / float(dur)

    return [p_kw] * dur

def _assign_modes_equal(n: int, active_modes: List[ModeName], rng: np.random.Generator) -> List[ModeName]:
    """
    Assegna le modalità in modo (quasi) perfettamente uniforme:
    - se n divisibile per M => esatto
    - altrimenti distribuisce il resto (differenza max 1 tra modalità)
    """
    if n <= 0:
        return []
    if not active_modes:
        active_modes = list(LAUNDRY_MODES)

    m = len(active_modes)
    q, r = divmod(n, m)

    out: List[ModeName] = []
    for i, mode in enumerate(active_modes):
        out.extend([mode] * (q + (1 if i < r else 0)))

    rng.shuffle(out)
    return out

def _simulate_device_profile(
    index: pd.DatetimeIndex,
    starts: Sequence[pd.Timestamp],
    mode_probs: Dict[ModeName, float],
    device: str,
    power_kw: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Costruisce il profilo di potenza (kW) per il dispositivo richiesto.

    La funzione è *resolution-agnostic*:
    - se l'indice è orario (Δt ≈ 1 h) si comporta come prima;
    - se l'indice è sub-orario (es. 15 min) replica ogni step del ciclo
      su più sotto-step all'interno dell'ora, mantenendo coerente l'energia totale.
    """
    s = pd.Series(0.0, index=index, dtype=float)

    if len(index) == 0 or power_kw <= 0:
        return s

    # passo temporale in ore (assumiamo indice equispaziato)
    if len(index) > 1:
        dt_hours = (index[1] - index[0]).total_seconds() / 3600.0
        if dt_hours <= 0:
            dt_hours = 1.0
    else:
        dt_hours = 1.0

    # modalità attive (peso > 0) — se vuote, fallback a tutte
    active_modes = [m for m, p in mode_probs.items() if p > 0]
    if not active_modes:
        active_modes = list(LAUNDRY_MODES)

    # assegnazione esatta delle modalità sugli start
    assigned_modes = _assign_modes_equal(len(starts), active_modes, rng)

    # se dt_hours < 1h, determiniamo quante "sotto-steps" ci stanno in un'ora
    # (es. 15 min => n_sub = 4)
    if dt_hours < 1.0:
        n_sub = max(1, int(round(1.0 / dt_hours)))
    else:
        n_sub = 1

    for ts0, mode in zip(starts, assigned_modes):
        # snap dello start al punto più vicino dell'indice (15 min)
        pos = index.get_indexer([pd.Timestamp(ts0)], method="nearest")[0]
        if pos < 0:
            continue
        ts0 = index[pos]

        chunk = _mode_profile_kw(mode=mode, device=device, power_kw=power_kw)

        if n_sub == 1:
            # comportamento: un punto per ogni step del ciclo a passo 1h
            for i, val in enumerate(chunk):
                t = ts0 + pd.Timedelta(hours=i)
                if t in s.index:
                    s.loc[t] += val
        else:
            # replica ogni step orario su sotto-step (es. 15 min)
            for i, val in enumerate(chunk):
                base_hour = float(i)
                for j in range(n_sub):
                    t = ts0 + pd.Timedelta(hours=base_hour + j * dt_hours)
                    if t in s.index:
                        s.loc[t] += val

    return s

def _simulation_base_and_full_weeks(index: pd.DatetimeIndex, max_weeks: int = 52) -> tuple[pd.Timestamp, int]:
    """
    Definisce le settimane di simulazione come blocchi consecutivi di 7 giorni,
    a partire da index.min().normalize(), senza allineamento al calendario.
    Ritorna (base_day, n_full_weeks), dove n_full_weeks è il numero di settimane COMPLETE.
    """
    if len(index) == 0:
        return pd.Timestamp("1970-01-01"), 0

    idx = index.sort_values()
    base = idx.min().normalize()
    last = idx.max().normalize()

    n_days = int((last - base).days) + 1  # giorni inclusivi
    n_full_weeks = n_days // 7           # solo settimane complete
    n_full_weeks = min(max_weeks, n_full_weeks)

    return base, n_full_weeks


def _sample_cycle_starts(
    index: pd.DatetimeIndex,
    windows: Dict[int, List[Tuple[int, int]]],
    cycles_per_week: float,
    rng: np.random.Generator,
    weekly_enabled: Optional[Sequence[bool]] = None,
) -> List[pd.Timestamp]:
    """
    Seleziona ESATTAMENTE cpw start per ciascuna settimana completa.
    - I candidati sono gli SLOT della start-matrix (bande h1-h2) per ciascun giorno.
    - Ogni slot può contenere al massimo 1 ciclo per settimana (scelta senza rimpiazzo).
    - Start time randomico dentro la banda, con step 15 minuti.
    - Nessun fallback: se un giorno non ha bande, è impossibile usarlo.
    """
    if cycles_per_week <= 0 or len(index) == 0:
        return []

    idx = index.sort_values()
    tmin = idx.min()
    tmax = idx.max()

    cpw = int(round(float(cycles_per_week)))
    if cpw <= 0:
        return []

    base, n_weeks = _simulation_base_and_full_weeks(idx, max_weeks=52)
    if n_weeks <= 0:
        return []

    # se weekly_enabled è fornito, deve avere almeno n_weeks valori
    if weekly_enabled is not None:
        weekly_enabled = list(weekly_enabled)[:n_weeks]
        if len(weekly_enabled) < n_weeks:
            weekly_enabled = weekly_enabled + [True] * (n_weeks - len(weekly_enabled))

    starts: List[pd.Timestamp] = []
    step_min = 15

    for wk in range(n_weeks):
        # se settimana disabilitata (es. winter rule dryer), nessun ciclo
        if weekly_enabled is not None and not bool(weekly_enabled[wk]):
            continue

        week_start = base + pd.Timedelta(days=7 * wk)

        # 1) costruisco la lista di SLOT candidati per questa settimana
        #    ogni candidato è (day_timestamp_normalizzato, h1, h2)
        candidates: List[Tuple[pd.Timestamp, int, int]] = []
        for d_off in range(7):
            day0 = (week_start + pd.Timedelta(days=d_off)).normalize()
            wd = int(day0.weekday())  # 0=Mon

            bands = windows.get(wd, [])
            if not bands:
                continue

            # un candidato per ogni banda selezionata
            for (h1, h2) in bands:
                h1 = int(h1)
                h2 = int(h2)
                if h2 <= h1:
                    continue
                candidates.append((day0, h1, h2))

        # vincolo: numero slot >= cpw
        if len(candidates) < cpw:
            raise ValueError(
                f"Settimana {wk+1} (dal {week_start.date()}): "
                f"slot disponibili={len(candidates)} < cicli richiesti={cpw}. "
                "Aumenta le finestre in start_matrix o riduci cycles_per_week."
            )

        chosen_idx = rng.choice(np.arange(len(candidates)), size=cpw, replace=False)

        # 2) per ciascuno slot scelto, estraggo uno start randomico a 15 minuti nella banda
        for ci in chosen_idx:
            day0, h1, h2 = candidates[int(ci)]

            start_min = h1 * 60
            end_min = h2 * 60

            # possibili start a step 15 minuti
            grid = np.arange(start_min, end_min, step_min)
            if len(grid) == 0:
                continue

            minute_of_day = int(rng.choice(grid))
            ts = day0 + pd.Timedelta(minutes=minute_of_day)

            # garantisco che lo start sia dentro l'indice
            if ts < tmin or ts > tmax:
                # se siamo al bordo (inizio/fine simulazione), scarto
                continue

            starts.append(ts)

    return sorted(starts)


def build_laundry_profiles(
    index: pd.DatetimeIndex,
    cfg: LaundryConfig,
    temp: Optional[pd.Series] = None,
    seed: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """
    Costruisce i profili di potenza (kW) di lavatrice e asciugatrice sul range di INDEX.

    Ritorna un dizionario con tre serie, tutte indicizzate su `index` e con dtype float:
        {
          "washer": potenza lavatrice (kW),
          "dryer":  potenza asciugatrice (kW),
          "total":  somma washer+dryer (kW)
        }
    """
    rng = np.random.default_rng(seed)

    # ----- LAVATRICE -----
    washer_series = pd.Series(0.0, index=index, dtype=float)
    if cfg.washer.has_washer and cfg.washer.cycles_per_week > 0:
        w_bands = cfg.washer_windows.to_bands()
        starts_w = _sample_cycle_starts(
            index=index,
            windows=w_bands,
            cycles_per_week=cfg.washer.cycles_per_week,
            rng=rng,
        )
        p_kw = cfg.washer.power_kw or 1.0  # default prudenziale
        washer_series = _simulate_device_profile(
            index=index,
            starts=starts_w,
            mode_probs=cfg.washer.normalised_mode_probs(),
            device="washer",
            power_kw=p_kw,
            rng=rng,
        )

    # ----- ASCIUGATRICE -----
    dryer_series = pd.Series(0.0, index=index, dtype=float)
    if cfg.dryer.has_dryer:
        # euristica per n° asciugature/settimana
        cycles_dry_week = float(cfg.dryer.cycles_per_week or 0.0)
        d_bands = cfg.dryer_windows.to_bands()

        weekly_enabled = None

        if (cfg.dryer.seasonality == "inverno") and (temp is not None):
            base, n_weeks = _simulation_base_and_full_weeks(index, max_weeks=52)
            weekly_enabled = []
            for wk in range(n_weeks):
                ws = base + pd.Timedelta(days=7 * wk)
                we = ws + pd.Timedelta(days=7)
                t_week = temp.loc[(temp.index >= ws) & (temp.index < we)]
                t_mean = float(t_week.mean()) if len(t_week) > 0 else float("nan")

                # REGOLA come da tuo messaggio:
                # Heuristica 'winter-only': se la temperatura media settimanale è sopra una soglia,
                # assumiamo che non sia necessario usare l'asciugatrice.
                # Nota: la soglia corrente è 15°C (scelta empirica e mantenuta per compatibilità).
                if np.isfinite(t_mean) and (t_mean > 15.0):
                    weekly_enabled.append(False)
                else:
                    weekly_enabled.append(True)
        starts_d = _sample_cycle_starts(
            index=index,
            windows=d_bands,
            cycles_per_week=cycles_dry_week,
            rng=rng,
            weekly_enabled=weekly_enabled,
        )

        p_kw_d = cfg.dryer.power_kw or 0.8
        dryer_series = _simulate_device_profile(
            index=index,
            starts=starts_d,
            mode_probs=cfg.dryer.normalised_mode_probs(),
            device="dryer",
            power_kw=p_kw_d,
            rng=rng,
        )


    total = washer_series + dryer_series
    return {"washer": washer_series, "dryer": dryer_series, "total": total}


# ---------------------------------------------------------------------------
# Wrapper di alto livello per compatibilità con 2_Consumatori.py
# ---------------------------------------------------------------------------

def _dict_to_laundry_config_for_washer(inputs: dict) -> LaundryConfig:
    """
    Converte il dict lavatrice della UI in un LaundryConfig minimale.

    Chiavi attese (se presenti):
      - "present": bool
      - "start_matrix": dict / griglia
      - "cycles_per_week": int/float
      - "P_nominal_W": W
      - "modes_selected": list[str]
      - "seasonality", "energy_class" (ignorate nel modello base)
    """
    raw_matrix = inputs.get("start_matrix")
    # la UI salva già un dict day->list di bande; per sicurezza normalizzo
    parsed = parse_start_matrix(raw_matrix)
    # trasformo in LaundryWindows con label SLOT_LABELS
    windows_by_day: Dict[int, List[SlotLabel]] = {}
    band_to_label = {b: lab for lab, b in zip(SLOT_LABELS, SLOT_BANDS)}
    for di, bands in parsed.items():
        labs: List[SlotLabel] = []
        for band in bands:
            lab = band_to_label.get(tuple(band))
            if lab:
                labs.append(lab)
        windows_by_day[int(di)] = labs
    lw = LaundryWindows(windows=windows_by_day)

    cpw_raw = inputs.get("cycles_per_week")
    try:
        cpw = float(cpw_raw) if cpw_raw is not None else 0.0
    except Exception:
        cpw = 0.0
    if cpw < 0:
        cpw = 0.0

    p_nom_w = inputs.get("P_nominal_W") or 0.0
    p_nom_kw = float(p_nom_w) / 1000.0 if p_nom_w else 1.0

    ec = inputs.get("energy_class", None)
    factor = ENERGY_CLASS_FACTOR.get(ec, 1.0)
    p_nom_kw = p_nom_kw * factor

    modes = inputs.get("modes_selected") or ["standard"]
    mode_weights = {m: 1.0 for m in modes if m in LAUNDRY_MODES}
    if not mode_weights:
        mode_weights = {"eco": 0.6, "standard": 0.4}

    washer_cfg = WasherConfig(
        has_washer=bool(inputs.get("present", True)),
        cycles_per_week=cpw,
        power_kw=p_nom_kw,
        mode_weights=mode_weights,
    )

    # nessuna asciugatrice in questa configurazione minimale
    dryer_cfg = DryerConfig(has_dryer=False)

    return LaundryConfig(
        washer_windows=lw,
        dryer_windows=lw,
        washer=washer_cfg,
        dryer=dryer_cfg,
    )


def _dict_to_laundry_config_for_dryer(inputs: dict) -> LaundryConfig:
    """
    Variante per simulare solo l'asciugatrice da UI.
    """
    raw_matrix = inputs.get("start_matrix")
    parsed = parse_start_matrix(raw_matrix)
    windows_by_day: Dict[int, List[SlotLabel]] = {}
    band_to_label = {b: lab for lab, b in zip(SLOT_LABELS, SLOT_BANDS)}
    for di, bands in parsed.items():
        labs: List[SlotLabel] = []
        for band in bands:
            lab = band_to_label.get(tuple(band))
            if lab:
                labs.append(lab)
        windows_by_day[int(di)] = labs
    lw = LaundryWindows(windows=windows_by_day)

    cpw_raw = inputs.get("cycles_per_week")
    try:
        cpw = float(cpw_raw) if cpw_raw is not None else 0.0
    except Exception:
        cpw = 0.0
    if cpw < 0:
        cpw = 0.0

    p_nom_w = inputs.get("P_nominal_W") or 0.0
    p_nom_kw = float(p_nom_w) / 1000.0 if p_nom_w else 0.8

    ec = inputs.get("energy_class", None)
    factor = ENERGY_CLASS_FACTOR.get(ec, 1.0)
    p_nom_kw = p_nom_kw * factor

    modes = inputs.get("modes_selected") or ["standard"]
    mode_weights = {m: 1.0 for m in modes if m in LAUNDRY_MODES}
    if not mode_weights:
        mode_weights = {"eco": 0.5, "standard": 0.5}

    washer_cfg = WasherConfig(
        has_washer=False,
        cycles_per_week=0.0,
        power_kw=None,
        mode_weights={"eco": 1.0},
    )

    dryer_cfg = DryerConfig(
        has_dryer=bool(inputs.get("present", True)),
        type="pompa_di_calore",
        power_kw=p_nom_kw,
        cycles_per_week=cpw,
        seasonality=inputs.get("seasonality", "tutto_anno"),
        ratio_dry_per_wash=1.0,
        link_to_washer="independent",
        mode_weights=mode_weights,
    )

    return LaundryConfig(
        washer_windows=LaundryWindows.empty(),
        dryer_windows=lw,
        washer=washer_cfg,
        dryer=dryer_cfg,
    )


def simulate_washer(
    index: pd.DatetimeIndex,
    inputs: dict | LaundryConfig,
    temp: Optional[pd.Series] = None,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Wrapper compatibile con la UI: restituisce solo la curva della lavatrice.
    """
    if isinstance(inputs, LaundryConfig):
        cfg = inputs
    else:
        cfg = _dict_to_laundry_config_for_washer(inputs)
    profiles = build_laundry_profiles(index, cfg, temp=temp, seed=seed)
    n_dev = inputs.get("n_devices", 1) if isinstance(inputs, dict) else 1
    try:
        n_dev = max(int(n_dev), 1)
    except Exception:
        n_dev = 1

    return profiles["washer"] * n_dev


def simulate_dryer(
    index: pd.DatetimeIndex,
    inputs: dict | LaundryConfig,
    temp: Optional[pd.Series] = None,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Wrapper compatibile con la UI: restituisce solo la curva dell'asciugatrice.
    """
    if isinstance(inputs, LaundryConfig):
        cfg = inputs
        n_dev = 1
    else:
        cfg = _dict_to_laundry_config_for_dryer(inputs)
        n_dev = inputs.get("n_devices", 1)

    profiles = build_laundry_profiles(index, cfg, temp=temp, seed=seed)

    try:
        n_dev = max(int(n_dev), 1)
    except Exception:
        n_dev = 1

    return profiles["dryer"] * n_dev


def simulate(
    appliance: str,
    index: pd.DatetimeIndex,
    inputs,
    temp: Optional[pd.Series] = None,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    API generica:

        simulate("washer", index, inputs, temp, seed)
        simulate("dryer",  index, inputs, temp, seed)

    dove `inputs` è:
      - dict (come salvato in consumers.json)
      - oppure un LaundryConfig già costruito.
    """
    a = (appliance or "").lower()
    if a == "washer":
        return simulate_washer(index=index, inputs=inputs, temp=temp, seed=seed)
    elif a == "dryer":
        return simulate_dryer(index=index, inputs=inputs, temp=temp, seed=seed)
    else:
        raise ValueError(f"Appliance non supportato in lavanderia.simulate(): {appliance!r}")
