"""cer_core.consumatori.cucina

Modelli sintetici di carico per l'area *cucina* di un consumatore domestico.

Questo modulo genera profili di **potenza elettrica** (serie temporali in **kW**) per
alcuni apparecchi tipici della cucina:

- forno elettrico (cicli di cottura con preriscaldamento + plateau)
- piano a induzione (sessioni legate ai pasti)
- microonde (sessioni brevi, come supporto/riscaldo)
- lavastoviglie (cicli legati a pranzo/cena; start time derivati da abitudini)
- cappa aspirante (correlata a eventi di cottura, proxy: forno+induzione attivi)

Il modello è pensato per essere *compatibile* con l'orchestrazione Streamlit del progetto
(CER simulator) e quindi adotta le seguenti convenzioni operative.

Convenzioni e assunzioni
------------------------
- L'input temporale è un :class:`pandas.DatetimeIndex` (tipicamente a 15 minuti nel
  simulatore CER). L'indice deve essere **equispaziato**; il passo `dt` viene stimato
  da ``idx[1] - idx[0]`` e usato per discretizzare durate in minuti.
- Le potenze sono espresse in **kW**; eventuali parametri in watt (es. cappa) vengono
  convertiti in kW.
- Le funzioni sono *pure* (nessun I/O su disco): la persistenza su CSV/JSON è gestita
  dai livelli UI (``cer_app/schede_consumatori``).
- Riproducibilità: quasi tutte le componenti accettano un ``seed`` opzionale e usano
  un generatore ``numpy.random.default_rng``. **Eccezione**: la cappa utilizza
  ``np.random.random()`` (RNG globale) per una scelta binaria evento-per-evento;
  questo significa che la componente "hood" non è deterministica anche a seed fissato.

Output
------
La funzione di ingresso consigliata è :func:`build_kitchen_profiles`, che restituisce
un dizionario di serie ``pd.Series`` indicizzate da ``idx`` con chiavi:

``"oven"``, ``"induction"``, ``"microwave"``, ``"dishwasher"``, ``"hood"``, ``"aggregated"``.

La serie ``"aggregated"`` è la somma punto-a-punto delle singole componenti.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import math

# ---------------------------------------------------------------------------
# Tipi di base
# ---------------------------------------------------------------------------

DayIndex = int          # 0 = Monday ... 6 = Sunday
MealType = Literal["breakfast", "lunch", "dinner"]
DayType = Literal["weekday", "weekend"]

# Rappresentazione compatta dei pasti a casa:
# meal_matrix[day][slot] -> insieme di meal_type (es. {"lunch"}, {"dinner"})
MealMatrix = Dict[DayIndex, Dict[int, List[MealType]]]


# ---------------------------------------------------------------------------
# Configurazione "pasti a casa" (comune a tutti gli apparecchi cucina)
# ---------------------------------------------------------------------------

@dataclass
class MealHabitConfig:
    """Abitudini di pasti a casa, in forma compatta.

    Questa struttura incapsula esattamente la logica che la UI può chiedere
    con slider / select:

    - quante volte pranzo/cena a casa nei feriali/weekend
    - in quali fasce orarie di solito pranzo/ceno
    """

    # Frequenza pasti a casa (categorie discrete, vedi mapping più sotto)
    weekday_lunch_at_home: int = 2      # 0..3
    weekday_dinner_at_home: int = 3     # 0..3
    weekend_lunch_at_home: int = 2      # 0..2
    weekend_dinner_at_home: int = 2     # 0..2

    # Orari tipici (slot index, non stringhe; la UI mappa "12-13" -> 3, ecc.)
    # slot: 0..N_slots-1, in coerenza con SLOT_LABELS del progetto
    lunch_slot_weekday: int = 3
    dinner_slot_weekday: int = 6
    lunch_slot_weekend: int = 3
    dinner_slot_weekend: int = 6

    # Colazione (opzionale; se non ti serve puoi ignorarla)
    enable_breakfast: bool = False
    breakfast_at_home: int = 1  # 0..2
    breakfast_slot: int = 1     # es. 1 = 8-10


# ---------------------------------------------------------------------------
# Configurazioni per singolo apparecchio
# ---------------------------------------------------------------------------

# --- Forno -----------------------------------------------------------------

@dataclass
class OvenConfig:
    has_oven: bool = False

    # Interpretato come "volte/settimana" (0..N) nella UI
    oven_weekly_intensity: int = 2

    oven_days_preference: Literal[
        "solo_weekend", "soprattutto_weekend",
        "equilibrato", "anche_feriali_spesso"
    ] = "soprattutto_weekend"

    oven_meal_preference: Literal[
        "solo_cena", "prevalentemente_cena",
        "pranzo_e_cena", "soprattutto_pranzo_festivi"
    ] = "prevalentemente_cena"

    oven_complexity: Literal["veloce", "misto", "elaborato"] = "misto"

    # Nuovo: potenza nominale configurabile (kW)
    oven_power_kw: float = 2.0


# --- Piano a induzione -----------------------------------------------------

@dataclass
class InductionConfig:
    has_induction: bool = False

    induction_is_primary: bool = True
    induction_use_ratio: Literal["quasi_sempre", "spesso", "meta", "raramente"] = "spesso"
    induction_meal_preference: Literal[
        "colazione_pranzo", "pranzo_cena", "solo_cena", "tutti_i_pasti_caldi"
    ] = "pranzo_cena"
    cooking_intensity: Literal["semplice", "misto", "elaborata"] = "misto"
    typical_burners_in_use: int = 2     # 1,2,3 (>=3 -> 3)


# --- Microonde -------------------------------------------------------------

@dataclass
class MicrowaveConfig:
    has_microwave: bool = False

    microwave_main_use: Literal[
        "riscaldare_pranzo_lavoro",
        "scongelare",
        "cucinare_piatti_pronti",
        "vario",
    ] = "vario"

    microwave_time_preference: Literal[
        "solo_pranzo_feriale",
        "spesso_cena",
        "anche_colazioni_snack",
        "distribuito",
    ] = "distribuito"

    # Interpretato come "volte/settimana" nella UI
    microwave_weekly_intensity: int = 2

    microwave_session_duration: Literal["1-3", "3-5", "5-10"] = "3-5"
    microwave_as_oven_substitute: Literal["mai", "a_volte", "spesso"] = "mai"

    # Nuovo: potenza nominale configurabile (kW)
    microwave_power_kw: float = 0.9


# --- Cappa cucina ----------------------------------------------------------

@dataclass
class HoodConfig:
    has_hood: bool = False

    hood_use_habit: Literal["quasi_mai", "piatti_importanti", "spesso", "quasi_sempre"] = "spesso"
    hood_cooking_type: Literal[
        "solo_fritture_griglia",
        "piatti_elaborati",
        "qualsiasi_piatto_caldo"
    ] = "piatti_elaborati"
    hood_duration_relative: Literal["solo_inizio", "meta_tempo", "quasi_tutto"] = "meta_tempo"
    hood_power_w: float = 100.0     # potenza nominale cappa

# --- Lavastoviglie --------------------------------------------------------

@dataclass
class DishwasherConfig:
    """Configurazione lavastoviglie (carico event-driven legato ai pasti).

    Parametri chiave
    --------------
    has_dishwasher:
        Abilita/disabilita la lavastoviglie.
    cycles_per_week:
        Numero medio di cicli per settimana.
    program:
        Programma (eco/standard/quick) che influenza durata e shape del profilo.
    energy_class:
        Classe energetica (A..G) che influenza l'energia per ciclo (kWh).
    mode:
        - "after_meal": avvio dopo pranzo o cena (in base a meal_matrix).
        - "scheduled_fixed_time": avvio a orario fisso (tutti i giorni con ciclo).
    fixed_start_time:
        Orario HH:MM in locale (Europe/Rome), usato solo in modalità scheduled.

    Note
    ----
    - Il modello è costruito per il timebase tipico del simulatore (15 minuti, UTC).
      In caso di passi diversi, i template vengono interpolati.
    - La variabilità è gestita da jitter sugli orari dei pasti (Metodo A) e da piccoli
      jitter/delay post-pasto, tutti controllati da seed.
    """

    has_dishwasher: bool = False
    cycles_per_week: int = 4

    program: Literal["eco", "standard", "quick"] = "eco"
    energy_class: Literal["A", "B", "C", "D", "E", "F", "G"] = "C"

    mode: Literal["after_meal", "scheduled_fixed_time"] = "after_meal"
    fixed_start_time: str = "22:30"



# ---------------------------------------------------------------------------
# Configurazione complessiva cucina per un singolo consumatore
# ---------------------------------------------------------------------------

@dataclass
class KitchenConfig:
    habits: MealHabitConfig = field(default_factory=MealHabitConfig)
    oven: OvenConfig = field(default_factory=OvenConfig)
    induction: InductionConfig = field(default_factory=InductionConfig)
    microwave: MicrowaveConfig = field(default_factory=MicrowaveConfig)
    dishwasher: DishwasherConfig = field(default_factory=DishwasherConfig)
    hood: HoodConfig = field(default_factory=HoodConfig)


# ---------------------------------------------------------------------------
# Utility di supporto: day_type, conversioni, ecc.
# ---------------------------------------------------------------------------

def day_type(day: DayIndex) -> DayType:
    """Classifica un indice di giorno (0=lunedì ... 6=domenica) in feriale/weekend."""
    return "weekday" if day < 5 else "weekend"



def _estimate_weeks(idx: pd.DatetimeIndex) -> float:
    """Stima le *settimane equivalenti* coperte da ``idx``.

    La stima è basata sulla durata ``idx[-1] - idx[0]`` (in giorni) divisa per 7.
    È un’approssimazione sufficiente per convertire intensità espresse in "cicli/settimana"
    nel numero di eventi da generare sul periodo simulato.
    """
    if len(idx) < 2:
        return 1.0  # fallback neutro
    days = (idx[-1] - idx[0]).total_seconds() / 86400.0
    if days <= 0:
        return 1.0
    return days / 7.0


# ---------------------------------------------------------------------------
# STEP 1 — Generazione della meal_matrix settimanale
# ---------------------------------------------------------------------------

def build_meal_matrix(cfg: MealHabitConfig, n_slots: int, seed: Optional[int] = None) -> MealMatrix:
    """Genera la *meal matrix* settimanale (pasti svolti a casa).

    La rappresentazione è compatta: per ciascun giorno della settimana e per ciascuno
    ``slot`` (fascia oraria discreta scelta dalla UI), viene registrata una lista di
    etichette di pasto (``"breakfast"``, ``"lunch"``, ``"dinner"``).

    Notes
    -----
    - Le frequenze impostate dalla UI sono mappate in **giorni/settimana** tramite tabelle
      discrete (``freq_map_weekday``, ``freq_map_weekend``) e poi convertite in probabilità
      per giorno (es. ``p_lunch_weekday = days/5``).
    - La funzione genera una singola settimana "tipica"; i modelli dei singoli apparecchi
      useranno questa matrice come base per distribuire eventi anche su periodi più lunghi.

    Parameters
    ----------
    cfg:
        Configurazione delle abitudini alimentari.
    n_slots:
        Numero di slot discreti della giornata (coerente con la UI; nel progetto spesso 10).
    seed:
        Seed per rendere riproducibile l’estrazione casuale.

    Returns
    -------
    MealMatrix
        Dizionario ``meal_matrix[day][slot] -> List[MealType]``.
    """
    rng = np.random.default_rng(seed)
    meal_matrix: MealMatrix = {d: {s: [] for s in range(n_slots)} for d in range(7)}

    # ---- mapping frequenza -> giorni/settimana ----
    freq_map_weekday = {0: 0.0, 1: 1.5, 2: 3.5, 3: 5.0}
    freq_map_weekend = {0: 0.0, 1: 1.0, 2: 2.0}

    weekday_lunch_days = freq_map_weekday.get(cfg.weekday_lunch_at_home, 3.5)
    weekday_dinner_days = freq_map_weekday.get(cfg.weekday_dinner_at_home, 5.0)
    weekend_lunch_days = freq_map_weekend.get(cfg.weekend_lunch_at_home, 2.0)
    weekend_dinner_days = freq_map_weekend.get(cfg.weekend_dinner_at_home, 2.0)

    p_lunch_weekday = weekday_lunch_days / 5.0
    p_dinner_weekday = weekday_dinner_days / 5.0
    p_lunch_weekend = weekend_lunch_days / 2.0
    p_dinner_weekend = weekend_dinner_days / 2.0

    # Colazione (se abilitata)
    if cfg.enable_breakfast:
        freq_map_breakfast = {0: 0.0, 1: 4.0, 2: 7.0}
        breakfast_days = freq_map_breakfast.get(cfg.breakfast_at_home, 4.0)
        p_breakfast_weekday = breakfast_days / 5.0
    else:
        p_breakfast_weekday = 0.0

    # ---- generazione per ogni giorno ----
    for d in range(7):
        dt = day_type(d)
        if dt == "weekday":
            # colazione
            if cfg.enable_breakfast:
                if rng.random() < p_breakfast_weekday:
                    meal_matrix[d][cfg.breakfast_slot].append("breakfast")
            # pranzo
            if rng.random() < p_lunch_weekday:
                meal_matrix[d][cfg.lunch_slot_weekday].append("lunch")
            # cena
            if rng.random() < p_dinner_weekday:
                meal_matrix[d][cfg.dinner_slot_weekday].append("dinner")
        else:
            # weekend: colazione, se vuoi puoi aggiungerla con logica dedicata
            if cfg.enable_breakfast:
                if rng.random() < 0.8:
                    meal_matrix[d][cfg.breakfast_slot].append("breakfast")
            # pranzo
            if rng.random() < p_lunch_weekend:
                meal_matrix[d][cfg.lunch_slot_weekend].append("lunch")
            # cena
            if rng.random() < p_dinner_weekend:
                meal_matrix[d][cfg.dinner_slot_weekend].append("dinner")

    return meal_matrix


# ---------------------------------------------------------------------------
# STEP 2 — Forno: numero cicli stile "lavatrice" + profilo potenza
# ---------------------------------------------------------------------------

def _oven_cpw_from_intensity(level: int) -> float:
    """Mapping livello slider -> cicli/settimana.

    Qui assumiamo che lo slider rappresenti direttamente il numero medio
    di utilizzi a settimana.
    """
    return float(level)


def _oven_weights(cfg: OvenConfig, meals: MealMatrix) -> Tuple[Dict[Tuple[MealType, DayType], float],
                                                               Dict[Tuple[MealType, DayType], int]]:
    """Calcola pesi grezzi w_oven e numero di occasioni di pasto."""
    # pesi base per giorno
    if cfg.oven_days_preference == "solo_weekend":
        w_weekday, w_weekend = 1.0, 5.0
    elif cfg.oven_days_preference == "soprattutto_weekend":
        w_weekday, w_weekend = 2.0, 4.0
    elif cfg.oven_days_preference == "equilibrato":
        w_weekday, w_weekend = 3.0, 3.0
    else:  # "anche_feriali_spesso"
        w_weekday, w_weekend = 4.0, 3.0

    # pesi base per pasto
    base = {}
    if cfg.oven_meal_preference == "solo_cena":
        base[("lunch", "weekday")] = 1.0
        base[("dinner", "weekday")] = 5.0
        base[("lunch", "weekend")] = 1.0
        base[("dinner", "weekend")] = 5.0
    elif cfg.oven_meal_preference == "prevalentemente_cena":
        base[("lunch", "weekday")] = 2.0
        base[("dinner", "weekday")] = 4.0
        base[("lunch", "weekend")] = 2.0
        base[("dinner", "weekend")] = 4.0
    elif cfg.oven_meal_preference == "pranzo_e_cena":
        base[("lunch", "weekday")] = 3.0
        base[("dinner", "weekday")] = 3.0
        base[("lunch", "weekend")] = 3.0
        base[("dinner", "weekend")] = 3.0
    else:  # "soprattutto_pranzo_festivi"
        base[("lunch", "weekday")] = 2.0
        base[("dinner", "weekday")] = 3.0
        base[("lunch", "weekend")] = 4.0
        base[("dinner", "weekend")] = 2.0

    w: Dict[Tuple[MealType, DayType], float] = {}
    n_occ: Dict[Tuple[MealType, DayType], int] = {("lunch","weekday"):0,
                                                  ("dinner","weekday"):0,
                                                  ("lunch","weekend"):0,
                                                  ("dinner","weekend"):0}
    # conta occasioni di pasto
    for d, slots in meals.items():
        dt = day_type(d)
        for _s, meal_list in slots.items():
            for m in meal_list:
                if m not in ("lunch","dinner"):
                    continue
                key = (m, dt)
                n_occ[key] = n_occ.get(key, 0) + 1

    # costruisci pesi
    for (m, dt) in n_occ.keys():
        if dt == "weekday":
            day_w = w_weekday
        else:
            day_w = w_weekend
        base_w = base.get((m, dt), 0.0)
        w[(m, dt)] = day_w * base_w

    return w, n_occ


def _oven_cycle_shape(complexity: str, dt_hours: float) -> np.ndarray:
    """Restituisce un profilo normalizzato (somma ≈ 1) su passi dt_hours."""
    if complexity == "veloce":
        total_min = 25
        p1, p2 = 1.0, 0.6
    elif complexity == "elaborato":
        total_min = 75
        p1, p2 = 1.0, 0.7
    else:  # misto
        total_min = 45
        p1, p2 = 1.0, 0.65

    n_steps = max(1, int(round(total_min / (dt_hours * 60.0))))
    # 20% tempo preriscaldamento, 80% cottura
    n_p1 = max(1, int(0.2 * n_steps))
    n_p2 = max(1, n_steps - n_p1)

    shape = np.concatenate([np.full(n_p1, p1), np.full(n_p2, p2)])
    # normalizza a media 1 (per usare poi una potenza media prestabilita)
    shape = shape / max(shape.mean(), 1e-6)
    return shape


def generate_oven_events(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    meal_matrix: MealMatrix,
    seed: Optional[int] = None,
) -> List[pd.Timestamp]:
    """Genera start-time dei cicli forno con numero cicli ~ cpw * weeks."""
    if not cfg.oven.has_oven or len(idx) == 0:
        return []

    rng = np.random.default_rng(seed)

    # 1) cicli per settimana dalla UI (interpretato come "volte/settimana")
    cpw = _oven_cpw_from_intensity(cfg.oven.oven_weekly_intensity)
    if cpw <= 0:
        return []

    # 2) settimane coperte dall'indice temporale
    weeks = _estimate_weeks(idx)
    n_cycles = int(round(cpw * weeks))
    if n_cycles <= 0:
        return []

    # 3) pesi base per (meal_type, day_type)
    w_base, _ = _oven_weights(cfg.oven, meal_matrix)

    # 4) costruiamo una lista di CANDIDATI:
    #    per ogni pasto (lunch/dinner) una lista di timestamp possibili + un peso
    candidates_ts: List[pd.DatetimeIndex] = []
    candidates_w: List[float] = []

    hours = idx.hour.values
    dow = idx.dayofweek.values

    for d, slots in meal_matrix.items():
        dt = day_type(d)
        for slot_idx, meal_list in slots.items():
            for meal in meal_list:
                if meal not in ("lunch", "dinner"):
                    continue

                key = (meal, dt)
                base_w = w_base.get(key, 0.0)
                if base_w <= 0:
                    continue

                # definisco la banda oraria corrispondente allo slot
                h1 = (slot_idx * 2) % 24
                h2 = (h1 + 2) % 24

                mask_day = (dow == d)
                if h1 < h2:
                    mask_slot = (hours >= h1) & (hours < h2)
                else:
                    mask_slot = (hours >= h1) | (hours < h2)

                ts_candidates = idx[mask_day & mask_slot]
                if len(ts_candidates) == 0:
                    continue

                candidates_ts.append(ts_candidates)
                candidates_w.append(base_w * len(ts_candidates))

    if not candidates_ts:
        return []

    probs = np.array(candidates_w, dtype=float)
    s = probs.sum()
    if s <= 0:
        return []
    probs /= s

    # 5) scegli i "bucket" di (giorno, slot, pasto) in cui inserire i cicli
    chosen_bucket_idx = rng.choice(len(candidates_ts), size=n_cycles, replace=True, p=probs)

    events: List[pd.Timestamp] = []
    for b_idx in chosen_bucket_idx:
        ts_bucket = candidates_ts[b_idx]
        ts = rng.choice(ts_bucket)  # un timestamp specifico dentro la fascia
        events.append(ts)

    events.sort()
    return events


def oven_power_series(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    meal_matrix: MealMatrix,
    seed: Optional[int] = None,
) -> pd.Series:
    """Serie di potenza del forno su ``idx`` (kW).

    Implementa cicli di cottura la cui numerosità è determinata da:
    ``n_cycles ≈ oven_weekly_intensity * weeks(idx)``.

    Il profilo di ciascun ciclo è una forma discreta (preriscaldamento + plateau)
    normalizzata a media 1; la potenza assoluta è impostata da ``oven_power_kw``.
    """
    if not cfg.oven.has_oven or len(idx) == 0:
        return pd.Series(0.0, index=idx)

    idx = idx.sort_values()
    dt_hours = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0

    events = generate_oven_events(idx, cfg, meal_matrix, seed=seed)
    power = pd.Series(0.0, index=idx)

    # potenza media di riferimento (parametrizzata)
    p_kw = float(cfg.oven.oven_power_kw or 2.0)

    shape = _oven_cycle_shape(cfg.oven.oven_complexity, dt_hours)

    for ts in events:
        for i, val in enumerate(shape):
            t = ts + pd.Timedelta(hours=i * dt_hours)
            if t in power.index:
                power.loc[t] += p_kw * float(val)

    return power


# ---------------------------------------------------------------------------
# STEP 3 — Piano a induzione
# ---------------------------------------------------------------------------

def _induction_base_probabilities(cfg: InductionConfig) -> Dict[MealType, float]:
    # ratio
    k_ratio = {
        "quasi_sempre": 0.9,
        "spesso": 0.7,
        "meta": 0.5,
        "raramente": 0.2,
    }.get(cfg.induction_use_ratio, 0.7)

    k_primary = 1.0 if cfg.induction_is_primary else 0.6
    k_intensity = {
        "semplice": 0.8,
        "misto": 1.0,
        "elaborata": 1.2,
    }.get(cfg.cooking_intensity, 1.0)
    k_intensity = min(k_intensity, 1.2)

    if cfg.induction_meal_preference == "colazione_pranzo":
        w = {"breakfast": 1.0, "lunch": 1.0, "dinner": 0.3}
    elif cfg.induction_meal_preference == "pranzo_cena":
        w = {"breakfast": 0.2, "lunch": 1.0, "dinner": 1.0}
    elif cfg.induction_meal_preference == "solo_cena":
        w = {"breakfast": 0.1, "lunch": 0.2, "dinner": 1.0}
    else:  # tutti_i_pasti_caldi
        w = {"breakfast": 0.8, "lunch": 1.0, "dinner": 1.0}

    out = {}
    for m, wm in w.items():
        out[m] = min(1.0, k_ratio * k_primary * k_intensity * wm)
    return out


def _induction_duration_and_power(cfg: InductionConfig) -> Tuple[float, float]:
    if cfg.cooking_intensity == "semplice":
        duration_min = 20.0
        base_power_per_burner = 1.0
    elif cfg.cooking_intensity == "elaborata":
        duration_min = 45.0
        base_power_per_burner = 1.8
    else:
        duration_min = 30.0
        base_power_per_burner = 1.4

    n_burners = max(1, min(int(cfg.typical_burners_in_use), 3))
    duty = 0.6
    avg_power_kw = base_power_per_burner * n_burners * duty
    return duration_min, avg_power_kw


def induction_power_series(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    meal_matrix: MealMatrix,
    seed: Optional[int] = None,
) -> pd.Series:
    """Serie di potenza del piano a induzione su ``idx`` (kW).

    Per ogni pasto presente in ``meal_matrix`` viene estratta (con probabilità dipendente
    dalla configurazione) una sessione di cottura. Durata e potenza media sono ricavate
    da :func:`_induction_duration_and_power`.
    """
    if not cfg.induction.has_induction or len(idx) == 0:
        return pd.Series(0.0, index=idx)

    idx = idx.sort_values()
    dt_hours = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0

    rng = np.random.default_rng(seed)
    p_base = _induction_base_probabilities(cfg.induction)
    duration_min, avg_power_kw = _induction_duration_and_power(cfg.induction)

    n_steps = max(1, int(round(duration_min / (dt_hours * 60.0))))
    # profilo: primi 20% step a potenza più alta, resto a plateau
    n_p1 = max(1, int(0.2 * n_steps))
    n_p2 = max(1, n_steps - n_p1)
    shape = np.concatenate([np.full(n_p1, 1.2), np.full(n_p2, 0.8)])
    shape = shape / max(shape.mean(), 1e-6)

    power = pd.Series(0.0, index=idx)

    for d, slots in meal_matrix.items():
        dt = day_type(d)
        for slot_idx, meal_list in slots.items():
            for meal in meal_list:
                p_use_base = p_base.get(meal, 0.0)
                if p_use_base <= 0:
                    continue
                # leggero boost weekend
                if dt == "weekend":
                    p_use = min(1.0, p_use_base * 1.1)
                else:
                    p_use = p_use_base
                if rng.random() < p_use:
                    # scegli timestamp nello slot
                    mask_day = idx.dayofweek == d
                    hours = idx.hour.values
                    h1 = (slot_idx * 2) % 24
                    h2 = (h1 + 2) % 24
                    if h1 < h2:
                        mask_slot = (hours >= h1) & (hours < h2)
                    else:
                        mask_slot = (hours >= h1) | (hours < h2)
                    candidates = idx[mask_day & mask_slot]
                    if len(candidates) == 0:
                        continue
                    ts = rng.choice(candidates)
                    for i, val in enumerate(shape):
                        t = ts + pd.Timedelta(hours=i * dt_hours)
                        if t in power.index:
                            power.loc[t] += avg_power_kw * float(val)

    return power


# ---------------------------------------------------------------------------
# STEP 4 — Microonde
# ---------------------------------------------------------------------------

def _microwave_cpw(level: int) -> float:
    """Interpretazione diretta dello slider come cicli/settimana."""
    return float(level)


def _microwave_weights(cfg: MicrowaveConfig, has_oven: bool, meals: MealMatrix
                       ) -> Tuple[Dict[Tuple[MealType, DayType], float],
                                  Dict[Tuple[MealType, DayType], int]]:
    w: Dict[Tuple[MealType, DayType], float] = {}
    n_occ: Dict[Tuple[MealType, DayType], int] = {}

    def add_weight(m: MealType, dt: DayType, val: float) -> None:
        key = (m, dt)
        w[key] = w.get(key, 0.0) + val

    # Base da main_use
    if cfg.microwave_main_use == "riscaldare_pranzo_lavoro":
        add_weight("lunch", "weekday", 3.0)
    elif cfg.microwave_main_use == "scongelare":
        add_weight("dinner", "weekday", 2.0)
        add_weight("dinner", "weekend", 2.0)
    elif cfg.microwave_main_use == "cucinare_piatti_pronti":
        add_weight("lunch", "weekday", 2.0)
        add_weight("dinner", "weekday", 2.0)
    else:  # vario
        add_weight("lunch", "weekday", 1.0)
        add_weight("dinner", "weekday", 1.0)
        add_weight("dinner", "weekend", 1.0)

    # time_preference
    if cfg.microwave_time_preference == "solo_pranzo_feriale":
        w[("lunch", "weekday")] = w.get(("lunch", "weekday"), 0.0) * 2.0
    elif cfg.microwave_time_preference == "spesso_cena":
        for key in [("dinner","weekday"), ("dinner","weekend")]:
            w[key] = w.get(key, 0.0) * 1.5
    elif cfg.microwave_time_preference == "anche_colazioni_snack":
        add_weight("breakfast", "weekday", 1.0)
        add_weight("breakfast", "weekend", 1.0)
    else:  # distribuito
        add_weight("breakfast", "weekday", 0.5)
        add_weight("breakfast", "weekend", 0.5)

    # sostituto forno
    if not has_oven and cfg.microwave_as_oven_substitute != "mai":
        factor = 1.4 if cfg.microwave_as_oven_substitute == "a_volte" else 2.0
        for key in [("dinner","weekend"), ("lunch","weekend")]:
            w[key] = w.get(key, 0.0) * factor

    # conta occasioni
    for d, slots in meals.items():
        dt = day_type(d)
        for _s, meal_list in slots.items():
            for m in meal_list:
                key = (m, dt)
                n_occ[key] = n_occ.get(key, 0) + 1

    return w, n_occ


def microwave_power_series(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    meal_matrix: MealMatrix,
    seed: Optional[int] = None,
) -> pd.Series:
    """Serie di potenza del microonde su ``idx`` (kW).

    Il numero di sessioni nel periodo è proporzionale a ``microwave_weekly_intensity``
    (interpretato come "cicli/settimana") e al numero di settimane stimate su ``idx``.
    La durata della sessione è discretizzata su ``dt`` in base a ``microwave_session_duration``.
    """
    # se il microonde non è presente, ritorno zero
    if not cfg.microwave.has_microwave or len(idx) == 0:
        return pd.Series(0.0, index=idx)

    idx = idx.sort_values()
    dt_hours = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0

    rng = np.random.default_rng(seed)

    # Numero di cicli desiderati nel periodo
    cpw = _microwave_cpw(cfg.microwave.microwave_weekly_intensity)
    if cpw <= 0:
        return pd.Series(0.0, index=idx)

    weeks = _estimate_weeks(idx)
    n_sessions = int(round(cpw * weeks))
    if n_sessions <= 0:
        return pd.Series(0.0, index=idx)

    # pesi per (meal_type, day_type) in base alla config
    w_base, _ = _microwave_weights(cfg.microwave, cfg.oven.has_oven, meal_matrix)

    candidates_ts: List[pd.DatetimeIndex] = []
    candidates_w: List[float] = []

    hours = idx.hour.values
    dow = idx.dayofweek.values

    for d, slots in meal_matrix.items():
        dt = day_type(d)
        for slot_idx, meal_list in slots.items():
            for meal in meal_list:
                key = (meal, dt)
                base_w = w_base.get(key, 0.0)
                if base_w <= 0:
                    continue

                mask_day = (dow == d)
                h1 = (slot_idx * 2) % 24
                h2 = (h1 + 2) % 24
                if h1 < h2:
                    mask_slot = (hours >= h1) & (hours < h2)
                else:
                    mask_slot = (hours >= h1) | (hours < h2)

                ts_candidates = idx[mask_day & mask_slot]
                if len(ts_candidates) == 0:
                    continue

                candidates_ts.append(ts_candidates)
                candidates_w.append(base_w * len(ts_candidates))

    if not candidates_ts:
        return pd.Series(0.0, index=idx)

    probs = np.array(candidates_w, dtype=float)
    s = probs.sum()
    if s <= 0:
        return pd.Series(0.0, index=idx)
    probs /= s

    chosen_bucket_idx = rng.choice(len(candidates_ts), size=n_sessions, replace=True, p=probs)

    events: List[pd.Timestamp] = []
    for b_idx in chosen_bucket_idx:
        ts_bucket = candidates_ts[b_idx]
        ts = rng.choice(ts_bucket)
        events.append(ts)

    # durata / potenza — prendo la durata dalla config del microonde
    session_dur = cfg.microwave.microwave_session_duration
    if session_dur == "1-3":
        duration_min = 2.0
    elif session_dur == "5-10":
        duration_min = 7.0
    else:  # "3-5"
        duration_min = 4.0

    n_steps = max(1, int(round(duration_min / (dt_hours * 60.0))))
    shape = np.ones(n_steps, dtype=float)  # microonde ~ potenza costante
    shape = shape / max(shape.mean(), 1e-6)
    p_kw = float(cfg.microwave.microwave_power_kw or 0.9)  # kW

    power = pd.Series(0.0, index=idx)

    for ts in events:
        for i, val in enumerate(shape):
            t = ts + pd.Timedelta(hours=i * dt_hours)
            if t in power.index:
                power.loc[t] += p_kw * float(val)

    return power


# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# STEP 4b — Lavastoviglie (legata ai pasti, senza matrice oraria manuale)
# ---------------------------------------------------------------------------

_DW_CLASS_FACTOR = {
    "A": 0.85,
    "B": 0.92,
    "C": 1.00,
    "D": 1.10,
    "E": 1.20,
    "F": 1.35,
    "G": 1.50,
}

_DW_BASE_ENERGY_C_KWH = {
    "eco": 0.96,
    "standard": 1.20,
    "quick": 0.66,
}

_DW_DURATION_MIN = {
    "eco": 210.0,
    "standard": 150.0,
    "quick": 60.0,
}

# Template discreti a 15 minuti (kW), calibrati per classe C (senza scaling).
_DW_TEMPLATES_15MIN = {
    "eco": np.array([0.07, 0.09, 1.15, 0.20, 0.16, 0.14, 0.16, 0.14, 0.16, 1.05, 0.20, 0.12, 0.10, 0.09], dtype=float),
    "standard": np.array([0.10, 1.60, 0.28, 0.24, 0.24, 0.28, 1.45, 0.32, 0.18, 0.10], dtype=float),
    "quick": np.array([0.15, 1.90, 0.40, 0.20], dtype=float),
}


def _clip_minutes(x: float, cap: float) -> float:
    return max(-cap, min(cap, float(x)))


def _round_to_15min(ts: pd.Timestamp) -> pd.Timestamp:
    # Nearest 15 minutes. Pandas gestisce correttamente tz-aware.
    return ts.round("15min")


def _dishwasher_target_energy_kwh(program: str, energy_class: str) -> float:
    base = float(_DW_BASE_ENERGY_C_KWH.get(program, 1.2))
    factor = float(_DW_CLASS_FACTOR.get(str(energy_class).upper(), 1.0))
    return base * factor


def _resample_template_to_dt(template_15min: np.ndarray, duration_min: float, dt_hours: float) -> np.ndarray:
    """Interpola un template definito a 15 minuti sul dt effettivo dell'indice."""
    if dt_hours <= 0:
        return template_15min.copy()
    dt_min = dt_hours * 60.0
    n_steps = max(1, int(round(duration_min / dt_min)))
    # interpola su [0, 1]
    x0 = np.linspace(0.0, 1.0, num=len(template_15min))
    x1 = np.linspace(0.0, 1.0, num=n_steps)
    y = np.interp(x1, x0, template_15min.astype(float))
    y = np.maximum(y, 0.0)
    return y


def _scale_template_to_energy(template: np.ndarray, dt_hours: float, target_kwh: float, pmax_kw: float = 2.2) -> np.ndarray:
    """Scala un template di potenza per ottenere una data energia (kWh)."""
    e = float(template.sum() * dt_hours)
    if e <= 1e-9:
        return np.zeros_like(template)
    s = float(target_kwh / e)
    out = template * s
    # cap soft: se supera pmax, riscalo globalmente (mantiene forma, riduce energia)
    m = float(np.max(out)) if len(out) else 0.0
    if m > pmax_kw and m > 0:
        out = out * (pmax_kw / m)
    return out


def _meal_presence_weekly(meal_matrix: MealMatrix) -> Tuple[List[bool], List[bool]]:
    """Ritorna due liste di 7 boolean: lunch_home[dow], dinner_home[dow]."""
    lunch = [False] * 7
    dinner = [False] * 7
    for d in range(7):
        slots = meal_matrix.get(d, {})
        for _slot, meals in slots.items():
            if "lunch" in meals:
                lunch[d] = True
            if "dinner" in meals:
                dinner[d] = True
    return lunch, dinner


def dishwasher_power_series(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    meal_matrix: MealMatrix,
    seed: Optional[int] = None,
) -> pd.Series:
    """Serie di potenza della lavastoviglie su ``idx`` (kW).

    - Distribuisce ``cycles_per_week`` sui giorni della settimana in base alla presenza
      di pranzo/cena in ``meal_matrix`` (cena pesa di più).
    - Genera start time in locale (Europe/Rome) con Metodo A (orari canonici + jitter).
    - In modalità ``after_meal`` può agganciarsi sia al pranzo che alla cena.
    """
    if not getattr(cfg, "dishwasher", None) or not cfg.dishwasher.has_dishwasher or len(idx) == 0:
        return pd.Series(0.0, index=idx)

    idx = idx.sort_values()
    # porta l'indice in UTC tz-aware
    if idx.tz is None:
        idx_utc = idx.tz_localize("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")

    dt_hours = (idx_utc[1] - idx_utc[0]).total_seconds() / 3600.0 if len(idx_utc) > 1 else 0.25
    rng = np.random.default_rng(seed)

    # template -> dt effettivo
    program = str(cfg.dishwasher.program)
    duration_min = float(_DW_DURATION_MIN.get(program, 150.0))
    template_15 = _DW_TEMPLATES_15MIN.get(program, _DW_TEMPLATES_15MIN["standard"])
    base_template = _resample_template_to_dt(template_15, duration_min, dt_hours)

    # energia target (con lieve variabilità)
    energy_class = str(cfg.dishwasher.energy_class).upper()
    e_target = _dishwasher_target_energy_kwh(program, energy_class)
    # variabilità energia ±10% (cap ±20%), riproducibile
    eps_e = _clip_minutes(rng.normal(0.0, 0.10), 0.20)  # qui "minutes" è solo un clip numerico
    e_target = float(e_target * (1.0 + eps_e))
    shape = _scale_template_to_energy(base_template, dt_hours, e_target)

    # presenza pasti per giorno della settimana (pattern settimanale)
    lunch_home_dow, dinner_home_dow = _meal_presence_weekly(meal_matrix)
    alpha, beta = 1.0, 0.4

    w_dow = []
    for d in range(7):
        w = alpha * (1.0 if dinner_home_dow[d] else 0.0) + beta * (1.0 if lunch_home_dow[d] else 0.0)
        w_dow.append(w)
    # se tutto zero, uniform
    if sum(w_dow) <= 1e-9:
        w_dow = [1.0] * 7

    # date list in locale
    local = idx_utc.tz_convert("Europe/Rome")
    local_dates = pd.Index(local.normalize().unique()).sort_values()

    if len(local_dates) == 0:
        return pd.Series(0.0, index=idx_utc)

    # gruppo per settimana ISO (gestisce in modo naturale anni/periodi lunghi)
    iso = local_dates.isocalendar()
    df = pd.DataFrame({
        "date": local_dates,
        "dow": local_dates.dayofweek,
        "iso_year": iso["year"].astype(int).values,
        "iso_week": iso["week"].astype(int).values,
    })

    # memo jitter pasti per data (Metodo A)
    meal_times: Dict[pd.Timestamp, Dict[str, pd.Timestamp]] = {}

    def _meal_time_for(date_local_midnight: pd.Timestamp, which: str) -> pd.Timestamp:
        if date_local_midnight not in meal_times:
            meal_times[date_local_midnight] = {}
        if which in meal_times[date_local_midnight]:
            return meal_times[date_local_midnight][which]

        dow = int(date_local_midnight.dayofweek)
        is_weekend = dow >= 5

        if which == "lunch":
            base_h, base_m = (13, 15) if is_weekend else (12, 45)
            sigma = 20.0
            cap = 45.0
        else:  # dinner
            base_h, base_m = (20, 30) if is_weekend else (20, 0)
            sigma = 25.0
            cap = 60.0

        base = date_local_midnight + pd.Timedelta(hours=base_h, minutes=base_m)
        jitter = _clip_minutes(rng.normal(0.0, sigma), cap)  # minuti
        out = base + pd.Timedelta(minutes=float(jitter))
        out = _round_to_15min(out)
        meal_times[date_local_midnight][which] = out
        return out

    def _parse_fixed_time(s: str) -> Tuple[int, int]:
        try:
            parts = str(s).strip().split(":")
            hh = int(parts[0]); mm = int(parts[1]) if len(parts) > 1 else 0
            hh = max(0, min(hh, 23)); mm = max(0, min(mm, 59))
            return hh, mm
        except Exception:
            return 22, 30

    starts_utc: List[pd.Timestamp] = []

    Cw = max(0, int(cfg.dishwasher.cycles_per_week))
    if Cw <= 0:
        return pd.Series(0.0, index=idx_utc)

    # per ciascuna settimana ISO nel periodo simulato
    for (y, w), g in df.groupby(["iso_year", "iso_week"]):
        days = list(g["date"].to_list())
        dows = list(g["dow"].values)

        # peso solo sui giorni presenti
        weights = np.array([w_dow[int(d)] for d in dows], dtype=float)
        if weights.sum() <= 1e-9:
            weights = np.ones_like(weights, dtype=float)
        probs = weights / weights.sum()

        # scale per settimane parziali
        frac = float(len(days) / 7.0)
        expected = float(Cw * frac)
        n_cycles = int(math.floor(expected))
        if rng.random() < (expected - n_cycles):
            n_cycles += 1
        if n_cycles <= 0:
            continue

        counts = rng.multinomial(n_cycles, probs)

        # max 1 ciclo/giorno (redistribuisci eccedenze)
        counts = counts.astype(int)
        # prima tronca a 1 e contabilizza eccessi
        extra = int(np.sum(np.maximum(counts - 1, 0)))
        counts = np.minimum(counts, 1)
        if extra > 0:
            zero_idx = [i for i, c in enumerate(counts) if c == 0]
            # redistribuisci sugli zero, pesando per weights
            while extra > 0 and len(zero_idx) > 0:
                wz = np.array([weights[i] for i in zero_idx], dtype=float)
                if wz.sum() <= 1e-9:
                    wz = np.ones_like(wz)
                pz = wz / wz.sum()
                pick = int(rng.choice(zero_idx, p=pz))
                counts[pick] = 1
                zero_idx.remove(pick)
                extra -= 1

        for i, c in enumerate(counts):
            if int(c) <= 0:
                continue
            day_mid = pd.Timestamp(days[i])
            # days[i] può arrivare tz-naive (es. numpy.datetime64) a seconda di come viene estratto dal DataFrame.
            # In quel caso rappresenta la mezzanotte *locale* (Europe/Rome), quindi va tz_localize.
            if day_mid.tz is None:
                day_mid = day_mid.tz_localize("Europe/Rome")
            else:
                day_mid = day_mid.tz_convert("Europe/Rome")
            # midnight local
            dow = int(pd.Timestamp(days[i]).dayofweek)

            mode = str(cfg.dishwasher.mode)
            if mode == "scheduled_fixed_time":
                hh, mm = _parse_fixed_time(cfg.dishwasher.fixed_start_time)
                t_start_local = day_mid + pd.Timedelta(hours=hh, minutes=mm)
                t_start_local = _round_to_15min(t_start_local)
            else:
                # after_meal: scegli anchor (pranzo/cena) seguendo il pattern settimanale
                L = bool(lunch_home_dow[dow])
                D = bool(dinner_home_dow[dow])

                if D and L:
                    p_dinner = alpha / (alpha + beta)
                    anchor = "dinner" if rng.random() < p_dinner else "lunch"
                elif D:
                    anchor = "dinner"
                elif L:
                    anchor = "lunch"
                else:
                    anchor = "fallback"

                if anchor == "lunch":
                    t_anchor = _meal_time_for(day_mid, "lunch")
                elif anchor == "dinner":
                    t_anchor = _meal_time_for(day_mid, "dinner")
                else:
                    # fallback 21:00 locale
                    t_anchor = _round_to_15min(day_mid + pd.Timedelta(hours=21))

                # delay post pasto + jitter piccolo
                delta = _clip_minutes(rng.normal(60.0, 20.0), 45.0)  # minuti
                eps = _clip_minutes(rng.normal(0.0, 10.0), 20.0)     # minuti
                t_start_local = t_anchor + pd.Timedelta(minutes=float(delta + eps))
                t_start_local = _round_to_15min(t_start_local)

            t_start_utc = t_start_local.tz_convert("UTC")
            t_start_utc = _round_to_15min(t_start_utc)
            starts_utc.append(t_start_utc)

    power = pd.Series(0.0, index=idx_utc)

    dt = pd.Timedelta(hours=float(dt_hours))
    for ts in starts_utc:
        for i, val in enumerate(shape):
            t = ts + i * dt
            if t in power.index:
                power.loc[t] += float(val)

    # ritorna nello stesso tz dell'indice originale
    if idx.tz is None:
        return power.tz_convert(None)  # naive, coerente con input naive
    return power.tz_convert(idx.tz)
# STEP 5 — Cappa (condizionata a forno + induzione)
# ---------------------------------------------------------------------------

def _hood_prob_for_event(cfg: HoodConfig, cooking_intensity: str) -> float:
    k_hood = {
        "quasi_mai": 0.1,
        "piatti_importanti": 0.4,
        "spesso": 0.7,
        "quasi_sempre": 0.9,
    }.get(cfg.hood_use_habit, 0.7)

    if cfg.hood_cooking_type == "solo_fritture_griglia":
        k_type = 0.7 if cooking_intensity == "elaborata" else 0.2
    elif cfg.hood_cooking_type == "piatti_elaborati":
        k_type = 0.7 if cooking_intensity in ("misto", "elaborata") else 0.3
    else:
        k_type = 1.0

    return min(1.0, k_hood * k_type)


def hood_power_series(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    oven_power: pd.Series,
    induction_power: pd.Series,
) -> pd.Series:
    """Serie di potenza della cappa (kW), correlata a forno + induzione.

    Strategia semplificata:
    - si costruisce un segnale combinato ``oven + induction``;
    - gli intervalli con potenza > 0.2 kW sono interpretati come "sessioni di cottura";
    - su ciascun intervallo si decide (stocasticamente) se attivare la cappa e per quale
      frazione di durata.

    Nota di riproducibilità: la decisione stocastica usa il RNG globale di NumPy
    (``np.random.random``) e quindi non è controllata da seed.
    """
    if not cfg.hood.has_hood or len(idx) == 0:
        return pd.Series(0.0, index=idx)

    idx = idx.sort_values()
    dt_hours = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0

    power = pd.Series(0.0, index=idx)
    # Troviamo intervalli di "cottura" (forno o induzione attivi)
    combined = oven_power.add(induction_power, fill_value=0.0)
    active = combined > 0.2  # kW

    if not active.any():
        return power

    # Individua blocchi continui di attività
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start: Optional[pd.Timestamp] = None
    for t, is_active in active.items():
        if is_active and current_start is None:
            current_start = t
        elif not is_active and current_start is not None:
            segments.append((current_start, t))
            current_start = None
    if current_start is not None:
        segments.append((current_start, active.index[-1] + (idx[1]-idx[0])))

    # Per ogni segmento, decidi se usare la cappa e con quale durata
    for start, end in segments:
        duration_hours = (end - start).total_seconds() / 3600.0
        if duration_hours <= 0:
            continue

        # proxy cooking_intensity: se induzione+forno medi > soglia, considerala elaborata
        avg_p = combined.loc[(combined.index >= start) & (combined.index < end)].mean()
        if avg_p > 3.0:
            cook_intensity = "elaborata"
        elif avg_p > 1.5:
            cook_intensity = "misto"
        else:
            cook_intensity = "semplice"

        p_use = _hood_prob_for_event(cfg.hood, cook_intensity)
        if np.random.random() > p_use:
            continue

        if cfg.hood.hood_duration_relative == "solo_inizio":
            f_dur = 0.3
        elif cfg.hood.hood_duration_relative == "quasi_tutto":
            f_dur = 0.8
        else:
            f_dur = 0.5

        hood_dur_hours = max(dt_hours, f_dur * duration_hours)
        # posiziona cappa (inizio, centro, quasi tutto)
        if cfg.hood.hood_duration_relative == "solo_inizio":
            hood_start = start
        elif cfg.hood.hood_duration_relative == "quasi_tutto":
            hood_start = start + pd.Timedelta(hours=0.1 * duration_hours)
        else:
            hood_start = start + pd.Timedelta(hours=0.25 * duration_hours)

        n_steps = int(round(hood_dur_hours / dt_hours))
        for i in range(n_steps):
            t = hood_start + pd.Timedelta(hours=i * dt_hours)
            if t in power.index:
                power.loc[t] += cfg.hood.hood_power_w / 1000.0  # kW

    return power


# ---------------------------------------------------------------------------
# API high-level: profilo complessivo cucina
# ---------------------------------------------------------------------------

def build_kitchen_profiles(
    idx: pd.DatetimeIndex,
    cfg: KitchenConfig,
    n_slots_per_day: int = 10,
    meal_matrix_seed: Optional[int] = None,
    seed_oven: Optional[int] = None,
    seed_induction: Optional[int] = None,
    seed_microwave: Optional[int] = None,
    seed_dishwasher: Optional[int] = None,
) -> Dict[str, pd.Series]:
    """Costruisce i profili di carico cucina (kW) per un consumatore.

    Parameters
    ----------
    idx:
        Indice temporale equispaziato (nel simulatore tipicamente 15 minuti, UTC).
    cfg:
        Configurazione completa della cucina (abitudini + dispositivi).
    n_slots_per_day:
        Numero di slot giornalieri usati per la ``meal_matrix`` (default 10).
    meal_matrix_seed, seed_oven, seed_induction, seed_microwave:
        Seed indipendenti per isolare la casualità dei sottosistemi.

    Returns
    -------
    Dict[str, pd.Series]
        Dizionario di serie in kW: ``oven``, ``induction``, ``microwave``, ``dishwasher``, ``hood`` e ``aggregated``.
    """
    if len(idx) == 0:
        empty = pd.Series([], dtype=float)
        return {
            "oven": empty,
            "induction": empty,
            "microwave": empty,
            "dishwasher": empty,
            "hood": empty,
            "aggregated": empty,
        }

    idx = idx.sort_values()
    # 1) genera pasti
    meal_matrix = build_meal_matrix(cfg.habits, n_slots_per_day, seed=meal_matrix_seed)

    # 2) profili forno / induzione / microonde
    oven = oven_power_series(idx, cfg, meal_matrix, seed=seed_oven)
    induction = induction_power_series(idx, cfg, meal_matrix, seed=seed_induction)
    microwave = microwave_power_series(idx, cfg, meal_matrix, seed=seed_microwave)
    dishwasher = dishwasher_power_series(idx, cfg, meal_matrix, seed=seed_dishwasher)

    # 3) cappa dipende da forno+induzione
    hood = hood_power_series(idx, cfg, oven, induction)

    aggregated = oven.add(induction, fill_value=0.0)
    aggregated = aggregated.add(microwave, fill_value=0.0)
    aggregated = aggregated.add(dishwasher, fill_value=0.0)
    aggregated = aggregated.add(hood, fill_value=0.0)

    return {
        "oven": oven,
        "induction": induction,
        "microwave": microwave,
        "dishwasher": dishwasher,
        "hood": hood,
        "aggregated": aggregated,
    }
