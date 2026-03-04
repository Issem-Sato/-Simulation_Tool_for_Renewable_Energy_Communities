from __future__ import annotations

"""cer_core.consumatori.occupancy

Modello *Occupancy* e carichi dipendenti dall'uso (illuminazione, TV/streaming,
PC, ricarica smartphone).

Il modulo traduce una *settimana tipo* definita tramite matrice 7×24 (o 7×48)
in driver di presenza e attività, poi sintetizza una curva di potenza per i
carichi tipicamente correlati al comportamento degli occupanti.

Contratti e assunzioni
----------------------
Timebase
  - ``idx`` è un :class:`pandas.DatetimeIndex` a passo costante; nel simulatore
    CER corrente è **tz-aware UTC** a **15 minuti**.
  - La settimana tipo è definita in **tempo locale** (``cfg.timezone``, default
    ``Europe/Rome``). Il modulo converte ``idx`` in locale solo per:
    - applicare la matrice di stati per giorno della settimana;
    - stimare il buio (proxy sunrise/sunset stagionale).
  - Gli output ritornano sempre indicizzati come ``idx`` (quindi in UTC se
    l'input è in UTC).

Risoluzione della matrice
  - ``matrix_resolution_minutes`` può essere 60 (7×24) o 30 (7×48).
  - Se ``idx`` è più fine (es. 15 minuti), lo stato viene "replicato" su più
    step: ogni quarter-hour della stessa fascia oraria/mezzo-oraria condivide lo
    stesso stato.

Unità
  - Tutte le serie restituite sono **potenze** in **kW**.
  - L'energia (kWh) si ottiene a valle con ``sum(P_kW) * dt_hours``.

Riproducibilità
  - La stocasticità (variabilità giornaliera, allocazione sessioni, rumore
    luci) dipende da ``cfg.seed`` e usa :class:`numpy.random.Generator`.
  - A seed fissato e a parità di ``idx`` e configurazione, l'output è
    deterministico.

Failure modes (gestiti)
-----------------------
  - ``idx`` vuoto o ``cfg.present == False`` -> dizionario di serie vuote.
  - Matrici residenti invalide -> fallback su schedule di default.

Concetto chiave
---------------
L'utente definisce (per ciascun residente) una settimana tipo con 3 stati:

    0 = Away        (fuori casa)
    1 = Home-Awake  (a casa e sveglio)
    2 = Home-Sleep  (a casa e dorme)

Da questa settimana tipo si costruiscono driver di occupancy (``n_home``,
``n_awake``) e si simulano carichi condivisi/personali.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tipi e costanti
# ---------------------------------------------------------------------------

StateCode = int  # 0=Away, 1=Home-Awake, 2=Home-Sleep

VariabilityLevel = Literal["off", "low", "medium", "high"]
PCType = Literal["none", "laptop", "desktop"]
Intensity3 = Literal["low", "medium", "high"]
TimePref = Literal["morning", "afternoon", "evening", "mixed"]
ChargingPref = Literal["night", "return_home", "distributed"]
ChargeType = Literal["standard", "fast"]

TVTimePref = Literal["weekday_evening", "weekend", "both"]
TVPowerClass = Literal["efficient", "standard", "large"]

LightingTech = Literal["led", "mixed", "halogen"]
LightingStyle = Literal["minimal", "standard", "intense"]
LightingSwitchPref = Literal["as_soon_as_dark", "evening_only", "frugal"]


# Mapping numerici (default ragionevoli) ------------------------------------

PC_POWER_W = {
    "laptop": (60.0, 20.0),   # mean, sd
    "desktop": (180.0, 60.0),
}

PC_HOURS_WEEK = {"low": 3.0, "medium": 10.0, "high": 25.0}

PC_DUR_BLOCKS = {
    # durata sessione in blocchi da 15 min (min,max)
    "low": (2, 4),     # 30–60 min
    "medium": (4, 8),  # 1–2 h
    "high": (6, 12),   # 1.5–3 h
}

PHONE_E_DAY_WH = {"low": 6.0, "medium": 12.0, "high": 20.0}
PHONE_CHARGE_W = {"standard": 10.0, "fast": 20.0}

TV_HOURS_WEEK = {"low": 6.0, "medium": 14.0, "high": 28.0}
TV_POWER_W = {"efficient": 70.0, "standard": 120.0, "large": 200.0}

LIGHT_TECH_FACTOR = {"led": 1.0, "mixed": 1.5, "halogen": 2.5}
LIGHT_W_PER_PERSON = {"minimal": 15.0, "standard": 30.0, "intense": 60.0}
LIGHT_EXTRA_W = {"minimal": 30.0, "standard": 80.0, "intense": 150.0}

VAR_NOISE_SIGMA = {"off": 0.0, "low": 0.05, "medium": 0.10, "high": 0.15}


# ---------------------------------------------------------------------------
# Dataclass di configurazione
# ---------------------------------------------------------------------------


@dataclass
class ResidentScheduleConfig:
    state_matrix_7x24: List[List[StateCode]] = field(default_factory=list)


@dataclass
class ResidentPCConfig:
    type: PCType = "none"
    intensity: Intensity3 = "low"
    time_pref: TimePref = "mixed"
    weekend_enabled: bool = True


@dataclass
class ResidentChargingConfig:
    phone_intensity: Intensity3 = "medium"
    preference: ChargingPref = "night"
    charge_type: ChargeType = "standard"


@dataclass
class ResidentConfig:
    rid: int = 0
    label: str = "Residente"
    schedule: ResidentScheduleConfig = field(default_factory=ResidentScheduleConfig)
    pc: ResidentPCConfig = field(default_factory=ResidentPCConfig)
    charging: ResidentChargingConfig = field(default_factory=ResidentChargingConfig)


@dataclass
class TVConfig:
    intensity: Intensity3 = "medium"
    time_pref: TVTimePref = "both"
    tv_count: int = 1
    power_class: TVPowerClass = "standard"


@dataclass
class LightingConfig:
    tech: LightingTech = "led"
    style: LightingStyle = "standard"
    switch_pref: LightingSwitchPref = "as_soon_as_dark"
    twilight_minutes: int = 45


@dataclass
class VariabilityConfig:
    level: VariabilityLevel = "medium"


@dataclass
class SharedConfig:
    tv: TVConfig = field(default_factory=TVConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)


@dataclass
class OccupancyConfig:
    """Configurazione complessiva del modulo occupancy.

    Nota: `seed` viene in genere derivato a monte (da seed di sessione)
    e passato qui per riproducibilità.
    """

    present: bool = True
    version: int = 1
    timezone: str = "Europe/Rome"
    matrix_resolution_minutes: int = 60
    variability: VariabilityConfig = field(default_factory=VariabilityConfig)
    residents: List[ResidentConfig] = field(default_factory=list)
    shared: SharedConfig = field(default_factory=SharedConfig)
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict | None, n_residents: int, seed: int = 0) -> "OccupancyConfig":
        """Costruisce una config robusta da dict (tipicamente da consumers.json).

        - applica default per campi mancanti
        - padding/troncamento lista residenti in base a `n_residents`
        """
        data = data or {}

        tz = str(data.get("timezone") or "Europe/Rome")
        res_min = int(data.get("matrix_resolution_minutes") or 60)
        if res_min not in (60, 30):
            res_min = 60

        var_level = (data.get("variability") or {}).get("level", "medium")
        if var_level not in ("off", "low", "medium", "high"):
            var_level = "medium"

        # shared
        shared = data.get("shared") or {}
        tv = shared.get("tv") or {}
        lighting = shared.get("lighting") or {}
        tv_cfg = TVConfig(
            intensity=str(tv.get("intensity") or "medium"),
            time_pref=str(tv.get("time_pref") or "both"),
            tv_count=int(tv.get("tv_count") or 1),
            power_class=str(tv.get("power_class") or "standard"),
        )
        if tv_cfg.intensity not in ("low", "medium", "high"):
            tv_cfg.intensity = "medium"
        if tv_cfg.time_pref not in ("weekday_evening", "weekend", "both"):
            tv_cfg.time_pref = "both"
        if tv_cfg.power_class not in ("efficient", "standard", "large"):
            tv_cfg.power_class = "standard"
        tv_cfg.tv_count = max(int(tv_cfg.tv_count or 1), 1)

        light_cfg = LightingConfig(
            tech=str(lighting.get("tech") or "led"),
            style=str(lighting.get("style") or "standard"),
            switch_pref=str(lighting.get("switch_pref") or "as_soon_as_dark"),
            twilight_minutes=int(lighting.get("twilight_minutes") or 45),
        )
        if light_cfg.tech not in ("led", "mixed", "halogen"):
            light_cfg.tech = "led"
        if light_cfg.style not in ("minimal", "standard", "intense"):
            light_cfg.style = "standard"
        if light_cfg.switch_pref not in ("as_soon_as_dark", "evening_only", "frugal"):
            light_cfg.switch_pref = "as_soon_as_dark"
        light_cfg.twilight_minutes = int(max(min(light_cfg.twilight_minutes, 120), 0))

        shared_cfg = SharedConfig(tv=tv_cfg, lighting=light_cfg)

        # residents
        residents_in = data.get("residents") or []
        residents_out: List[ResidentConfig] = []
        for i in range(min(len(residents_in), n_residents)):
            r = residents_in[i] or {}
            rid = int(r.get("rid") if r.get("rid") is not None else i)
            label = str(r.get("label") or f"Residente {i+1}")

            # schedule
            sched = (r.get("schedule") or {})
            mat = sched.get("state_matrix_7x24")
            mat_norm = _normalize_state_matrix(mat, res_min)
            schedule_cfg = ResidentScheduleConfig(state_matrix_7x24=mat_norm)

            # pc
            pc = r.get("pc") or {}
            pc_cfg = ResidentPCConfig(
                type=str(pc.get("type") or "none"),
                intensity=str(pc.get("intensity") or "low"),
                time_pref=str(pc.get("time_pref") or "mixed"),
                weekend_enabled=bool(pc.get("weekend_enabled", True)),
            )
            if pc_cfg.type not in ("none", "laptop", "desktop"):
                pc_cfg.type = "none"
            if pc_cfg.intensity not in ("low", "medium", "high"):
                pc_cfg.intensity = "low"
            if pc_cfg.time_pref not in ("morning", "afternoon", "evening", "mixed"):
                pc_cfg.time_pref = "mixed"

            # charging
            ch = r.get("charging") or {}
            ch_cfg = ResidentChargingConfig(
                phone_intensity=str(ch.get("phone_intensity") or "medium"),
                preference=str(ch.get("preference") or "night"),
                charge_type=str(ch.get("charge_type") or "standard"),
            )
            if ch_cfg.phone_intensity not in ("low", "medium", "high"):
                ch_cfg.phone_intensity = "medium"
            if ch_cfg.preference not in ("night", "return_home", "distributed"):
                ch_cfg.preference = "night"
            if ch_cfg.charge_type not in ("standard", "fast"):
                ch_cfg.charge_type = "standard"

            residents_out.append(
                ResidentConfig(rid=rid, label=label, schedule=schedule_cfg, pc=pc_cfg, charging=ch_cfg)
            )

        # padding se necessario
        for i in range(len(residents_out), n_residents):
            residents_out.append(_default_resident(i))

        return cls(
            present=bool(data.get("present", True)),
            version=int(data.get("version") or 1),
            timezone=tz,
            matrix_resolution_minutes=res_min,
            variability=VariabilityConfig(level=var_level),
            residents=residents_out,
            shared=shared_cfg,
            seed=int(seed or 0),
        )


# ---------------------------------------------------------------------------
# Default schedule/resident
# ---------------------------------------------------------------------------


def _default_state_matrix() -> List[List[StateCode]]:
    """Template 7x24: feriali fuori 9-18, weekend prevalentemente a casa."""
    out: List[List[StateCode]] = []
    for dow in range(7):
        row: List[StateCode] = [2] * 24  # sleep
        if dow <= 4:  # Mon-Fri
            for h in range(7, 9):
                row[h] = 1  # awake
            for h in range(9, 18):
                row[h] = 0  # away
            for h in range(18, 23):
                row[h] = 1  # home awake
            row[23] = 2
        else:  # weekend
            for h in range(8, 23):
                row[h] = 1
            row[23] = 2
        out.append(row)
    return out


def _default_resident(i: int) -> ResidentConfig:
    return ResidentConfig(
        rid=i,
        label=f"Residente {i+1}",
        schedule=ResidentScheduleConfig(state_matrix_7x24=_default_state_matrix()),
        pc=ResidentPCConfig(type="laptop", intensity="low", time_pref="mixed", weekend_enabled=True),
        charging=ResidentChargingConfig(phone_intensity="medium", preference="night", charge_type="standard"),
    )


def _normalize_state_matrix(mat: object, res_min: int = 60) -> List[List[StateCode]]:
    """Normalizza la matrice 7x24 di stati.

    Se input invalido, ritorna un default.
    """
    slots = 24 if int(res_min) == 60 else 48
    if not isinstance(mat, list) or len(mat) != 7:
        return _default_state_matrix()

    out: List[List[StateCode]] = []
    for r in mat:
        if not isinstance(r, list):
            return _default_state_matrix()
        row = [int(x) if x is not None else 0 for x in r]
        if len(row) < slots:
            row = row + [row[-1] if row else 2] * (slots - len(row))
        if len(row) > slots:
            row = row[:slots]
        # clamp
        row = [0 if x < 0 else 2 if x > 2 else x for x in row]
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Helper: grouping per giorno locale
# ---------------------------------------------------------------------------


def _group_positions_by_local_date(idx_local: pd.DatetimeIndex) -> Dict[pd.Timestamp, np.ndarray]:
    """Mappa date (midnight locale) -> posizioni nell'indice."""
    if len(idx_local) == 0:
        return {}
    dates = idx_local.normalize()
    groups: Dict[pd.Timestamp, List[int]] = {}
    for i, d in enumerate(dates):
        groups.setdefault(d, []).append(i)
    return {d: np.asarray(pos, dtype=int) for d, pos in groups.items()}


# ---------------------------------------------------------------------------
# Driver occupancy (state/home/awake) con jitter
# ---------------------------------------------------------------------------


def _shift_row_keep_edges(row: List[int], delta: int) -> List[int]:
    if delta == 0:
        return list(row)
    n = len(row)
    if delta > 0:
        return [row[0]] * delta + row[: n - delta]
    # delta < 0
    d = -delta
    return row[d:] + [row[-1]] * d


def _apply_variability_to_row(
    base_row: List[StateCode],
    rng: np.random.Generator,
    level: VariabilityLevel,
) -> List[StateCode]:
    """Applica piccole perturbazioni a una riga 24h (stile "jitter").

    Obiettivo: evitare settimane identiche senza stravolgere la settimana tipo.
    """
    if level == "off":
        return list(base_row)

    # probabilità/forza perturbazioni per giorno
    if level == "low":
        p_shift, p_flip = 0.05, 0.02
    elif level == "medium":
        p_shift, p_flip = 0.10, 0.05
    else:  # high
        p_shift, p_flip = 0.20, 0.10

    row = list(base_row)

    # Shift del giorno (±1h)
    if rng.random() < p_shift:
        delta = int(rng.choice([-1, 0, 1], p=[0.35, 0.30, 0.35]))
        row = _shift_row_keep_edges(row, delta)

    # Micro-eccezione: flip di 1 ora away<->home_awake (solo in fascia diurna)
    if rng.random() < p_flip:
        h = int(rng.integers(9, 18))
        if row[h] == 0:
            row[h] = 1
        elif row[h] == 1:
            row[h] = 0

    return row


def _build_resident_state_series(
    idx_local: pd.DatetimeIndex,
    matrix_7x24: List[List[StateCode]],
    res_min: int,
    variability_level: VariabilityLevel,
    rng: np.random.Generator,
) -> pd.Series:
    """Costruisce la serie di stati (0/1/2) per un residente su idx_local."""
    if len(idx_local) == 0:
        return pd.Series([], dtype=int)

    day_map = _group_positions_by_local_date(idx_local)
    state = np.zeros(len(idx_local), dtype=int)

    for day0, pos in day_map.items():
        dow = int(day0.dayofweek)  # 0=Mon
        base_row = matrix_7x24[dow]
        row = _apply_variability_to_row(base_row, rng=rng, level=variability_level)

        # assegnazione per timestamp
        ts = idx_local[pos]
        if int(res_min) == 30:
            slots = ts.hour.to_numpy(dtype=int) * 2 + (ts.minute.to_numpy(dtype=int) // 30)
        else:
            slots = ts.hour.to_numpy(dtype=int)

        for j, s in enumerate(slots):
            state[pos[j]] = row[int(s)]

    return pd.Series(state, index=idx_local, dtype=int)


# ---------------------------------------------------------------------------
# Driver buio (proxy stagionale sunrise/sunset)
# ---------------------------------------------------------------------------


def _compute_is_dark(idx_local: pd.DatetimeIndex, twilight_minutes: int) -> pd.Series:
    """Stima se è "buio" con un proxy stagionale.

    day_length = 12 + 4*sin(2*pi*(doy-80)/365)  -> range ~[8,16] ore
    sunrise = 12 - day_length/2
    sunset  = 12 + day_length/2

    twilight_minutes estende la finestra di buio intorno a sunrise/sunset.
    """
    if len(idx_local) == 0:
        return pd.Series([], dtype=int)

    tw_h = max(float(twilight_minutes), 0.0) / 60.0
    doy = idx_local.dayofyear.to_numpy(dtype=float)
    frac = (doy - 80.0) / 365.0
    day_len = 12.0 + 4.0 * np.sin(2.0 * np.pi * frac)
    sunrise = 12.0 - day_len / 2.0
    sunset = 12.0 + day_len / 2.0

    tod = idx_local.hour.to_numpy(dtype=float) + idx_local.minute.to_numpy(dtype=float) / 60.0
    # Buio se prima di sunrise-tw oppure dopo sunset+tw
    dark = (tod < (sunrise - tw_h)) | (tod >= (sunset + tw_h))
    return pd.Series(dark.astype(int), index=idx_local, dtype=int)


# ---------------------------------------------------------------------------
# Generazione sessioni (PC / TV)
# ---------------------------------------------------------------------------


def _time_weights(hours: np.ndarray, pref: TimePref) -> np.ndarray:
    """Pesi per selezione start-time in base a preferenza."""
    w = np.ones_like(hours, dtype=float)
    if pref == "morning":
        w[(hours >= 8) & (hours < 12)] = 3.0
        w[(hours >= 19) & (hours < 23)] = 0.8
    elif pref == "afternoon":
        w[(hours >= 13) & (hours < 18)] = 3.0
        w[(hours >= 8) & (hours < 12)] = 0.8
    elif pref == "evening":
        w[(hours >= 19) & (hours < 23)] = 3.0
        w[(hours >= 8) & (hours < 12)] = 0.8
    else:
        # mixed
        w[(hours >= 8) & (hours < 23)] = 1.2
    return w


def _generate_daily_sessions(
    pos: np.ndarray,
    allow_mask: np.ndarray,
    idx_local: pd.DatetimeIndex,
    rng: np.random.Generator,
    expected_hours: float,
    dur_blocks_minmax: Tuple[int, int],
    pref: TimePref,
) -> List[Tuple[int, int]]:
    """Ritorna lista di sessioni come coppie (start_pos_idx, end_pos_idx_exclusive).

    Le posizioni sono riferite all'indice globale (posizioni in idx).
    """
    if expected_hours <= 0.0:
        return []
    if len(pos) == 0:
        return []

    # massima disponibilità (in blocchi da 15 min)
    allow = allow_mask[pos]
    if not allow.any():
        return []

    # target blocchi
    mean = expected_hours
    sd = max(0.2, 0.35 * mean)
    hours_today = float(np.clip(rng.normal(mean, sd), 0.0, 24.0))
    blocks_target = int(round(hours_today * 4))
    if blocks_target <= 0:
        return []

    # vincolo: non oltre blocchi disponibili
    blocks_available = int(allow.sum())
    blocks_target = min(blocks_target, blocks_available)

    # candidati start: posizioni dove allow==1
    cand_local_idx = np.where(allow)[0]
    if len(cand_local_idx) == 0:
        return []
    cand_pos = pos[cand_local_idx]
    cand_hours = idx_local[cand_pos].hour.to_numpy()
    w = _time_weights(cand_hours, pref)
    w = w / w.sum() if w.sum() > 0 else None

    min_d, max_d = dur_blocks_minmax
    used = np.zeros(len(pos), dtype=bool)
    sessions: List[Tuple[int, int]] = []

    # tentativi di allocazione: limitiamo per evitare loop lunghi
    attempts = 0
    while blocks_target > 0 and attempts < 2000:
        attempts += 1
        dur = int(rng.integers(min_d, max_d + 1))
        dur = min(dur, blocks_target)
        # scegli start tra candidati
        si = int(rng.choice(len(cand_pos), p=w))
        start_global = int(cand_pos[si])

        # start_global deve ricadere nel sotto-array di pos: troviamo l'indice locale
        # (pos è ordinato crescente, quindi possiamo usare searchsorted)
        start_local = int(np.searchsorted(pos, start_global))
        end_local = start_local + dur
        if end_local > len(pos):
            continue
        # vincoli: tutta la durata deve essere allow e non usata
        if not allow[start_local:end_local].all():
            continue
        if used[start_local:end_local].any():
            continue
        used[start_local:end_local] = True
        sessions.append((start_global, int(pos[end_local - 1]) + 1))
        blocks_target -= dur

    return sessions


# ---------------------------------------------------------------------------
# Carichi: PC, Charging, TV, Luci
# ---------------------------------------------------------------------------


def _build_pc_profiles(
    idx: pd.DatetimeIndex,
    idx_local: pd.DatetimeIndex,
    resident_states: List[pd.Series],
    cfg: OccupancyConfig,
    rng: np.random.Generator,
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Ritorna dict per-residente + totale PC (kW)."""
    out: Dict[str, pd.Series] = {}
    total = pd.Series(0.0, index=idx, dtype=float)

    positions_by_day = _group_positions_by_local_date(idx_local)

    for r_i, r_cfg in enumerate(cfg.residents):
        pc_cfg = r_cfg.pc
        if pc_cfg.type == "none":
            s = pd.Series(0.0, index=idx, dtype=float)
            out[f"pc_r{r_i}"] = s
            continue

        mean_w, sd_w = PC_POWER_W[pc_cfg.type]
        hours_week = float(PC_HOURS_WEEK.get(pc_cfg.intensity, 3.0))

        # distribuiamo su 7 giorni: weekend solo se abilitato
        weekday_expected = hours_week / 7.0

        awake_mask = (resident_states[r_i].to_numpy() == 1)  # Home-Awake
        power = np.zeros(len(idx), dtype=float)

        # session params
        dur_minmax = PC_DUR_BLOCKS.get(pc_cfg.intensity, (2, 4))

        for day0, pos in positions_by_day.items():
            dow = int(day0.dayofweek)
            is_weekend = dow >= 5
            if is_weekend and (not bool(pc_cfg.weekend_enabled)):
                expected = 0.0
            else:
                expected = weekday_expected

            sessions = _generate_daily_sessions(
                pos=pos,
                allow_mask=awake_mask,
                idx_local=idx_local,
                rng=rng,
                expected_hours=expected,
                dur_blocks_minmax=dur_minmax,
                pref=pc_cfg.time_pref,
            )
            for start_g, end_g in sessions:
                # potenza sessione
                p_w = float(np.clip(rng.normal(mean_w, sd_w), 0.2 * mean_w, 2.0 * mean_w))
                power[start_g:end_g] = np.maximum(power[start_g:end_g], p_w / 1000.0)

        s = pd.Series(power, index=idx, dtype=float)
        out[f"pc_r{r_i}"] = s
        total = total.add(s, fill_value=0.0)

    return out, total


def _build_charging_profiles(
    idx: pd.DatetimeIndex,
    idx_local: pd.DatetimeIndex,
    resident_states: List[pd.Series],
    cfg: OccupancyConfig,
    rng: np.random.Generator,
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Ritorna dict per-residente + totale charging (kW)."""
    out: Dict[str, pd.Series] = {}
    total = pd.Series(0.0, index=idx, dtype=float)
    positions_by_day = _group_positions_by_local_date(idx_local)

    for r_i, r_cfg in enumerate(cfg.residents):
        ch = r_cfg.charging
        e_day = float(PHONE_E_DAY_WH.get(ch.phone_intensity, 12.0))
        p_w = float(PHONE_CHARGE_W.get(ch.charge_type, 10.0))
        blocks_needed = int(math.ceil((e_day / max(p_w, 1e-6)) / 0.25))  # 0.25h per step
        if blocks_needed <= 0:
            s = pd.Series(0.0, index=idx, dtype=float)
            out[f"charging_r{r_i}"] = s
            continue

        state = resident_states[r_i].to_numpy(dtype=int)
        power = np.zeros(len(idx), dtype=float)

        for day0, pos in positions_by_day.items():
            if len(pos) == 0:
                continue

            # candidate mask per preferenza
            if ch.preference == "night":
                cand = (state[pos] == 2)  # sleep
            elif ch.preference == "return_home":
                # finestre dopo rientro (Away->Home-Awake)
                st_day = state[pos]
                trans = np.where((st_day[1:] != st_day[:-1]) & (st_day[:-1] == 0) & (st_day[1:] == 1))[0]
                cand = np.zeros(len(pos), dtype=bool)
                for t0 in trans:
                    # 3 ore dopo il rientro
                    t1 = min(len(pos), t0 + 1 + 12)
                    cand[t0 + 1 : t1] = True
                # vincolo: deve essere a casa (awake o sleep)
                cand &= (st_day != 0)
            else:  # distributed
                st_day = state[pos]
                cand = (st_day == 1)  # awake
                # piccola preferenza sera
                hours = idx_local[pos].hour.to_numpy()
                cand = cand & ((hours >= 6) & (hours <= 23))

            if not cand.any():
                # fallback: se night non disponibile usa awake serale
                st_day = state[pos]
                hours = idx_local[pos].hour.to_numpy()
                cand = (st_day == 1) & (hours >= 18)
            if not cand.any():
                continue

            cand_idx = np.where(cand)[0]
            # scegli un start: per night preferisci inizio del primo segmento sleep
            if ch.preference == "night":
                start_local = int(cand_idx[0])
            else:
                start_local = int(rng.choice(cand_idx))

            # allochiamo blocchi in modo greedy lungo i candidati contigui
            remaining = blocks_needed
            cur = start_local
            while remaining > 0 and cur < len(pos):
                if not cand[cur]:
                    cur += 1
                    continue
                # estendi segmento contiguo
                seg_end = cur
                while seg_end < len(pos) and cand[seg_end]:
                    seg_end += 1
                seg_len = seg_end - cur
                take = min(seg_len, remaining)
                g0 = int(pos[cur])
                g1 = int(pos[cur + take - 1]) + 1
                power[g0:g1] = np.maximum(power[g0:g1], p_w / 1000.0)
                remaining -= take
                cur = seg_end

        s = pd.Series(power, index=idx, dtype=float)
        out[f"charging_r{r_i}"] = s
        total = total.add(s, fill_value=0.0)

    return out, total


def _build_tv_profile(
    idx: pd.DatetimeIndex,
    idx_local: pd.DatetimeIndex,
    n_awake: pd.Series,
    cfg: OccupancyConfig,
    rng: np.random.Generator,
) -> pd.Series:
    tv = cfg.shared.tv
    hours_week = float(TV_HOURS_WEEK.get(tv.intensity, 14.0))
    base_p_w = float(TV_POWER_W.get(tv.power_class, 120.0))
    tv_count = max(int(tv.tv_count or 1), 1)

    positions_by_day = _group_positions_by_local_date(idx_local)
    awake_mask = (n_awake.to_numpy() > 0)
    power = np.zeros(len(idx), dtype=float)

    # ripartizione ore tra weekday/weekend
    if tv.time_pref == "weekday_evening":
        frac_weekday, frac_weekend = 0.8, 0.2
    elif tv.time_pref == "weekend":
        frac_weekday, frac_weekend = 0.3, 0.7
    else:
        frac_weekday, frac_weekend = 0.6, 0.4

    exp_weekday = (hours_week * frac_weekday) / 5.0
    exp_weekend = (hours_week * frac_weekend) / 2.0

    for day0, pos in positions_by_day.items():
        dow = int(day0.dayofweek)
        is_weekend = dow >= 5
        expected = exp_weekend if is_weekend else exp_weekday

        # finestra time preference
        hours = idx_local[pos].hour.to_numpy()
        if tv.time_pref == "weekday_evening":
            allow_time = (hours >= 19) & (hours < 23)
        elif tv.time_pref == "weekend":
            allow_time = (hours >= 12) & (hours < 23)
        else:
            allow_time = (hours >= 19) & (hours < 23) if not is_weekend else (hours >= 12) & (hours < 23)

        allow_day = awake_mask[pos] & allow_time
        if not allow_day.any():
            continue

        # _generate_daily_sessions si aspetta una maschera di lunghezza=len(idx)
        allow_global = np.zeros(len(idx), dtype=bool)
        allow_global[pos] = allow_day

        sessions = _generate_daily_sessions(
            pos=pos,
            allow_mask=allow_global,
            idx_local=idx_local,
            rng=rng,
            expected_hours=expected,
            dur_blocks_minmax=(4, 12),
            pref="evening" if not is_weekend else "mixed",
        )

        for start_g, end_g in sessions:
            n_tv = 1
            if tv_count >= 2:
                # seconda TV rara: solo se in media ci sono >=3 persone sveglie nella sessione
                mean_aw = float(np.mean(n_awake.to_numpy()[start_g:end_g]))
                if mean_aw >= 3 and rng.random() < 0.10:
                    n_tv = min(2, tv_count)
            power[start_g:end_g] = np.maximum(power[start_g:end_g], (n_tv * base_p_w) / 1000.0)

    return pd.Series(power, index=idx, dtype=float)


def _build_lighting_profile(
    idx: pd.DatetimeIndex,
    idx_local: pd.DatetimeIndex,
    n_awake: pd.Series,
    cfg: OccupancyConfig,
    rng: np.random.Generator,
) -> Tuple[pd.Series, pd.Series]:
    """Ritorna (lighting_kW, is_dark_int)."""
    light = cfg.shared.lighting
    tech_fac = float(LIGHT_TECH_FACTOR.get(light.tech, 1.0))
    w_per_person = float(LIGHT_W_PER_PERSON.get(light.style, 30.0))
    extra_w = float(LIGHT_EXTRA_W.get(light.style, 80.0))

    is_dark = _compute_is_dark(idx_local, twilight_minutes=light.twilight_minutes)
    dark = is_dark.to_numpy(dtype=int)
    awake = n_awake.to_numpy(dtype=float)
    hours = idx_local.hour.to_numpy()

    # switch pref
    if light.switch_pref == "evening_only":
        dark = dark * (hours >= 18)
        frugal_fac = 1.0
    elif light.switch_pref == "frugal":
        frugal_fac = 0.6
    else:
        frugal_fac = 1.0

    evening = ((hours >= 18) & (hours < 23)).astype(float)

    p_base_w = dark * awake * w_per_person * tech_fac
    p_extra_w = dark * evening * (awake > 0).astype(float) * extra_w * tech_fac

    p_w = (p_base_w + p_extra_w) * frugal_fac
    p_kw = p_w / 1000.0

    # rumore (dipende dal livello di variabilità globale)
    sigma = float(VAR_NOISE_SIGMA.get(cfg.variability.level, 0.10))
    if sigma > 0:
        noise = 1.0 + rng.normal(0.0, sigma, size=len(idx))
        noise = np.clip(noise, 0.7, 1.3)
        p_kw = p_kw * noise

    return pd.Series(p_kw, index=idx, dtype=float), is_dark


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------


def build_occupancy_profiles(idx: pd.DatetimeIndex, cfg: OccupancyConfig) -> Dict[str, pd.Series]:
    """Simula occupancy + carichi dipendenti.

    Parameters
    ----------
    idx:
        DatetimeIndex (tipicamente 15 minuti) dell'orizzonte di simulazione.
        Nel simulatore corrente è tz-aware UTC.
    cfg:
        OccupancyConfig.

    Returns
    -------
    dict con chiavi principali:
        - "lighting", "tv", "pc_total", "charging_total", "aggregated"
        - per-residente: "pc_r{i}", "charging_r{i}"
        - diagnostica: "n_home", "n_awake", "is_dark"
    """
    if len(idx) == 0 or (not bool(cfg.present)):
        empty_f = pd.Series([], dtype=float)
        empty_i = pd.Series([], dtype=int)
        return {
            "lighting": empty_f,
            "tv": empty_f,
            "pc_total": empty_f,
            "charging_total": empty_f,
            "aggregated": empty_f,
            "n_home": empty_i,
            "n_awake": empty_i,
            "is_dark": empty_i,
        }

    if idx.tz is None:
        # Assumiamo già locale se naive
        idx_local = idx
    else:
        # conversione a timezone locale
        idx_local = idx.tz_convert(cfg.timezone)

    rng_master = np.random.default_rng(int(cfg.seed or 0))

    # --- driver per residente ---
    resident_states: List[pd.Series] = []
    for r_i, r in enumerate(cfg.residents):
        rng_r = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        st = _build_resident_state_series(
            idx_local=idx_local,
            matrix_7x24=_normalize_state_matrix(r.schedule.state_matrix_7x24, cfg.matrix_resolution_minutes),
            res_min=cfg.matrix_resolution_minutes,
            variability_level=cfg.variability.level,
            rng=rng_r,
        )
        resident_states.append(st)

    # home/awake aggregati
    home_arr = np.vstack([(st.to_numpy(dtype=int) != 0).astype(int) for st in resident_states])
    awake_arr = np.vstack([(st.to_numpy(dtype=int) == 1).astype(int) for st in resident_states])
    n_home = pd.Series(home_arr.sum(axis=0), index=idx, dtype=int)
    n_awake = pd.Series(awake_arr.sum(axis=0), index=idx, dtype=int)

    # --- luci ---
    rng_light = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    lighting, is_dark = _build_lighting_profile(idx, idx_local, n_awake, cfg, rng_light)

    # --- tv ---
    rng_tv = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    tv = _build_tv_profile(idx, idx_local, n_awake, cfg, rng_tv)

    # --- pc per residente ---
    rng_pc = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    pc_res, pc_total = _build_pc_profiles(idx, idx_local, resident_states, cfg, rng_pc)

    # --- charging per residente ---
    rng_ch = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    ch_res, ch_total = _build_charging_profiles(idx, idx_local, resident_states, cfg, rng_ch)

    aggregated = (
        lighting.add(tv, fill_value=0.0)
        .add(pc_total, fill_value=0.0)
        .add(ch_total, fill_value=0.0)
    )

    out: Dict[str, pd.Series] = {
        "lighting": lighting,
        "tv": tv,
        "pc_total": pc_total,
        "charging_total": ch_total,
        "aggregated": aggregated,
        "n_home": n_home,
        "n_awake": n_awake,
        "is_dark": is_dark.astype(int).reindex(idx, fill_value=0),
    }
    out.update(pc_res)
    out.update(ch_res)
    return out
