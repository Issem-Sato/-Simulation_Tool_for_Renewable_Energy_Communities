"""cer_core.consumatori.clima

Questo modulo implementa un modello *compatto* per stimare i carichi elettrici legati al
"clima" domestico:

1) **Space heating** (riscaldamento ambiente) tramite:
   - pompe di calore aria-aria / aria-acqua (COP funzione della temperatura esterna),
   - riscaldamento elettrico diretto (COP≈1),
   - pavimento radiante elettrico con inerzia (smorzamento della richiesta).

2) **Space cooling** (raffrescamento) tramite split aria-aria (EER funzione della temperatura
   esterna).

3) **DHW / ACS** (acqua calda sanitaria) tramite:
   - boiler elettrico (COP≈1),
   - pompa di calore per ACS (COP funzione della temperatura esterna).

Il cuore del modulo è :func:`build_climate_profiles`, che restituisce un dizionario di
serie temporali.

## Contratti e convenzioni

- **Timebase**: tutte le serie sono indicizzate su un ``pd.DatetimeIndex`` equispaziato.
  Nella pipeline CER l'indice è tipicamente a **15 minuti** e timezone-aware in **UTC**.
  Il modulo non forza la timezone: richiede solo che l'indice sia ordinato e con passo
  costante (o quasi-costante).

- **Unità**:
  - tutte le curve *elettriche* in output sono in **kW_el** (potenza media sul passo);
  - le variabili con suffisso ``_th`` sono in **kW_th** (termico);
  - temperature in **°C**.

- **Energia**: l'energia si ottiene esternamente come ``E_kWh = sum(P_kW) * dt_hours``.
  Questo modulo produce potenze (non integra energia).

- **Riproducibilità**: questo modulo, a differenza di altri consumatori (baseload, occupancy,
  cucina, lavanderia), *non* usa RNG. A parità di input e configurazione produce risultati
  deterministici.

## Nota sul modello dinamico (2R2C)

Il ramo principale usa un modello termico lumped **2R2C** (aria interna + massa) con
integrazione esplicita (Euler) e *sub-stepping* interno (tipicamente 5 minuti) per
migliorare la stabilità numerica.

Una modalità legacy denominata ``static_UA`` (domanda proporzionale a UA) è ancora presente
nel codice per retro-compatibilità concettuale; tuttavia, **nell'implementazione attuale**
la variabile ``model`` viene forzata a ``"2R2C"`` per consistenza/stabilità (quindi il ramo
``static_UA`` non viene eseguito). Questa scelta è intenzionale e va considerata parte del
comportamento corrente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Strutture di supporto: comfort e domanda di riscaldamento
# ---------------------------------------------------------------------------


@dataclass
class ComfortSchedule:
    """
    Rappresenta le fasce orarie di comfort termico per la casa.

    per_day: dizionario {weekday -> list[(start_hour, end_hour)]}
        weekday: 0 = lunedì, ..., 6 = domenica
        start_hour/end_hour: ore in formato decimale (es. 6.5 = 06:30).
        Se il dizionario è vuoto, il comfort è considerato sempre attivo.
    """

    per_day: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)

    def is_always_on(self) -> bool:
        return not self.per_day

    def is_on(self, ts: pd.Timestamp) -> bool:
        if self.is_always_on():
            return True
        day = int(ts.weekday())
        ranges = self.per_day.get(day)
        if not ranges:
            return False
        h = ts.hour + ts.minute / 60.0
        for start, end in ranges:
            if start == end:
                continue
            # Intervallo normale (start < end)
            if start < end:
                if start <= h < end:
                    return True
            else:
                # Intervallo che attraversa la mezzanotte (es. 22:00–02:00)
                if h >= start or h < end:
                    return True
        return False


@dataclass
class HeatingDemandConfig:
    """
    Parametri globali della domanda di riscaldamento dell'abitazione
    (una sola "zona termica virtuale").
    """

    t_set_heating_c: float = 20.0
    """Temperatura interna desiderata in riscaldamento [°C]."""

    heating_enable_temp_c: float = 15.0
    """Soglia di temperatura esterna sotto cui il riscaldamento è attivo [°C]."""

    t_design_outdoor_c: float = -5.0
    """Temperatura esterna di progetto per il calcolo della dispersione [°C]."""

    floor_area_m2: float = 80.0
    """Superficie riscaldata [m²]."""

    insulation_level: Literal["scarso", "medio", "buono"] = "medio"
    """
    Livello di isolamento termico dell'involucro.
    Determina i carichi di progetto se design_heat_load_kw non è specificato.
    """

    design_heat_load_kw: Optional[float] = None
    """
    Carico termico di progetto [kW_th] alla temperatura t_design_outdoor_c.
    Se None viene stimato da superficie e isolamento, e/o dalla potenza
    termica complessiva degli impianti.
    """

    comfort_schedule: ComfortSchedule = field(default_factory=ComfortSchedule)
    """Fasce orarie di comfort (se vuoto: comfort sempre attivo)."""


# ---------------------------------------------------------------------------
# Dataclass per i generatori di riscaldamento ambiente
# ---------------------------------------------------------------------------

@dataclass
class CoolingDemandConfig:
    """Parametri globali della domanda di raffrescamento (space cooling).

    Il raffrescamento è opzionale e deve essere coerente con la presenza di un
    generatore dedicato (attualmente :class:`AirToAirACConfig`).

    Note
    ----
    La logica di abilitazione usa due condizioni:
    1) ``enabled`` (flag esplicito),
    2) temperatura esterna sopra ``cooling_enable_temp_c`` *e* comfort schedule attivo.
    """
    enabled: bool = False
    t_set_cooling_c: float = 26.0
    cooling_enable_temp_c: float = 24.0
    design_cool_load_kw: Optional[float] = None
    comfort_schedule: ComfortSchedule = field(default_factory=ComfortSchedule)

@dataclass
class HousingConfig:
    """Parametri sintetici dell'abitazione.

    Questa struttura è pensata per un'interfaccia semplificata (pochi parametri
    facilmente interpretabili). Nel modello 2R2C tali parametri vengono convertiti
    in grandezze fisiche:
    - volume interno ``V = area_m2 * ceiling_height_m``;
    - infiltrazioni (ACH) stimate dal livello di isolamento;
    - capacità termica della massa da ``thermal_mass_level``.
    """

    area_m2: float = 80.0
    ceiling_height_m: float = 2.7
    insulation_level: Literal["scarso", "medio", "buono", "ottimo"] = "medio"
    thermal_mass_level: Literal["light", "medium", "heavy"] = "medium"


@dataclass
class ThermalModelConfig:
    """Parametri numerici del modello termico (domanda ambiente).

    ``model`` è mantenuto per retro-compatibilità concettuale; nel comportamento
    corrente il ramo 2R2C è forzato in :func:`build_climate_profiles`.

    Parametri chiave
    ---------------
    leakage_ach:
        Infiltrazioni d'aria (Air Changes per Hour). Se ``auto_leakage_from_housing``
        è True, questo valore è ignorato e l'ACH viene stimato da ``housing.insulation_level``.

    thermal_mass_level:
        Mapping qualitativo verso la capacità termica della massa (light/medium/heavy).

    deadband_c:
        Isteresi del controllo ON/OFF per heating e cooling.

    internal_gains_fraction:
        Frazione della potenza elettrica interna che si trasforma in calore utile.

    gains_to_mass_fraction:
        Quota degli apporti termici assegnata al nodo massa (il resto va al nodo aria).

    mass_time_constant_h:
        Costante di tempo della massa usata per derivare ``R_am``; se None viene stimata
        da ``thermal_mass_level``.
    """
    model: Literal["2R2C", "static_UA"] = "2R2C"
    leakage_ach: float = 0.5
    ceiling_height_m: float = 2.7
    thermal_mass_level: Literal["light", "medium", "heavy"] = "medium"
    deadband_c: float = 0.5
    internal_gains_fraction: float = 0.90
    gains_to_mass_fraction: float = 0.30
    auto_leakage_from_housing: bool = True
    mass_time_constant_h: Optional[float] = None


@dataclass
class AirToAirACConfig:
    present: bool = False
    p_cool_nom_kw: float = 0.0     # kW_th
    eer_at_27c: float = 3.2
    eer_at_35c: float = 2.6
    n_units: int = 1
    priority: int = 1
    max_share_of_load: float = 1.0


@dataclass
class AirToAirHeatPumpConfig:
    """
    Pompa di calore aria-aria (split/multisplit) usata per riscaldamento ambiente.
    """

    present: bool = False
    p_heat_nom_kw: float = 0.0
    """Potenza termica nominale in modalità riscaldamento [kW_th]."""

    cop_at_7c: float = 4.0
    cop_at_minus7c: float = 2.5

    n_units: int = 1
    """Numero di unità interne (solo informativo)."""

    priority: int = 1
    """
    Priorità nell'allocazione della domanda rispetto agli altri generatori.
    1 = più prioritario.
    """

    max_share_of_load: float = 1.0
    """
    Quota massima della domanda che questo generatore può coprire (0-1).
    Utilizzata come limite superiore ulteriore rispetto a p_heat_nom_kw.
    """


@dataclass
class AirToWaterHeatPumpConfig:
    """
    Pompa di calore aria-acqua per riscaldamento ambiente.
    """

    present: bool = False
    p_heat_nom_kw: float = 0.0
    """Potenza termica nominale in modalità riscaldamento [kW_th]."""

    cop_at_7c: float = 3.2
    cop_at_minus7c: float = 2.0

    emitter_type: Literal["radiators", "fan_coils", "floor"] = "radiators"
    priority: int = 1
    max_share_of_load: float = 1.0


@dataclass
class DirectElectricHeatingConfig:
    """
    Riscaldamento diretto elettrico (stufe, termoconvettori, radiatori elettrici, pannelli).
    """

    present: bool = False
    p_el_nom_kw: float = 0.0
    """Potenza elettrica installata totale [kW_el]."""

    priority: int = 2
    max_share_of_load: float = 1.0


@dataclass
class ElectricFloorHeatingConfig:
    """
    Pavimento radiante elettrico.
    """

    present: bool = False
    p_el_nom_kw: float = 0.0
    """Potenza elettrica nominale totale del pavimento [kW_el]."""

    thermal_inertia_hours: float = 4.0
    """
    Tempo caratteristico di risposta del massetto [h].
    Usato per smussare la domanda tramite media mobile.
    """

    priority: int = 1
    max_share_of_load: float = 1.0


# ---------------------------------------------------------------------------
# Dataclass per ACS elettrica
# ---------------------------------------------------------------------------


@dataclass
class ElectricBoilerConfig:
    """
    Boiler elettrico tradizionale per ACS (resistenza in serbatoio).
    """

    present: bool = False
    p_el_kw: float = 0.0
    """Potenza elettrica della resistenza [kW_el]."""

    volume_liters: float = 80.0
    t_set_c: float = 55.0
    people: int = 0

    morning_window: Tuple[int, int] = (6, 9)
    evening_window: Tuple[int, int] = (18, 22)


@dataclass
class DHWHeatPumpConfig:
    """
    Pompa di calore dedicata all'ACS.
    """

    present: bool = False
    p_heat_nom_kw: float = 0.0
    """Potenza termica nominale in produzione ACS [kW_th]."""

    cop_at_7c: float = 2.5
    cop_at_minus7c: float = 2.0

    volume_liters: float = 200.0
    t_set_c: float = 50.0
    people: int = 0

    morning_window: Tuple[int, int] = (6, 9)
    evening_window: Tuple[int, int] = (18, 22)


# ---------------------------------------------------------------------------
# Configurazione complessiva del "clima"
# ---------------------------------------------------------------------------


@dataclass
class ClimateConfig:
    housing: Optional[HousingConfig] = None

    thermal: ThermalModelConfig = field(default_factory=ThermalModelConfig)

    heating: HeatingDemandConfig = field(default_factory=HeatingDemandConfig)
    cooling: CoolingDemandConfig = field(default_factory=CoolingDemandConfig)

    # heating generators (già esistenti)
    air_to_air_hp: AirToAirHeatPumpConfig = field(default_factory=AirToAirHeatPumpConfig)
    air_to_water_hp: AirToWaterHeatPumpConfig = field(default_factory=AirToWaterHeatPumpConfig)
    direct_heating: DirectElectricHeatingConfig = field(default_factory=DirectElectricHeatingConfig)
    floor_heating: ElectricFloorHeatingConfig = field(default_factory=ElectricFloorHeatingConfig)

    # cooling generator
    air_to_air_ac: AirToAirACConfig = field(default_factory=AirToAirACConfig)

    # dhw (già esistenti)
    dhw_electric_boiler: ElectricBoilerConfig = field(default_factory=ElectricBoilerConfig)
    dhw_hp: DHWHeatPumpConfig = field(default_factory=DHWHeatPumpConfig)



# ---------------------------------------------------------------------------
# Funzioni interne di utilità
# ---------------------------------------------------------------------------


def _align_series(index: pd.DatetimeIndex, s: Optional[pd.Series], fill: float = 0.0) -> pd.Series:
    """Allinea una serie all'indice richiesto.

    Questa utility è usata per trattare ingressi opzionali (es. ``internal_gains_kw_el``)
    senza propagare ``NaN`` o mismatch di indice.

    Regole:
    - se ``s is None``: restituisce una serie costante ``fill``;
    - se l'indice è diverso: ricampiona su ``index`` tramite interpolazione temporale.

    Nota: l'interpolazione è usata solo per serie che rappresentano grandezze
    "lente" (gains / temperature) o che hanno senso continuo nel tempo.
    """

    if s is None:
        return pd.Series(float(fill), index=index, dtype=float)
    if not s.index.equals(index):
        s = s.sort_index()
        s = s.reindex(index.union(s.index)).interpolate("time").reindex(index)
    return s.astype(float).fillna(fill)


def _linear_eer_from_outdoor(
    t_outdoor: pd.Series,
    eer_at_27c: float,
    eer_at_35c: float,
    eer_min: float = 1.2,
) -> pd.Series:
    """Stima l'EER in funzione della temperatura esterna.

    Il modello è una retta tra (27°C, ``eer_at_27c``) e (35°C, ``eer_at_35c``) con:
    - clipping tra i due valori estremi (quindi EER non cresce/decresce oltre il range);
    - soglia inferiore ``eer_min`` per robustezza numerica.

    Se i parametri sono non validi (<=0) usa un fallback costante.
    """

    t = t_outdoor.astype(float)
    if eer_at_27c <= 0 or eer_at_35c <= 0:
        return pd.Series(max(eer_min, 2.8), index=t.index, dtype=float)
    slope = (eer_at_35c - eer_at_27c) / 8.0
    eer = eer_at_27c + slope * (t - 27.0)
    lower, upper = sorted([eer_at_27c, eer_at_35c])
    return eer.clip(lower=lower, upper=upper).clip(lower=eer_min).astype(float)


def _align_outdoor_temperature(index: pd.DatetimeIndex, t_outdoor: pd.Series) -> pd.Series:
    """Allinea la temperatura esterna sull'indice richiesto.

    Il simulatore generale produce tipicamente ``meteo_hourly.csv`` e poi
    interpola a 15 minuti lato UI; questa funzione rende il core robusto
    anche in caso di input con indice diverso.

    Se l'indice non coincide:
    1) unione degli indici, 2) interpolazione temporale, 3) reindex su ``index``.

    Post-condizione: nessun ``NaN`` (``ffill/bfill`` ai bordi).
    """
    if not isinstance(t_outdoor, pd.Series):
        raise TypeError("t_outdoor deve essere una pandas.Series")

    if not t_outdoor.index.equals(index):
        # reindicizza e interpola nel tempo
        t_outdoor = t_outdoor.sort_index()
        t_outdoor = t_outdoor.reindex(index.union(t_outdoor.index)).interpolate("time")
        t_outdoor = t_outdoor.reindex(index)

    return t_outdoor.astype(float).ffill().bfill()


def _time_step_hours(index: pd.DatetimeIndex) -> float:
    """Restituisce il passo di integrazione in ore.

    Assunzione: ``index`` è (quasi) equispaziato e ordinato.
    In caso di indice corto o non monotono, viene restituito 1.0 h come fallback.

    Nota: nel simulatore CER il passo tipico è 15 minuti (0.25 h).
    """

    if len(index) < 2:
        return 1.0
    dt_sec = (index[1] - index[0]).total_seconds()
    if dt_sec <= 0:
        return 1.0
    return dt_sec / 3600.0


def _comfort_mask(index: pd.DatetimeIndex, schedule: ComfortSchedule) -> pd.Series:
    """Costruisce una maschera (0/1) che abilita il comfort sui time-step.

    La maschera viene applicata sia alla domanda di riscaldamento/raffrescamento sia
    all'abilitazione degli stati (hysteresis) nel modello 2R2C.

    - schedule vuoto/None => sempre comfort (tutti 1)
    - schedule con giorni non specificati => comfort disabilitato in quei giorni
    """
    if schedule is None or schedule.is_always_on():
        return pd.Series(1.0, index=index)
    vals = [1.0 if schedule.is_on(ts) else 0.0 for ts in index]
    return pd.Series(vals, index=index, dtype=float)


def _estimate_design_heat_load_kw(cfg: HeatingDemandConfig, total_nominal_heat_kw: float) -> float:
    """
    Stima il carico termico di progetto alla temperatura t_design_outdoor_c.

    Priorità:
    1. Se cfg.design_heat_load_kw è impostato, usa quello.
    2. Altrimenti, se total_nominal_heat_kw > 0, usa quello.
    3. Altrimenti, stima da superficie e isolamento (W/m²).
    """
    if cfg.design_heat_load_kw is not None and cfg.design_heat_load_kw > 0:
        return float(cfg.design_heat_load_kw)

    if total_nominal_heat_kw > 0:
        return float(total_nominal_heat_kw)

    # Stima semplificata in kW_th
    # scarso ~ 100 W/m², medio ~ 60 W/m², buono ~ 40 W/m²
    w_per_m2 = {
        "scarso": 0.100,  # kW per m²
        "medio": 0.060,
        "buono": 0.040,
        "ottimo": 0.030,
    }.get(cfg.insulation_level, 0.060)

    return cfg.floor_area_m2 * w_per_m2


def _ach_from_insulation(insulation_level: str) -> float:
    """Stima ACH (infiltrazioni) da livello isolamento.

    Valori conservativi per un modello base stabile.
    """
    lvl = (insulation_level or "").strip().lower()
    return {
        "scarso": 0.8,
        "medio": 0.5,
        "buono": 0.3,
        "ottimo": 0.2,
    }.get(lvl, 0.5)


def _effective_housing_inputs(cfg: "ClimateConfig") -> Tuple[float, float, str, str, float]:
    """Ritorna (area_m2, height_m, insulation_level, mass_level, ach).

    Priorità:
    - Se cfg.housing presente: usa quelli.
    - Altrimenti: usa i campi legacy.
    ACH:
    - Se cfg.thermal.auto_leakage_from_housing True: derivato da insulation_level
    - Altrimenti: usa cfg.thermal.leakage_ach
    """
    if cfg.housing is not None:
        area_m2 = float(getattr(cfg.housing, "area_m2", 80.0))
        height_m = float(getattr(cfg.housing, "ceiling_height_m", 2.7))
        insulation = str(getattr(cfg.housing, "insulation_level", "medio"))
        mass_level = str(getattr(cfg.housing, "thermal_mass_level", getattr(cfg.thermal, "thermal_mass_level", "medium")))
    else:
        area_m2 = float(getattr(cfg.heating, "floor_area_m2", 80.0))
        height_m = float(getattr(cfg.thermal, "ceiling_height_m", 2.7))
        insulation = str(getattr(cfg.heating, "insulation_level", "medio"))
        mass_level = str(getattr(cfg.thermal, "thermal_mass_level", "medium"))

    area_m2 = max(1e-6, area_m2)
    height_m = max(1.0, height_m)

    auto_ach = bool(getattr(cfg.thermal, "auto_leakage_from_housing", True))
    if auto_ach:
        ach = _ach_from_insulation(insulation)
    else:
        ach = max(0.0, float(getattr(cfg.thermal, "leakage_ach", 0.0)))

    return area_m2, height_m, insulation, mass_level, ach


def _compute_heating_demand_kw_th(
    index: pd.DatetimeIndex,
    t_outdoor: pd.Series,
    cfg: HeatingDemandConfig,
    total_nominal_heat_kw: float,
) -> pd.Series:
    """
    Calcola la domanda termica istantanea di riscaldamento ambiente [kW_th]
    per l'abitazione (una sola "zona"), prima dell'allocazione ai generatori.
    """
    t_outdoor = _align_outdoor_temperature(index, t_outdoor)

    t_set = float(cfg.t_set_heating_c)
    delta_t_design = max(t_set - float(cfg.t_design_outdoor_c), 1e-3)

    q_design_kw = _estimate_design_heat_load_kw(cfg, total_nominal_heat_kw)
    # Coefficiente globale di dispersione UA [kW_th / °C]
    ua_kw_per_deg = q_design_kw / delta_t_design

    # ΔT positiva solo quando T_out < T_set
    delta_t = (t_set - t_outdoor).clip(lower=0.0)

    # Stagione di riscaldamento: attivo solo se T_out < soglia
    season_mask = (t_outdoor < cfg.heating_enable_temp_c).astype(float)

    # Fasce di comfort
    comfort_mask = _comfort_mask(index, cfg.comfort_schedule)

    q_raw = ua_kw_per_deg * delta_t * season_mask * comfort_mask

    # Limita a q_design_kw e alla potenza nominale totale disponibile (se nota)
    if total_nominal_heat_kw > 0:
        q_cap = q_raw.clip(upper=min(q_design_kw, total_nominal_heat_kw))
    else:
        q_cap = q_raw.clip(upper=q_design_kw)

    return q_cap.astype(float)


def _linear_cop_from_outdoor(
    t_outdoor: pd.Series,
    cop_at_7c: float,
    cop_at_minus7c: float,
    cop_min: float = 1.0,
) -> pd.Series:
    """
    Calcola il COP della pompa di calore in funzione della temperatura esterna
    con interpolazione lineare tra -7°C e +7°C e saturazione ai bordi.
    """
    t = t_outdoor.astype(float)
    idx = t.index

    if cop_at_7c <= 0 or cop_at_minus7c <= 0:
        # fallback costante
        return pd.Series(max(cop_min, 2.5), index=idx, dtype=float)

    slope = (cop_at_7c - cop_at_minus7c) / 14.0  # da -7 a +7
    cop = cop_at_7c + slope * (t - 7.0)

    # limita ai valori estremi e a cop_min
    lower = min(cop_at_7c, cop_at_minus7c)
    upper = max(cop_at_7c, cop_at_minus7c)
    cop = cop.clip(lower=lower, upper=upper)
    cop = cop.clip(lower=cop_min)

    return cop.astype(float)


# ---------------------------------------------------------------------------
# Simulazione impianti di riscaldamento ambiente (space heating)
# ---------------------------------------------------------------------------


def _simulate_air_to_air_hp(
    q_req_rem_kw_th: pd.Series,
    t_outdoor: pd.Series,
    cfg: AirToAirHeatPumpConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simula il contributo della/e PDC aria-aria al riscaldamento ambiente.

    Restituisce:
        (q_hp_th, p_el_hp) come serie [kW_th], [kW_el].
    """
    index = q_req_rem_kw_th.index
    q_zero = pd.Series(0.0, index=index)

    if not cfg.present or cfg.p_heat_nom_kw <= 0:
        return q_zero, q_zero

    max_th_kw = max(0.0, cfg.p_heat_nom_kw * max(0.0, min(cfg.max_share_of_load, 1.0)))
    if max_th_kw <= 0:
        return q_zero, q_zero

    q_req_rem_kw_th = q_req_rem_kw_th.clip(lower=0.0)
    q_hp_th = q_req_rem_kw_th.clip(upper=max_th_kw)

    cop = _linear_cop_from_outdoor(t_outdoor, cfg.cop_at_7c, cfg.cop_at_minus7c)
    p_el_hp = pd.Series(0.0, index=index, dtype=float)
    mask = q_hp_th > 0
    p_el_hp[mask] = (q_hp_th[mask] / cop[mask]).astype(float)

    return q_hp_th.astype(float), p_el_hp.astype(float)


def _simulate_air_to_water_hp(
    q_req_rem_kw_th: pd.Series,
    t_outdoor: pd.Series,
    cfg: AirToWaterHeatPumpConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simula il contributo della PDC aria-acqua al riscaldamento ambiente.
    """
    index = q_req_rem_kw_th.index
    q_zero = pd.Series(0.0, index=index)

    if not cfg.present or cfg.p_heat_nom_kw <= 0:
        return q_zero, q_zero

    max_th_kw = max(0.0, cfg.p_heat_nom_kw * max(0.0, min(cfg.max_share_of_load, 1.0)))
    if max_th_kw <= 0:
        return q_zero, q_zero

    q_req_rem_kw_th = q_req_rem_kw_th.clip(lower=0.0)
    q_hp_th = q_req_rem_kw_th.clip(upper=max_th_kw)

    cop = _linear_cop_from_outdoor(t_outdoor, cfg.cop_at_7c, cfg.cop_at_minus7c)
    p_el_hp = pd.Series(0.0, index=index, dtype=float)
    mask = q_hp_th > 0
    p_el_hp[mask] = (q_hp_th[mask] / cop[mask]).astype(float)

    return q_hp_th.astype(float), p_el_hp.astype(float)


def _simulate_direct_heating(
    q_req_rem_kw_th: pd.Series,
    t_outdoor: pd.Series,
    cfg: DirectElectricHeatingConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simula il contributo del riscaldamento diretto elettrico.
    (COP ~ 1, potenza termica = potenza elettrica.)
    """
    index = q_req_rem_kw_th.index
    q_zero = pd.Series(0.0, index=index)

    if not cfg.present or cfg.p_el_nom_kw <= 0:
        return q_zero, q_zero

    max_th_kw = max(0.0, cfg.p_el_nom_kw * max(0.0, min(cfg.max_share_of_load, 1.0)))
    if max_th_kw <= 0:
        return q_zero, q_zero

    q_req_rem_kw_th = q_req_rem_kw_th.clip(lower=0.0)
    q_dev_th = q_req_rem_kw_th.clip(upper=max_th_kw)

    # COP = 1
    p_el = q_dev_th.copy()

    return q_dev_th.astype(float), p_el.astype(float)


def _simulate_floor_heating(
    q_req_rem_kw_th: pd.Series,
    t_outdoor: pd.Series,
    cfg: ElectricFloorHeatingConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """Riscaldamento elettrico a pavimento: COP = 1 con inerzia termica.

    Nota: per evitare sovra-erogazioni (overshoot) dovute allo smoothing,
    la potenza termica erogata e sempre limitata anche dal carico richiesto
    istantaneo q_req_rem_kw_th.
    """
    idx = q_req_rem_kw_th.index
    q_req_rem_kw_th = _align_series(idx, q_req_rem_kw_th, 0.0).clip(lower=0.0)

    if not cfg.present or cfg.p_el_nom_kw <= 0:
        z = pd.Series(0.0, index=idx, dtype=float)
        return z, z

    max_th_kw = float(cfg.p_el_nom_kw) * max(0.0, min(float(cfg.max_share_of_load), 1.0))
    if max_th_kw <= 0:
        z = pd.Series(0.0, index=idx, dtype=float)
        return z, z

    # Smoothing: rolling mean su finestra equivalente a thermal_inertia_hours
    dt_hours = float(dt_hours) if dt_hours and dt_hours > 0 else _time_step_hours(idx)
    window_steps = max(1, int(round(float(cfg.thermal_inertia_hours) / dt_hours))) if dt_hours > 0 else 1

    q_smoothed = q_req_rem_kw_th.rolling(window=window_steps, min_periods=1).mean()

    # Limiti: non superare la domanda istantanea e la potenza disponibile
    q_floor_th = pd.concat([q_smoothed, q_req_rem_kw_th], axis=1).min(axis=1)
    q_floor_th = q_floor_th.clip(lower=0.0, upper=max_th_kw)

    p_el_floor = q_floor_th  # COP = 1

    return q_floor_th.astype(float), p_el_floor.astype(float)

# ---------------------------------------------------------------------------
# Domanda ambiente: modello 2R2C
# ---------------------------------------------------------------------------

def _simulate_air_to_air_ac(q_cool_kw_th: pd.Series, t_outdoor: pd.Series, cfg: AirToAirACConfig) -> Tuple[pd.Series, pd.Series]:
    idx = q_cool_kw_th.index
    z = pd.Series(0.0, index=idx, dtype=float)
    if not cfg.present or cfg.p_cool_nom_kw <= 0:
        return z, z

    cap_th = float(cfg.p_cool_nom_kw) * max(1, int(cfg.n_units)) * max(0.0, min(float(cfg.max_share_of_load), 1.0))
    q_th = q_cool_kw_th.clip(lower=0.0, upper=cap_th)

    eer = _linear_eer_from_outdoor(t_outdoor, cfg.eer_at_27c, cfg.eer_at_35c)
    p_el = pd.Series(0.0, index=idx, dtype=float)
    m = q_th > 0
    p_el[m] = (q_th[m] / eer[m]).astype(float)
    return q_th.astype(float), p_el.astype(float)



# ---------------------------------------------------------------------------
# Simulazione ACS (boiler elettrico / PDC ACS)
# ---------------------------------------------------------------------------


_DEF_E_TH_PER_CAPITA_KWH = 2.5  # kWh_th per persona al giorno (valore medio indicativo)


def _day_iterator(index: pd.DatetimeIndex):
    """
    Iteratore sui giorni presenti nell'indice.
    Restituisce (day_date, mask_index_for_day).
    """
    if index.empty:
        return
    days = index.normalize().unique()
    for d in days:
        mask = index.normalize() == d
        yield d, mask


def _simulate_electric_boiler(
    index: pd.DatetimeIndex,
    cfg: ElectricBoilerConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simula un boiler elettrico tradizionale.

    Restituisce:
        (q_th, p_el) = (domanda termica ACS, potenza elettrica) [kW_th], [kW_el].
    """
    p_el = pd.Series(0.0, index=index)
    q_th = pd.Series(0.0, index=index)

    if not cfg.present or cfg.p_el_kw <= 0 or cfg.people <= 0:
        return q_th, p_el

    daily_e_th = cfg.people * _DEF_E_TH_PER_CAPITA_KWH  # kWh_th/giorno

    for _, mask_day in _day_iterator(index):
        if not mask_day.any():
            continue

        # energia da fornire in questo giorno
        e_th_remaining = daily_e_th

        # finestre temporali (mattina, sera)
        windows = [cfg.morning_window, cfg.evening_window]

        for (start_h, end_h) in windows:
            if e_th_remaining <= 0:
                break

            # seleziona i timesteps della finestra
            ts_day = index[mask_day]
            hours = ts_day.hour + ts_day.minute / 60.0
            mask_window_local = (hours >= start_h) & (hours < end_h)

            if not mask_window_local.any():
                continue

            idx_window = ts_day[mask_window_local]
            n_steps = len(idx_window)
            if n_steps == 0:
                continue

            max_e_window = cfg.p_el_kw * dt_hours * n_steps  # kWh possibili
            e_window = min(e_th_remaining, max_e_window)

            if e_window <= 0:
                continue

            n_on = int(round(e_window / (cfg.p_el_kw * dt_hours)))
            n_on = max(0, min(n_on, n_steps))

            if n_on <= 0:
                continue

            idx_on = idx_window[:n_on]
            p_el[idx_on] = cfg.p_el_kw
            q_th[idx_on] = cfg.p_el_kw  # COP = 1

            e_th_remaining -= cfg.p_el_kw * dt_hours * n_on

    return q_th.astype(float), p_el.astype(float)


def _simulate_dhw_hp(
    index: pd.DatetimeIndex,
    t_outdoor: pd.Series,
    cfg: DHWHeatPumpConfig,
    dt_hours: float,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simula una PDC dedicata all'ACS.

    Restituisce:
        (q_th, p_el) = (domanda termica ACS, potenza elettrica) [kW_th], [kW_el].
    """
    p_el = pd.Series(0.0, index=index)
    q_th = pd.Series(0.0, index=index)

    if not cfg.present or cfg.p_heat_nom_kw <= 0 or cfg.people <= 0:
        return q_th, p_el

    t_outdoor = t_outdoor.reindex(index).astype(float)
    cop = _linear_cop_from_outdoor(t_outdoor, cfg.cop_at_7c, cfg.cop_at_minus7c)

    daily_e_th = cfg.people * _DEF_E_TH_PER_CAPITA_KWH  # kWh_th/giorno

    for _, mask_day in _day_iterator(index):
        if not mask_day.any():
            continue

        e_th_remaining = daily_e_th
        windows = [cfg.morning_window, cfg.evening_window]

        for (start_h, end_h) in windows:
            if e_th_remaining <= 0:
                break

            ts_day = index[mask_day]
            hours = ts_day.hour + ts_day.minute / 60.0
            mask_window_local = (hours >= start_h) & (hours < end_h)

            if not mask_window_local.any():
                continue

            idx_window = ts_day[mask_window_local]
            n_steps = len(idx_window)
            if n_steps == 0:
                continue

            max_e_window = cfg.p_heat_nom_kw * dt_hours * n_steps  # kWh_th possibili
            e_window = min(e_th_remaining, max_e_window)

            if e_window <= 0:
                continue

            n_on = int(round(e_window / (cfg.p_heat_nom_kw * dt_hours)))
            n_on = max(0, min(n_on, n_steps))

            if n_on <= 0:
                continue

            idx_on = idx_window[:n_on]
            q_th[idx_on] = cfg.p_heat_nom_kw

            e_th_remaining -= cfg.p_heat_nom_kw * dt_hours * n_on

    # potenza elettrica dalla serie termica e COP
    mask = q_th > 0
    p_el[mask] = (q_th[mask] / cop[mask]).astype(float)

    return q_th.astype(float), p_el.astype(float)


# ---------------------------------------------------------------------------
# Funzione principale: build_climate_profiles
# ---------------------------------------------------------------------------


def build_climate_profiles(
    index,
    cfg: ClimateConfig,
    t_outdoor: pd.Series,
    internal_gains_kw_el: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """Costruisce i profili di potenza elettrica (kW_el) del sistema clima.

    La funzione rappresenta l'entry-point del sottosistema *clima* e produce le curve
    che verranno poi sommate dall'orchestratore UI (pagina *Consumatori*).

    Parametri
    ---------
    index:
        Indice temporale della simulazione. In tutta la pipeline CER è tipicamente un
        ``pd.DatetimeIndex`` a **15 minuti**, timezone-aware in **UTC**.
        L'implementazione richiede soprattutto che:
        - l'indice sia ordinato;
        - il passo sia quasi costante (il passo viene stimato dal primo intervallo).

    cfg:
        Configurazione completa :class:`ClimateConfig`, che include:
        - parametri abitazione (area, altezza, isolamento, massa),
        - setpoint e abilitazioni stagionali per heating/cooling,
        - configurazione e priorità dei generatori,
        - configurazione ACS.

    t_outdoor:
        Temperatura esterna in °C come ``pd.Series``. Può essere oraria o a 15 minuti:
        viene allineata a ``index`` con interpolazione temporale.

    internal_gains_kw_el:
        (Opzionale) profilo di carichi elettrici interni (kW_el) riutilizzato come
        **apporto termico interno**. Il modello assume che una frazione
        ``cfg.thermal.internal_gains_fraction`` della potenza elettrica diventi calore
        in ambiente. Gli apporti vengono divisi tra nodo aria e nodo massa secondo
        ``cfg.thermal.gains_to_mass_fraction``.

    Modello di domanda ambiente
    ---------------------------
    Nel codice esistono due approcci:

    - ``static_UA`` (legacy): domanda termica proporzionale a ``UA * (T_set - T_out)``,
      solo per riscaldamento (e ACS), senza simulazione di temperatura interna.

    - ``2R2C`` (attuale): modello dinamico a due nodi (aria + massa) e due resistenze:

      *Rete termica* (unità in K/kW e kWh/K):

      - resistenza verso esterno: ``R_oa = 1 / UA_total``
      - resistenza aria↔massa: ``R_am`` derivata dalla costante di tempo della massa
      - capacità aria: ``C_air`` da volume d'aria (rho·cp·V)
      - capacità massa: ``C_mass`` da un mapping (light/medium/heavy)

      *Equazioni (forma discreta, Euler esplicito, con sub-stepping interno)*:

      - ``T_air`` evolve per dispersione verso esterno, scambio con la massa e gains:
        ``dT_air = (Q_env + Q_mass + G_air + Q_hvac) * dt / C_air``
      - ``T_mass`` evolve per scambio con l'aria e gains assegnati alla massa:
        ``dT_mass = (Q_air_to_mass + G_mass) * dt / C_mass``

    **Nota importante**: per comportamento corrente/stabilità, in questa versione
    l'implementazione forza ``model = "2R2C"`` (quindi il ramo ``static_UA`` non viene
    eseguito anche se configurato).

    Controllo e allocazione carichi
    -------------------------------
    - Abilitazione heating: ``T_out < heating_enable_temp_c`` e comfort schedule attivo.
    - Abilitazione cooling: ``T_out > cooling_enable_temp_c`` e comfort schedule attivo,
      *e* presenza di un dispositivo di raffrescamento.
    - Logica ON/OFF con isteresi su ``deadband_c`` (mezzo deadband per soglia).
    - Allocazione greedy della potenza termica richiesta ai generatori in ordine di
      priorità (campo ``priority``), rispettando capacità nominali e ``max_share_of_load``.
      La quota non coperta viene esposta come ``*_unserved_th`` (kW_th).

    Output
    ------
    Restituisce un dizionario ``Dict[str, pd.Series]`` con tutte le serie indicizzate
    su ``index``.

    Chiavi principali (kW_el):
    - ``space_heating_total``: somma dei consumi elettrici dei generatori di heating.
    - ``space_cooling_total``: consumo elettrico per raffrescamento (AC aria-aria).
    - ``dhw_total``: consumo elettrico ACS (boiler o PDC ACS).
    - ``aggregated``: somma dei precedenti (curva *clima* finale).

    Chiavi per-device (kW_el) mantenute per retro-compatibilità:
    - ``space_heating_air_to_air``
    - ``space_heating_air_to_water``
    - ``space_heating_direct``
    - ``space_heating_floor``
    - ``space_cooling_air_to_air_ac``
    - ``dhw_electric_boiler``
    - ``dhw_hp``

    Chiavi ausiliarie (kW_th / °C):
    - ``space_heating_unserved_th`` / ``space_cooling_unserved_th``
    - ``t_air_c`` / ``t_mass_c`` (solo significative nel ramo 2R2C)

    Diagnostica:
    La funzione produce anche serie ``dbg_*`` (0/1 o kW_th) utili in UI per capire
    perché il carico è nullo (comfort disattivo, condizioni stagionali, ecc.).
    """

    index = pd.DatetimeIndex(index)
    t_outdoor_aligned = _align_outdoor_temperature(index, t_outdoor)
    dt_hours = _time_step_hours(index)

    # ---------------------------------------------------------------------
    # Helper: inizializza serie zero
    # ---------------------------------------------------------------------
    z = pd.Series(0.0, index=index, dtype=float)

    # Potenza termica nominale complessiva disponibile per il riscaldamento
    total_nominal_heat_kw = 0.0
    if cfg.air_to_air_hp.present and cfg.air_to_air_hp.p_heat_nom_kw > 0:
        total_nominal_heat_kw += float(cfg.air_to_air_hp.p_heat_nom_kw) * max(0.0, min(float(cfg.air_to_air_hp.max_share_of_load), 1.0))
    if cfg.air_to_water_hp.present and cfg.air_to_water_hp.p_heat_nom_kw > 0:
        total_nominal_heat_kw += float(cfg.air_to_water_hp.p_heat_nom_kw) * max(0.0, min(float(cfg.air_to_water_hp.max_share_of_load), 1.0))
    if cfg.direct_heating.present and cfg.direct_heating.p_el_nom_kw > 0:
        total_nominal_heat_kw += float(cfg.direct_heating.p_el_nom_kw) * max(0.0, min(float(cfg.direct_heating.max_share_of_load), 1.0))
    if cfg.floor_heating.present and cfg.floor_heating.p_el_nom_kw > 0:
        total_nominal_heat_kw += float(cfg.floor_heating.p_el_nom_kw) * max(0.0, min(float(cfg.floor_heating.max_share_of_load), 1.0))

    # Warning diagnostico: nessun generatore di riscaldamento con potenza nominale > 0
    if total_nominal_heat_kw <= 0:
        try:
            import logging
            logging.getLogger(__name__).warning(
                "Nessun generatore di riscaldamento presente o potenza nominale totale = 0.0 kW -> curve elettriche per riscaldamento saranno 0."
            )
        except Exception:
            # In ambienti senza logging configurato, ignoriamo l'errore.
            pass

    # ---------------------------------------------------------------------
    # 1) DOMANDA + CONSUMI ELETTRICI PER SPACE HEATING / COOLING
    # ---------------------------------------------------------------------
    model = "2R2C"  # forced to 2R2C for stability and consistency

    # Serie di output per singolo generatore (sempre presenti per retro-compat)
    p_space_air_to_air = z.copy()
    p_space_air_to_water = z.copy()
    p_space_direct = z.copy()
    p_space_floor = z.copy()

    p_space_cooling_ac = z.copy()

    space_heating_unserved_th = z.copy()
    space_cooling_unserved_th = z.copy()

    t_air_c = pd.Series(np.nan, index=index, dtype=float)
    t_mass_c = pd.Series(np.nan, index=index, dtype=float)

    # Debug/diagnostics (returned as additional series; safe for callers that ignore them)
    dbg_heat_allowed = pd.Series(False, index=index, dtype=bool)
    dbg_cool_allowed = pd.Series(False, index=index, dtype=bool)
    dbg_heating_on = pd.Series(False, index=index, dtype=bool)
    dbg_cooling_on = pd.Series(False, index=index, dtype=bool)
    dbg_q_heat_req_th = z.copy()
    dbg_q_cool_req_th = z.copy()
    dbg_stability_coef = z.copy()  # dt / (C_air * R_am) (dimensionless)

    if str(model).lower() == "static_ua":
        # ---------------------------
        # Legacy: UA statico (come prima)
        # ---------------------------
        q_req_kw_th = _compute_heating_demand_kw_th(
            index=index,
            t_outdoor=t_outdoor_aligned,
            cfg=cfg.heating,
            total_nominal_heat_kw=total_nominal_heat_kw,
        )

        devices = []
        if cfg.air_to_water_hp.present and cfg.air_to_water_hp.p_heat_nom_kw > 0:
            devices.append(("space_heating_air_to_water", cfg.air_to_water_hp, _simulate_air_to_water_hp))
        if cfg.air_to_air_hp.present and cfg.air_to_air_hp.p_heat_nom_kw > 0:
            devices.append(("space_heating_air_to_air", cfg.air_to_air_hp, _simulate_air_to_air_hp))
        if cfg.floor_heating.present and cfg.floor_heating.p_el_nom_kw > 0:
            devices.append(("space_heating_floor", cfg.floor_heating, _simulate_floor_heating))
        if cfg.direct_heating.present and cfg.direct_heating.p_el_nom_kw > 0:
            devices.append(("space_heating_direct", cfg.direct_heating, _simulate_direct_heating))
        devices.sort(key=lambda x: getattr(x[1], "priority", 99))

        q_rem = q_req_kw_th.copy()
        for label, dev_cfg, sim_fn in devices:
            q_dev_th, p_el = sim_fn(q_rem, t_outdoor_aligned, dev_cfg, dt_hours)
            q_rem = (q_rem - q_dev_th).clip(lower=0.0)

            if label == "space_heating_air_to_air":
                p_space_air_to_air = p_el.astype(float)
            elif label == "space_heating_air_to_water":
                p_space_air_to_water = p_el.astype(float)
            elif label == "space_heating_floor":
                p_space_floor = p_el.astype(float)
            elif label == "space_heating_direct":
                p_space_direct = p_el.astype(float)

        space_heating_unserved_th = q_rem.astype(float)

        # temperatures are not simulated in static_UA
        # cooling is always zero in static_UA

    else:
        # ---------------------------
        # 2R2C dinamico (senza solare)
        # ---------------------------
        # Internal gains (kW_el -> kW_th)
        gains_el = _align_series(index, internal_gains_kw_el, 0.0).clip(lower=0.0)
        g_frac = float(getattr(cfg.thermal, "internal_gains_fraction", 0.90))
        g_to_mass = float(getattr(cfg.thermal, "gains_to_mass_fraction", 0.30))
        g_to_mass = max(0.0, min(g_to_mass, 1.0))
        gains_th = gains_el * max(0.0, min(g_frac, 1.0))
        gains_mass = gains_th * g_to_mass
        gains_air = gains_th - gains_mass

        # UA totale (envelope + infiltration) ricavata dalla stima legacy
        t_set_h = float(cfg.heating.t_set_heating_c)
        delta_t_design = max(t_set_h - float(cfg.heating.t_design_outdoor_c), 1e-3)
        q_design_kw = max(_estimate_design_heat_load_kw(cfg.heating, total_nominal_heat_kw), 1e-3)
        ua_env_kw_per_k = q_design_kw / delta_t_design  # kW/K

        # Infiltrazione: UA_inf = rho*cp*V*ACH/3600
        area_m2, h_m, _insul_lvl_eff, _mass_lvl_eff, ach = _effective_housing_inputs(cfg)
        v_m3 = area_m2 * h_m
        rho = 1.2  # kg/m3
        cp = 1005.0  # J/kgK
        ua_inf_kw_per_k = (rho * cp * v_m3 * ach / 3600.0) / 1000.0

        ua_total_kw_per_k = max(ua_env_kw_per_k + ua_inf_kw_per_k, 1e-6)

        # Parametri 2R2C
        r_oa_k_per_kw = 1.0 / ua_total_kw_per_k

        # Capacita' dell'aria: rho*cp*V [J/K] -> [kWh/K]
        c_air_kwh_per_k = max((rho * cp * v_m3) / 3_600_000.0, 1e-4)

        # Capacita' della massa: mapping per livello
        mass_level = str(_mass_lvl_eff).lower()
        c_mass_per_m2 = {
            "light": 0.05,
            "medium": 0.10,
            "heavy": 0.20,
        }.get(mass_level, 0.10)  # kWh/K per m2
        c_mass_kwh_per_k = max(c_mass_per_m2 * area_m2, 5.0 * c_air_kwh_per_k)

        # R_am da costante di tempo della massa
        tau_mass_h = getattr(cfg.thermal, "mass_time_constant_h", None)
        if tau_mass_h is None or float(tau_mass_h) <= 0:
            tau_mass_h = {
                "light": 6.0,
                "medium": 12.0,
                "heavy": 24.0,
            }.get(mass_level, 12.0)
        tau_mass_h = float(tau_mass_h)

        # NOTE (numerical stability): the 2R2C model is integrated with an explicit
        # Euler scheme using the simulation time step (typically 15 minutes).
        # The air-to-mass coupling term can become numerically unstable if the
        # coupling resistance is too small compared to the air capacitance.
        # A sufficient stability condition for the coupling term is:
        #   dt / (C_air * R_am) < 2  ->  R_am > dt / (2*C_air)
        # We therefore enforce a minimum R_am based on dt and C_air.
        r_am_from_tau = tau_mass_h / c_mass_kwh_per_k
        # use a small safety margin (denominator < 2) to keep the coefficient < 2
        r_am_min_stability = dt_hours / max(1.8 * c_air_kwh_per_k, 1e-6)
        r_am_k_per_kw = max(r_am_from_tau, r_am_min_stability, 0.05)
        dbg_stability_coef[:] = float(dt_hours) / max(float(c_air_kwh_per_k) * float(r_am_k_per_kw), 1e-9)

        # Comfort masks
        heat_comfort = _comfort_mask(index, cfg.heating.comfort_schedule)
        cool_comfort = _comfort_mask(index, cfg.cooling.comfort_schedule)

        heat_allowed = (t_outdoor_aligned < float(cfg.heating.heating_enable_temp_c)) & (heat_comfort > 0.5)
        dbg_heat_allowed = heat_allowed.astype(bool)

        cool_device_available = bool(cfg.cooling.enabled) and bool(cfg.air_to_air_ac.present) and float(cfg.air_to_air_ac.p_cool_nom_kw) > 0
        if cool_device_available:
            cool_allowed = (t_outdoor_aligned > float(cfg.cooling.cooling_enable_temp_c)) & (cool_comfort > 0.5)
        else:
            cool_allowed = pd.Series(False, index=index, dtype=bool)

        dbg_cool_allowed = cool_allowed.astype(bool)

        # Deadband
        db = max(0.0, float(getattr(cfg.thermal, "deadband_c", 0.5)))
        dbh = 0.5 * db

        t_set_cool = float(cfg.cooling.t_set_cooling_c)

        # Precompute COP/EER
        cop_aahp = _linear_cop_from_outdoor(t_outdoor_aligned, cfg.air_to_air_hp.cop_at_7c, cfg.air_to_air_hp.cop_at_minus7c) if cfg.air_to_air_hp.present else None
        cop_atwhp = _linear_cop_from_outdoor(t_outdoor_aligned, cfg.air_to_water_hp.cop_at_7c, cfg.air_to_water_hp.cop_at_minus7c) if cfg.air_to_water_hp.present else None
        eer_ac = _linear_eer_from_outdoor(t_outdoor_aligned, cfg.air_to_air_ac.eer_at_27c, cfg.air_to_air_ac.eer_at_35c) if cool_device_available else None

        # Capacita' termiche max
        cap_air_to_air_th = float(cfg.air_to_air_hp.p_heat_nom_kw) * max(0.0, min(float(cfg.air_to_air_hp.max_share_of_load), 1.0)) if (cfg.air_to_air_hp.present and cfg.air_to_air_hp.p_heat_nom_kw > 0) else 0.0
        cap_air_to_water_th = float(cfg.air_to_water_hp.p_heat_nom_kw) * max(0.0, min(float(cfg.air_to_water_hp.max_share_of_load), 1.0)) if (cfg.air_to_water_hp.present and cfg.air_to_water_hp.p_heat_nom_kw > 0) else 0.0
        cap_direct_th = float(cfg.direct_heating.p_el_nom_kw) * max(0.0, min(float(cfg.direct_heating.max_share_of_load), 1.0)) if (cfg.direct_heating.present and cfg.direct_heating.p_el_nom_kw > 0) else 0.0
        cap_floor_th = float(cfg.floor_heating.p_el_nom_kw) * max(0.0, min(float(cfg.floor_heating.max_share_of_load), 1.0)) if (cfg.floor_heating.present and cfg.floor_heating.p_el_nom_kw > 0) else 0.0

        cap_cool_th = 0.0
        if cool_device_available:
            cap_cool_th = float(cfg.air_to_air_ac.p_cool_nom_kw) * max(1, int(cfg.air_to_air_ac.n_units)) * max(0.0, min(float(cfg.air_to_air_ac.max_share_of_load), 1.0))

        # Setup floor smoothing buffer
        window_steps_floor = 1
        if cfg.floor_heating.present and float(cfg.floor_heating.thermal_inertia_hours) > 0 and dt_hours > 0:
            window_steps_floor = max(1, int(round(float(cfg.floor_heating.thermal_inertia_hours) / dt_hours)))
        floor_hist: list[float] = []

        # Inizializzazione temperature
        if bool(heat_allowed.iloc[0]):
            t0 = t_set_h
        elif bool(cool_allowed.iloc[0]):
            t0 = t_set_cool
        else:
            t0 = 22.0
        t_air = float(t0)
        t_mass = float(t0)

        heating_on = False
        cooling_on = False

        for i, ts in enumerate(index):
            tout = float(t_outdoor_aligned.iat[i])
            g_air = float(gains_air.iat[i])
            g_mass = float(gains_mass.iat[i])

            # Free evolution (no HVAC)
            # Use internal sub-stepping to improve numerical stability of the explicit Euler scheme.
            # This greatly reduces oscillations of the air node that can otherwise be clipped to [-10, 35] °C.
            sub_dt_target_h = 5.0 / 60.0  # 5 minutes
            n_sub = int(max(1, round(dt_hours / sub_dt_target_h)))
            dt_sub = dt_hours / n_sub
            # Free evolution (no HVAC) with internal sub-stepping
            t_air_tmp = float(t_air)
            t_mass_tmp = float(t_mass)
            for _ in range(n_sub):
                q_env = (tout - t_air_tmp) / r_oa_k_per_kw
                q_mass = (t_mass_tmp - t_air_tmp) / r_am_k_per_kw
                t_air_tmp = t_air_tmp + (dt_sub / c_air_kwh_per_k) * (q_env + q_mass + g_air)

                q_to_mass = (t_air_tmp - t_mass_tmp) / r_am_k_per_kw
                t_mass_tmp = t_mass_tmp + (dt_sub / c_mass_kwh_per_k) * (q_to_mass + g_mass)

            t_air_free = float(t_air_tmp)
            t_mass_free = float(t_mass_tmp)

            # Safety clamp (should rarely trigger with sub-stepping)
            if t_air_free < -30.0:
                t_air_free = -30.0
            elif t_air_free > 60.0:
                t_air_free = 60.0
            # Update hysteresis states
            if not bool(heat_allowed.iat[i]):
                heating_on = False
            else:
                if heating_on:
                    if t_air_free >= (t_set_h + dbh):
                        heating_on = False
                else:
                    if t_air_free <= (t_set_h - dbh):
                        heating_on = True

            if not bool(cool_allowed.iat[i]):
                cooling_on = False
            else:
                if cooling_on:
                    if t_air_free <= (t_set_cool - dbh):
                        cooling_on = False
                else:
                    if t_air_free >= (t_set_cool + dbh):
                        cooling_on = True

            # Mutua esclusione
            if heating_on and cooling_on:
                # In caso patologico, prevale la modalita' con errore maggiore rispetto al setpoint
                if (t_air_free - t_set_cool) > (t_set_h - t_air_free):
                    heating_on = False
                else:
                    cooling_on = False

            q_heat_req = 0.0
            q_cool_req = 0.0

            if heating_on:
                t_target = t_set_h + dbh
                q_hvac_req = (t_target - t_air_free) * c_air_kwh_per_k / max(dt_hours, 1e-6)
                q_heat_req = max(0.0, float(q_hvac_req))
            elif cooling_on:
                t_target = t_set_cool - dbh
                q_hvac_req = (t_target - t_air_free) * c_air_kwh_per_k / max(dt_hours, 1e-6)
                q_cool_req = max(0.0, float(-q_hvac_req))

            # Diagnostics
            dbg_heating_on.iat[i] = bool(heating_on)
            dbg_cooling_on.iat[i] = bool(cooling_on)
            dbg_q_heat_req_th.iat[i] = float(q_heat_req)
            dbg_q_cool_req_th.iat[i] = float(q_cool_req)

            # --- allocate heating thermal load across devices
            q_rem = q_heat_req
            q_del_total = 0.0

            # Device list in priority order
            devs = []
            if cfg.air_to_water_hp.present and cap_air_to_water_th > 0:
                devs.append((getattr(cfg.air_to_water_hp, "priority", 99), "air_to_water"))
            if cfg.air_to_air_hp.present and cap_air_to_air_th > 0:
                devs.append((getattr(cfg.air_to_air_hp, "priority", 99), "air_to_air"))
            if cfg.floor_heating.present and cap_floor_th > 0:
                devs.append((getattr(cfg.floor_heating, "priority", 99), "floor"))
            if cfg.direct_heating.present and cap_direct_th > 0:
                devs.append((getattr(cfg.direct_heating, "priority", 99), "direct"))
            devs.sort(key=lambda x: x[0])

            for _, kind in devs:
                if q_rem <= 0:
                    break

                if kind == "air_to_water":
                    q_dev = min(q_rem, cap_air_to_water_th)
                    cop = float(cop_atwhp.iat[i]) if cop_atwhp is not None else 2.5
                    p_space_air_to_water.iat[i] = q_dev / max(cop, 1e-3)
                    q_rem -= q_dev
                    q_del_total += q_dev

                elif kind == "air_to_air":
                    q_dev = min(q_rem, cap_air_to_air_th)
                    cop = float(cop_aahp.iat[i]) if cop_aahp is not None else 2.5
                    p_space_air_to_air.iat[i] = q_dev / max(cop, 1e-3)
                    q_rem -= q_dev
                    q_del_total += q_dev

                elif kind == "floor":
                    # apply inertia on remaining demand
                    floor_hist.append(float(q_rem))
                    if len(floor_hist) > window_steps_floor:
                        floor_hist = floor_hist[-window_steps_floor:]
                    q_cmd = float(sum(floor_hist) / len(floor_hist)) if floor_hist else float(q_rem)
                    q_dev = min(q_rem, q_cmd, cap_floor_th)
                    p_space_floor.iat[i] = q_dev  # COP=1
                    q_rem -= q_dev
                    q_del_total += q_dev

                elif kind == "direct":
                    q_dev = min(q_rem, cap_direct_th)
                    p_space_direct.iat[i] = q_dev  # COP=1
                    q_rem -= q_dev
                    q_del_total += q_dev

            space_heating_unserved_th.iat[i] = max(0.0, q_rem)

            # --- cooling via air-to-air AC
            q_cool_del = 0.0
            if q_cool_req > 0 and cap_cool_th > 0 and eer_ac is not None:
                q_cool_del = min(q_cool_req, cap_cool_th)
                eer = float(eer_ac.iat[i])
                p_space_cooling_ac.iat[i] = q_cool_del / max(eer, 1e-3)

            space_cooling_unserved_th.iat[i] = max(0.0, q_cool_req - q_cool_del)

            # Actual HVAC term on air node (kW_th)
            q_hvac_act = q_del_total - q_cool_del

            # Apply HVAC to air temperature
            t_air_next = t_air_free + (dt_hours / c_air_kwh_per_k) * q_hvac_act
            # Clamp updated indoor temperature to physical range
            if t_air_next < -10.0:
                t_air_next = -10.0
            elif t_air_next > 35.0:
                t_air_next = 35.0

            # Mass update (uses updated air temperature)
            q_air_to_mass = (t_air_next - t_mass) / r_am_k_per_kw
            t_mass_next = t_mass + (dt_hours / c_mass_kwh_per_k) * (q_air_to_mass + g_mass)
            # Clamp mass temperature to a reasonable range
            if t_mass_next < -20.0:
                t_mass_next = -20.0
            elif t_mass_next > 40.0:
                t_mass_next = 40.0

            t_air = float(t_air_next)
            t_mass = float(t_mass_next)

            t_air_c.iat[i] = t_air
            t_mass_c.iat[i] = t_mass

    # Totali space heating / cooling (kW_el)
    space_heating_total = (p_space_air_to_air + p_space_air_to_water + p_space_direct + p_space_floor).astype(float)
    space_cooling_total = p_space_cooling_ac.astype(float)

    # ---------------------------------------------------------------------
    # 2) ACS (boiler elettrico / PDC ACS) - invariata
    # ---------------------------------------------------------------------
    p_dhw_boiler = z.copy()
    p_dhw_hp = z.copy()

    if cfg.dhw_hp.present and cfg.dhw_hp.p_heat_nom_kw > 0 and cfg.dhw_hp.people > 0:
        _, p_dhw_hp = _simulate_dhw_hp(
            index=index,
            t_outdoor=t_outdoor_aligned,
            cfg=cfg.dhw_hp,
            dt_hours=dt_hours,
        )
    elif cfg.dhw_electric_boiler.present and cfg.dhw_electric_boiler.p_el_kw > 0 and cfg.dhw_electric_boiler.people > 0:
        _, p_dhw_boiler = _simulate_electric_boiler(
            index=index,
            cfg=cfg.dhw_electric_boiler,
            dt_hours=dt_hours,
        )

    dhw_total = (p_dhw_boiler + p_dhw_hp).astype(float)

    # ---------------------------------------------------------------------
    # 3) Output dict
    # ---------------------------------------------------------------------
    profiles: Dict[str, pd.Series] = {}

    # Legacy keys - per device
    profiles["space_heating_air_to_air"] = p_space_air_to_air.astype(float)
    profiles["space_heating_air_to_water"] = p_space_air_to_water.astype(float)
    profiles["space_heating_direct"] = p_space_direct.astype(float)
    profiles["space_heating_floor"] = p_space_floor.astype(float)

    profiles["space_heating_total"] = space_heating_total.astype(float)
    profiles["space_heating_unserved_th"] = space_heating_unserved_th.astype(float)

    # Cooling
    profiles["space_cooling_air_to_air_ac"] = p_space_cooling_ac.astype(float)
    profiles["space_cooling_total"] = space_cooling_total.astype(float)
    profiles["space_cooling_unserved_th"] = space_cooling_unserved_th.astype(float)

    # Temperatures (only meaningful in 2R2C)
    profiles["t_air_c"] = t_air_c.astype(float)
    profiles["t_mass_c"] = t_mass_c.astype(float)

    # DHW
    profiles["dhw_electric_boiler"] = p_dhw_boiler.astype(float)
    profiles["dhw_hp"] = p_dhw_hp.astype(float)
    profiles["dhw_total"] = dhw_total.astype(float)

    profiles["aggregated"] = (space_heating_total + space_cooling_total + dhw_total).astype(float)

    # Diagnostics (useful to understand why the curves are zero)
    profiles["dbg_heat_allowed"] = dbg_heat_allowed.astype(float)
    profiles["dbg_cool_allowed"] = dbg_cool_allowed.astype(float)
    profiles["dbg_heating_on"] = dbg_heating_on.astype(float)
    profiles["dbg_cooling_on"] = dbg_cooling_on.astype(float)
    profiles["dbg_q_heat_req_th"] = dbg_q_heat_req_th.astype(float)
    profiles["dbg_q_cool_req_th"] = dbg_q_cool_req_th.astype(float)
    profiles["dbg_stability_coef"] = dbg_stability_coef.astype(float)

    return profiles