"""
Scheda Streamlit: Clima (riscaldamento, raffrescamento, ACS) per un singolo consumatore.

Questa scheda è la controparte UI del modulo core :mod:`cer_core.consumatori.clima` e ha due ruoli:

1) **Configurazione**: popola e mantiene coerente ``consumer["devices"]["climate"]`` (salvato in
   ``data/sessions/<SESSION>/consumers.json`` tramite ``save_consumers_json``).

2) **Simulazione + cache**: quando l’utente richiede il calcolo, invoca
   ``cer_core.consumatori.clima.build_climate_profiles(...)`` e salva in cache la curva elettrica
   aggregata del sottosistema clima:
   ``data/sessions/<SESSION>/cache/consumer_<id>/climate.csv`` (potenza in **kW** su INDEX 15 minuti).

Inoltre, per compatibilità con l’orchestratore ``pages/2_Consumatori.py``, questo modulo espone alcune
utility riusate anche da altre schede (plot/KPI/rescaling) e funzioni di lettura cache.

Assunzioni e invarianti
- Timebase: ``INDEX`` è un ``pd.DatetimeIndex`` timezone-aware in **UTC** e (tipicamente) equispaziato
  a 15 minuti. Tutte le serie prodotte/consumate sono allineate a ``INDEX``.
- Unità: le curve salvate in cache sono potenze elettriche in **kW**; l’energia si ricava a valle come
  ``kWh = sum(P_kW) * dt_hours``.
- Riproducibilità: il core clima non usa RNG; a parità di input (config + temperatura esterna) l’output
  è deterministico.

Dependency injection (vincolo architetturale)
Questo file è progettato per essere importato da ``pages/2_Consumatori.py``, che inietta nel namespace
del modulo variabili/global helper (``SESSION_DIR``, ``BASE_SEED``, ``save_consumers_json``,
``ensure_device``, ecc.). Le funzioni di pannello assumono che tali simboli siano già disponibili.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from cer_core.consumatori import lavanderia as L
from cer_core.consumatori import clima as CL

# --- helper: ensure nominal defaults when device marked present but nominal power missing or <=0 ---
def _ensure_nominal_defaults(dev):
    try:
        present = bool(dev.get("present", False))
    except Exception:
        present = False
    if not present:
        return
    # heat pumps nominal heat
    if "p_heat_nom_kw" in dev:
        try:
            if float(dev.get("p_heat_nom_kw", 0) or 0) <= 0:
                dev["p_heat_nom_kw"] = 3.5
        except Exception:
            dev["p_heat_nom_kw"] = 3.5
    # resistive / floor heating nominal electric
    if "p_el_nom_kw" in dev:
        try:
            if float(dev.get("p_el_nom_kw", 0) or 0) <= 0:
                dev["p_el_nom_kw"] = 3.0
        except Exception:
            dev["p_el_nom_kw"] = 3.0
    # cooling nominal
    if "p_cool_nom_kw" in dev:
        try:
            if float(dev.get("p_cool_nom_kw", 0) or 0) <= 0:
                dev["p_cool_nom_kw"] = 3.5
        except Exception:
            dev["p_cool_nom_kw"] = 3.5
# --- end helper ---

# --- Clima ---
CLIMATE_DEF = {
    # --- abitazione (UI base) ---
    "housing": {
        "area_m2": 80.0,
        "ceiling_height_m": 2.7,
        "insulation_level": "medio",   # scarso | medio | buono | ottimo
        "thermal_mass_level": "medium" # light | medium | heavy
    },

    # --- comune / 2R2C ---
    "thermal": {
        "model": "2R2C",
        "leakage_ach": 0.5,                 # ricambi/h (infiltrazioni)
        "ceiling_height_m": 2.7,
        "thermal_mass_level": "medium",     # light | medium | heavy
        "deadband_c": 0.5,
        # IMPORTANT: il modello clima (2R2C) deve restare indipendente dagli altri
        # carichi elettrici del consumatore (baseload/cucina/lavanderia/occupancy).
        # Questi carichi vanno sommati SOLO a valle per ottenere la curva elettrica
        # aggregata del consumatore; non devono entrare nel bilancio termico.
        "use_auto_internal_gains": False,
        "internal_gains_fraction": 0.90,    # quota dei carichi che diventa calore interno
        "gains_to_mass_fraction": 0.60,     # quota gains su nodo massa (resto su aria)
        "auto_leakage_from_housing": True,
    },

    # --- domanda riscaldamento ---
    "heating": {
        "t_set_heating_c": 20.0,
        "heating_enable_temp_c": 15.0,
        "t_design_outdoor_c": -5.0,
        "floor_area_m2": 80.0,
        "insulation_level": "medio",        # scarso | medio | buono
        "design_heat_load_kw": None,        # 0/None => stima
        "comfort_mode": "always",           # always | custom
        "comfort_windows": {},              # {"mon": [["06:30","09:00"], ...], ...}
    },

    # --- domanda raffrescamento ---
    "cooling": {
        "enabled": False,
        "t_set_cooling_c": 26.0,
        "cooling_enable_temp_c": 24.0,
        "design_cool_load_kw": None,        # 0/None => stima
        "comfort_mode": "always",           # always | custom
        "comfort_windows": {},              # {"mon": [["12:00","18:00"], ...], ...}
    },

    # --- generatori riscaldamento ---
    "air_to_air_hp": {
        "present": False,
        "p_heat_nom_kw": 3.5,
        "cop_at_7c": 4.0,
        "cop_at_minus7c": 2.5,
        "n_units": 1,
        "priority": 1,
        "max_share_of_load": 1.0,
    },
    "air_to_water_hp": {
        "present": False,
        "p_heat_nom_kw": 6.0,
        "cop_at_7c": 3.2,
        "cop_at_minus7c": 2.0,
        "emitter_type": "radiators",
        "priority": 1,
        "max_share_of_load": 1.0,
    },
    "direct_heating": {
        "present": False,
        "p_el_nom_kw": 2.0,
        "priority": 3,
        "max_share_of_load": 1.0,
    },
    "floor_heating": {
        "present": False,
        "p_el_nom_kw": 3.0,
        "thermal_inertia_hours": 4.0,
        "priority": 2,
        "max_share_of_load": 1.0,
    },

    # --- ACS ---
    "dhw_electric_boiler": {
        "present": True,
        "p_el_kw": 1.5,
        "volume_liters": 80.0,
        "t_set_c": 55.0,
        "people": 2,
        "morning_window": [6, 9],
        "evening_window": [18, 22],
    },
    "dhw_hp": {
        "present": False,
        "p_heat_nom_kw": 2.0,
        "cop_at_7c": 2.5,
        "cop_at_minus7c": 2.0,
        "volume_liters": 200.0,
        "t_set_c": 50.0,
        "people": 3,
        "morning_window": [6, 9],
        "evening_window": [18, 22],
    },

    # --- generatore raffrescamento (split aria-aria) ---
    "air_to_air_ac": {
        "present": False,
        "eer_at_35c": 3.2,
        "eer_at_27c": 4.0,
        "p_cool_nom_kw": 3.5,
        "n_units": 1,
        "priority": 1,
        "max_share_of_load": 1.0,
    },
}
def _deep_merge_defaults(dst: dict, defaults: dict) -> dict:
    """Merge ricorsivo: aggiunge chiavi mancanti senza sovrascrivere l'esistente."""
    import copy
    for k, v in (defaults or {}).items():
        if k not in dst:
            dst[k] = copy.deepcopy(v)
        else:
            if isinstance(dst.get(k), dict) and isinstance(v, dict):
                _deep_merge_defaults(dst[k], v)
    return dst

def get_climate_device(consumer: dict) -> dict:
    """
    Restituisce (e se necessario crea) la sezione ``consumer["devices"]["climate"]``.

    La struttura è un dizionario annidato compatibile con ``CLIMATE_DEF``. In particolare:

    - se il consumer non ha ancora il device ``"climate"``, viene creato come deep-copy di
      :data:`CLIMATE_DEF`;
    - se esiste già (sessioni legacy), le nuove chiavi introdotte in ``CLIMATE_DEF`` vengono
      aggiunte senza sovrascrivere valori già presenti (merge ricorsivo);
    - per alcuni generatori, se la potenza nominale è mancante o non positiva, viene ripristinato un
      default > 0 per evitare curve nulle dovute a configurazioni salvate erroneamente.

    Nota: la persistenza su disco avviene **solo** quando la pagina orchestratrice invoca
    ``save_consumers_json(SESSION_DIR, consumers)`` (iniettato via dependency injection).
    """
    import copy
    devs = consumer.setdefault("devices", {})
    dev_climate = devs.get("climate")
    if dev_climate is None:
        dev_climate = copy.deepcopy(CLIMATE_DEF)
        devs["climate"] = dev_climate
    else:
        # aggiungi eventuali nuove chiavi introdotte nei default
        _deep_merge_defaults(dev_climate, CLIMATE_DEF)
    # Ensure that nominal powers are always sane (>0) using CLIMATE_DEF defaults.
    # Questo evita che una sessione salvata con p_nom=0 continui a produrre curve nulle.
    for _k in ("air_to_air_hp", "air_to_water_hp", "direct_heating", "floor_heating", "air_to_air_ac"):
        d = dev_climate.setdefault(_k, {})
        d_def = (CLIMATE_DEF.get(_k) or {})
        for _pkey in ("p_heat_nom_kw", "p_el_nom_kw", "p_cool_nom_kw"):
            if _pkey in d_def:
                try:
                    if float(d.get(_pkey, 0) or 0) <= 0:
                        d[_pkey] = float(d_def[_pkey])
                except Exception:
                    d[_pkey] = float(d_def[_pkey])
        # If the device is marked present, ensure again (handles missing keys/types)
        _ensure_nominal_defaults(d)

    return dev_climate


# ------------------------------------------------------------
# Comfort windows helpers (heating/cooling)
# ------------------------------------------------------------

_WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_WEEKDAY_LABELS_IT = {
    "mon": "Lun", "tue": "Mar", "wed": "Mer", "thu": "Gio", "fri": "Ven", "sat": "Sab", "sun": "Dom"
}
_WEEKDAY_TO_INT = {k: i for i, k in enumerate(_WEEKDAY_KEYS)}


def _time_to_str(t: "dt.time") -> str:
    return f"{t.hour:02d}:{t.minute:02d}"


def _str_to_time(s: str) -> "dt.time":
    hh, mm = s.split(":")
    return dt.time(int(hh), int(mm))


def _time_to_float_hours(t: "dt.time") -> float:
    return float(t.hour) + float(t.minute) / 60.0


def _normalize_windows_dict(windows: Optional[dict]) -> dict:
    """Ensures a complete dict with keys mon..sun and list-of-[start,end] strings."""
    out = {k: [] for k in _WEEKDAY_KEYS}
    if not windows:
        return out
    for k, v in windows.items():
        kk = str(k).lower().strip()
        if kk in out and isinstance(v, list):
            cleaned = []
            for pair in v:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = str(pair[0]), str(pair[1])
                    if re.match(r"^\d{2}:\d{2}$", a) and re.match(r"^\d{2}:\d{2}$", b):
                        cleaned.append([a, b])
            out[kk] = cleaned
    return out


def _windows_to_comfort_schedule(windows: dict) -> "CL.ComfortSchedule":
    """Convert {"mon":[["06:30","09:00"],...], ...} -> ComfortSchedule(per_day={0:[(6.5,9.0),...]...}).

    Supports intervals that cross midnight (e.g., 22:00-02:00) by keeping start>end; the core handles wrapping.
    """
    per_day: Dict[int, List[Tuple[float, float]]] = {}
    windows = _normalize_windows_dict(windows)
    for day_key, pairs in windows.items():
        wd = _WEEKDAY_TO_INT.get(day_key)
        if wd is None:
            continue
        floats: List[Tuple[float, float]] = []
        for a, b in pairs:
            ta = _str_to_time(a)
            tb = _str_to_time(b)
            floats.append((_time_to_float_hours(ta), _time_to_float_hours(tb)))
        if floats:
            per_day[wd] = floats
    return CL.ComfortSchedule(per_day=per_day)




def _day_profile_to_comfort_schedule(day_profile) -> "CL.ComfortSchedule":
    """
    Convert day_profile=[["06:30","09:00"],["18:00","23:00"]] into a ComfortSchedule
    applied to all days (0..6). If input is empty/None -> always-on schedule (empty per_day).
    """
    if not day_profile:
        return CL.ComfortSchedule()

    pairs = []
    for p in day_profile:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        s, e = str(p[0]), str(p[1])
        if not re.match(r"^\d{1,2}:\d{2}$", s) or not re.match(r"^\d{1,2}:\d{2}$", e):
            continue
        pairs.append((_hhmm_to_hours(s), _hhmm_to_hours(e)))

    if not pairs:
        return CL.ComfortSchedule()

    per_day = {d: pairs[:] for d in range(7)}
    return CL.ComfortSchedule(per_day=per_day)


def _hhmm_pad(x: str) -> str:
    """Normalizza 'H:MM' -> 'HH:MM'."""
    try:
        parts = x.split(":")
        h = int(parts[0])
        m = int(parts[1])
        h = max(0, min(23, h))
        m = max(0, min(59, m))
        return f"{h:02d}:{m:02d}"
    except Exception:
        return x


def _hhmm_to_hours(s: str) -> float:
    """Converte 'H:MM' o 'HH:MM' in ore decimali (es. '06:30' -> 6.5)."""
    s_norm = _hhmm_pad(str(s).strip())
    t = _str_to_time(s_norm)
    return _time_to_float_hours(t)


def _day_profile_editor(prefix: str, title: str, current: Optional[dict]) -> dict:
    """
    Editor semplificato basato su "giorno tipo" con griglia a fasce orarie (come Lavanderia,
    ma senza distinzione per giorno della settimana).

    Ritorna sempre:
      - comfort_mode: "custom"
      - day_profile: lista di coppie ["HH:MM", "HH:MM"] che valgono per tutti i giorni.

    Convenzione:
      - se day_profile è vuoto -> comfort sempre attivo (24/7), gestito nel core come schedule vuoto.
    """
    st.markdown(f"#### {title}")
    st.caption(
        "Seleziona le fasce orarie in cui l'impianto può essere attivo in una giornata tipo. "
        "Se non selezioni alcuna fascia, il comfort è considerato attivo 24 ore su 24."
    )

    current = current or {}

    # Recupera eventuale configurazione salvata
    raw_prof = current.get("day_profile") or current.get("comfort_day_profile") or []
    existing_pairs = set()
    for p in raw_prof:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        s = str(p[0]).strip()
        e = str(p[1]).strip()
        if re.match(r"^\d{1,2}:\d{2}$", s) and re.match(r"^\d{1,2}:\d{2}$", e):
            s_norm = _hhmm_pad(s)
            e_norm = _hhmm_pad(e)
            existing_pairs.add((s_norm, e_norm))

    # Slot e bande temporali allineati alla logica di Lavanderia
    slot_labels: List[str] = list(getattr(L, "SLOT_LABELS", [
        "06-08", "08-10", "10-12", "12-14",
        "14-16", "16-18", "18-20", "20-22",
        "22-24", "00-06",
    ]))
    slot_bands: List[Tuple[int, int]] = list(getattr(L, "SLOT_BANDS", [
        (6, 8), (8, 10), (10, 12), (12, 14),
        (14, 16), (16, 18), (18, 20), (20, 22),
        (22, 24), (0, 6),
    ]))

    # Header griglia
    cols = st.columns([1] + [1] * len(slot_labels))
    cols[0].markdown("**Giorno**")
    for j, lab in enumerate(slot_labels):
        cols[j + 1].markdown(f"**{lab}**")

    # Un'unica riga: "Giorno tipo"
    rcols = st.columns([1] + [1] * len(slot_labels))
    rcols[0].write("Giorno tipo")

    selected_slots: List[Tuple[int, int]] = []
    for j, (h_start, h_end) in enumerate(slot_bands):
        # mappa banda in coppia di orari stringa
        if (h_start, h_end) == (22, 24):
            s_slot = "22:00"
            e_slot = "23:59"
        else:
            s_slot = f"{h_start % 24:02d}:00"
            e_slot = f"{h_end % 24:02d}:00"

        init_sel = (s_slot, e_slot) in existing_pairs
        if rcols[j + 1].checkbox("", value=init_sel, key=f"{prefix}_slot_{j}"):
            selected_slots.append((h_start, h_end))

    # Conversione degli slot selezionati in lista di ["HH:MM","HH:MM"]
    day_profile: List[List[str]] = []
    for h_start, h_end in selected_slots:
        if (h_start, h_end) == (22, 24):
            s = "22:00"
            e = "23:59"
        else:
            s = f"{h_start % 24:02d}:00"
            e = f"{h_end % 24:02d}:00"
        day_profile.append([s, e])

    return {"comfort_mode": "custom", "day_profile": day_profile}
def build_climate_config_from_device_dict(dev_climate: dict) -> CL.ClimateConfig:
    """
    Converte il dizionario ``devices["climate"]`` (formato UI/JSON) in un oggetto dataclass
    :class:`cer_core.consumatori.clima.ClimateConfig`.

    Questo passaggio ha due motivazioni:

    - **validazione e normalizzazione**: la UI salva numeri come ``int/float/str``; qui vengono
      convertiti ai tipi richiesti dal core e vengono applicati default coerenti;
    - **separazione delle responsabilità**: il core opera su configurazioni tipizzate e non dipende
      dalla struttura Streamlit/JSON.

    Contratto
    - Input: ``dev_climate`` è tipicamente il risultato di :func:`get_climate_device`.
    - Output: un :class:`~cer_core.consumatori.clima.ClimateConfig` pronto per
      ``CL.build_climate_profiles(INDEX, cfg, T_AIR_15, ...)``.

    Note
    - Le ``comfort windows`` della UI sono salvate come ``["HH:MM","HH:MM"]`` (giorno tipo); qui sono
      convertite in :class:`~cer_core.consumatori.clima.ComfortSchedule` (ore decimali).
    - Le potenze nominali sono in **kW** (elettrici o termici a seconda del generatore), coerenti con
      il core.
    """
    # --- Thermal model (2R2C / static_UA) ---
    th_dict = dev_climate.get("thermal", {}) or {}
    thermal_cfg = CL.ThermalModelConfig(
        model=str(th_dict.get("model", "2R2C")),
        leakage_ach=float(th_dict.get("leakage_ach", 0.5)),
        ceiling_height_m=float(th_dict.get("ceiling_height_m", 2.7)),
        thermal_mass_level=str(th_dict.get("thermal_mass_level", "medium")),
        deadband_c=float(th_dict.get("deadband_c", 0.5)),
        internal_gains_fraction=float(th_dict.get("internal_gains_fraction", 0.90)),
        gains_to_mass_fraction=float(th_dict.get("gains_to_mass_fraction", 0.60)),
        auto_leakage_from_housing=bool(th_dict.get("auto_leakage_from_housing", True)),
        mass_time_constant_h=th_dict.get("mass_time_constant_h", None),
    )

    # --- Heating demand ---
    heating_dict = dev_climate.get("heating", {}) or {}

    heating_mode = str(heating_dict.get("comfort_mode", "custom")).lower().strip()
    if heating_mode == "custom":
        heating_schedule = _day_profile_to_comfort_schedule(
            heating_dict.get("day_profile") or heating_dict.get("comfort_day_profile")
        )
    else:
        # fallback: comfort sempre attivo
        heating_schedule = CL.ComfortSchedule()

    design_load = heating_dict.get("design_heat_load_kw")
    if design_load in (0, 0.0):
        design_load = None

    heating_cfg = CL.HeatingDemandConfig(
        t_set_heating_c=float(heating_dict.get("t_set_heating_c", 20.0)),
        heating_enable_temp_c=float(heating_dict.get("heating_enable_temp_c", 15.0)),
        t_design_outdoor_c=float(heating_dict.get("t_design_outdoor_c", -5.0)),
        floor_area_m2=float(heating_dict.get("floor_area_m2", 80.0)),
        insulation_level=str(heating_dict.get("insulation_level", "medio")),
        design_heat_load_kw=design_load,
        comfort_schedule=heating_schedule,
    )

    # --- Cooling demand ---
    cool_dict = dev_climate.get("cooling", {}) or {}

    cooling_mode = str(cool_dict.get("comfort_mode", "custom")).lower().strip()
    if cooling_mode == "custom":
        cooling_schedule = _day_profile_to_comfort_schedule(
            cool_dict.get("day_profile") or cool_dict.get("comfort_day_profile")
        )
    else:
        cooling_schedule = CL.ComfortSchedule()

    cool_design = cool_dict.get("design_cool_load_kw")
    if cool_design in (0, 0.0):
        cool_design = None

    cooling_cfg = CL.CoolingDemandConfig(
        enabled=bool(cool_dict.get("enabled", False)),
        t_set_cooling_c=float(cool_dict.get("t_set_cooling_c", 26.0)),
        cooling_enable_temp_c=float(cool_dict.get("cooling_enable_temp_c", 24.0)),
        design_cool_load_kw=cool_design,
        comfort_schedule=cooling_schedule,
    )



    # --- Housing (UI base) ---
    house_dict = dev_climate.get("housing", {}) or {}
    housing_cfg = CL.HousingConfig(
        area_m2=float(house_dict.get("area_m2", heating_cfg.floor_area_m2)),
        ceiling_height_m=float(house_dict.get("ceiling_height_m", thermal_cfg.ceiling_height_m)),
        insulation_level=str(house_dict.get("insulation_level", heating_cfg.insulation_level)),
        thermal_mass_level=str(house_dict.get("thermal_mass_level", thermal_cfg.thermal_mass_level)),
    )

    # --- Air-to-air HP (heating) ---

    aa_dict = dev_climate.get("air_to_air_hp", {}) or {}
    air_to_air_hp = CL.AirToAirHeatPumpConfig(
        present=bool(aa_dict.get("present", False)),
        p_heat_nom_kw=float(aa_dict.get("p_heat_nom_kw", 0.0)),
        cop_at_7c=float(aa_dict.get("cop_at_7c", 4.0)),
        cop_at_minus7c=float(aa_dict.get("cop_at_minus7c", 2.5)),
        n_units=int(aa_dict.get("n_units", 1)),
        priority=int(aa_dict.get("priority", 1)),
        max_share_of_load=float(aa_dict.get("max_share_of_load", 1.0)),
    )

    # --- Air-to-water HP (heating) ---
    aw_dict = dev_climate.get("air_to_water_hp", {}) or {}
    air_to_water_hp = CL.AirToWaterHeatPumpConfig(
        present=bool(aw_dict.get("present", False)),
        p_heat_nom_kw=float(aw_dict.get("p_heat_nom_kw", 0.0)),
        cop_at_7c=float(aw_dict.get("cop_at_7c", 3.2)),
        cop_at_minus7c=float(aw_dict.get("cop_at_minus7c", 2.0)),
        emitter_type=str(aw_dict.get("emitter_type", "radiators")),
        priority=int(aw_dict.get("priority", 1)),
        max_share_of_load=float(aw_dict.get("max_share_of_load", 1.0)),
    )

    # --- Direct heating ---
    de_dict = dev_climate.get("direct_heating", {}) or {}
    direct_heating = CL.DirectElectricHeatingConfig(
        present=bool(de_dict.get("present", False)),
        p_el_nom_kw=float(de_dict.get("p_el_nom_kw", 0.0)),
        priority=int(de_dict.get("priority", 3)),
        max_share_of_load=float(de_dict.get("max_share_of_load", 1.0)),
    )

    # --- Floor heating ---
    fl_dict = dev_climate.get("floor_heating", {}) or {}
    floor_heating = CL.ElectricFloorHeatingConfig(
        present=bool(fl_dict.get("present", False)),
        p_el_nom_kw=float(fl_dict.get("p_el_nom_kw", 0.0)),
        thermal_inertia_hours=float(fl_dict.get("thermal_inertia_hours", 4.0)),
        priority=int(fl_dict.get("priority", 2)),
        max_share_of_load=float(fl_dict.get("max_share_of_load", 1.0)),
    )

    # --- Air-to-air AC (cooling) ---
    ac_dict = dev_climate.get("air_to_air_ac", {}) or {}
    air_to_air_ac = CL.AirToAirACConfig(
        present=bool(ac_dict.get("present", False)),
        p_cool_nom_kw=float(ac_dict.get("p_cool_nom_kw", 0.0)),
        eer_at_27c=float(ac_dict.get("eer_at_27c", 4.0)),
        eer_at_35c=float(ac_dict.get("eer_at_35c", 3.2)),
        n_units=int(ac_dict.get("n_units", 1)),
        priority=int(ac_dict.get("priority", 1)),
        max_share_of_load=float(ac_dict.get("max_share_of_load", 1.0)),
    )

    # --- DHW (boiler) ---
    boil_dict = dev_climate.get("dhw_electric_boiler", {}) or {}
    mw_b = boil_dict.get("morning_window", [6, 9])
    ew_b = boil_dict.get("evening_window", [18, 22])
    morning_window_b = (mw_b[0], mw_b[1]) if isinstance(mw_b, (list, tuple)) and len(mw_b) == 2 else (6, 9)
    evening_window_b = (ew_b[0], ew_b[1]) if isinstance(ew_b, (list, tuple)) and len(ew_b) == 2 else (18, 22)

    dhw_electric_boiler = CL.ElectricBoilerConfig(
        present=bool(boil_dict.get("present", False)),
        p_el_kw=float(boil_dict.get("p_el_kw", 0.0)),
        volume_liters=float(boil_dict.get("volume_liters", 80.0)),
        t_set_c=float(boil_dict.get("t_set_c", 55.0)),
        people=int(boil_dict.get("people", 0)),
        morning_window=morning_window_b,
        evening_window=evening_window_b,
    )

    # --- DHW (hp) ---
    hp_dict = dev_climate.get("dhw_hp", {}) or {}
    mw_hp = hp_dict.get("morning_window", [6, 9])
    ew_hp = hp_dict.get("evening_window", [18, 22])
    morning_window_hp = (mw_hp[0], mw_hp[1]) if isinstance(mw_hp, (list, tuple)) and len(mw_hp) == 2 else (6, 9)
    evening_window_hp = (ew_hp[0], ew_hp[1]) if isinstance(ew_hp, (list, tuple)) and len(ew_hp) == 2 else (18, 22)

    dhw_hp = CL.DHWHeatPumpConfig(
        present=bool(hp_dict.get("present", False)),
        p_heat_nom_kw=float(hp_dict.get("p_heat_nom_kw", 0.0)),
        cop_at_7c=float(hp_dict.get("cop_at_7c", 2.5)),
        cop_at_minus7c=float(hp_dict.get("cop_at_minus7c", 2.0)),
        volume_liters=float(hp_dict.get("volume_liters", 200.0)),
        t_set_c=float(hp_dict.get("t_set_c", 50.0)),
        people=int(hp_dict.get("people", 0)),
        morning_window=morning_window_hp,
        evening_window=evening_window_hp,
    )

    return CL.ClimateConfig(
        housing=housing_cfg,
        thermal=thermal_cfg,
        heating=heating_cfg,
        cooling=cooling_cfg,
        air_to_air_hp=air_to_air_hp,
        air_to_water_hp=air_to_water_hp,
        direct_heating=direct_heating,
        floor_heating=floor_heating,
        air_to_air_ac=air_to_air_ac,
        dhw_electric_boiler=dhw_electric_boiler,
        dhw_hp=dhw_hp,
    )



def prepare_curve_for_plot(curve: pd.Series, curve_view_mode: str, curve_view_month):
    """
    Prepara una curva (kW su INDEX 15 min) per la visualizzazione Streamlit.

    - Modalità ``"annuale"``: downsample a 1 ora tramite media (profilo leggibile a scala annuale).
    - Modalità ``"mensile"``: mantiene la risoluzione nativa e filtra il solo mese selezionato.

    Parametri
    - curve: serie a passo 15 min (o comunque equispaziata) indicizzata da timestamp UTC.
    - curve_view_mode: stringa storica ``"annuale"`` o ``"mensile"``.
    - curve_view_month: ``(year, month)`` se mensile, altrimenti ``None``.
    """
    # annuale: 1h; mensile: 15 min sul mese
    if curve_view_mode == "annuale" or curve_view_month is None:
        return curve.resample("H").mean()
    year, month = curve_view_month
    mask_month = (curve.index.year == year) & (curve.index.month == month)
    return curve[mask_month]

def compute_kpis(curve: pd.Series, index: pd.DatetimeIndex):
    """
    Calcola KPI energetici a partire da una curva di potenza (kW).

    Restituisce una tripla:
    - energia nel periodo simulato [kWh]
    - energia media giornaliera [kWh/giorno]
    - energia annua stimata [kWh/anno], ottenuta scalando la media giornaliera a 365 giorni

    Nota: la conversione kW→kWh usa ``dt_hours`` stimato dal primo passo dell’indice della curva.
    """
    if len(curve) > 1:
        dt_hours = (curve.index[1] - curve.index[0]).total_seconds() / 3600.0
    else:
        dt_hours = 0.0
    kwh_periodo = float(curve.sum() * dt_hours)
    giorni = (index.max() - index.min()).days + 1
    kwh_giorno = kwh_periodo / max(giorni, 1)
    kwh_annuo = kwh_giorno * 365.0
    return kwh_periodo, kwh_giorno, kwh_annuo

def rescale_curve_to_annual_target(curve_kw: pd.Series, index: pd.DatetimeIndex, target_annual_kwh: float):
    """
    Riscala una curva di potenza (kW) affinché l'energia annua stimata corrisponda a target_annual_kwh.

    Nota: se l'indice non copre un anno intero, il target viene riproporzionato sul periodo simulato
    (target_period = target_annual_kwh * giorni_simulati / 365).
    """
    if target_annual_kwh is None:
        return None, {'reason': 'target_missing'}
    try:
        target_annual_kwh = float(target_annual_kwh)
    except Exception:
        return None, {'reason': 'target_not_numeric'}
    if target_annual_kwh <= 0:
        return None, {'reason': 'target_non_positive'}

    curve = curve_kw.astype(float).fillna(0.0).clip(lower=0.0)
    if len(curve) < 2:
        return None, {'reason': 'curve_too_short'}

    dt_hours = (curve.index[1] - curve.index[0]).total_seconds() / 3600.0
    e_period_kwh_raw = float(curve.sum() * dt_hours)
    giorni = (index.max() - index.min()).days + 1
    target_period_kwh = float(target_annual_kwh) * (float(giorni) / 365.0)

    if e_period_kwh_raw <= 0:
        return None, {'reason': 'raw_energy_zero', 'e_period_kwh_raw': e_period_kwh_raw}

    scale_factor = target_period_kwh / e_period_kwh_raw
    scaled = (curve * scale_factor).rename(curve_kw.name or 'total_load')
    meta = {
        'dt_hours': dt_hours,
        'days_simulated': int(giorni),
        'e_period_kwh_raw': e_period_kwh_raw,
        'target_annual_kwh': float(target_annual_kwh),
        'target_period_kwh': target_period_kwh,
        'scale_factor': float(scale_factor),
    }
    return scaled, meta

def _load_cached_curve(
    session_dir: Path,
    consumer_id: int,
    filename: str,
    index: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """
    Carica una curva componente dalla cache per-consumer e la riallinea a ``index``.

    Il formato atteso è un CSV con:
    - colonna ``timestamp`` (parseabile da pandas; viene forzata in UTC),
    - almeno una colonna dati (si usa la prima).

    In caso di file mancante o errore di parsing, ritorna ``None`` (failure non bloccante).
    """
    p = session_dir / "cache" / f"consumer_{consumer_id}" / filename
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        s = df.iloc[:, 0].astype(float)
        s = s.reindex(index).fillna(0.0)
        return s
    except Exception:
        return None

def build_internal_gains_from_cache(session_dir: Path, consumer_id: int, index: pd.DatetimeIndex) -> pd.Series:
    """
    Stima i guadagni interni a partire dalle curve elettriche già calcolate e salvate in cache.

    Questa funzione somma (quando presenti) le componenti:
    ``baseload.csv``, ``kitchen.csv``, ``laundry_total.csv``, ``occupancy_15min.csv``.

    Output
    - una serie in **kW_el** su ``index`` chiamata ``internal_gains_kw_el``.

    Nota metodologica
    - Nel pannello clima corrente, per scelta progettuale, questi gains NON vengono passati al core
      (``internal_gains_kw_el=None``) per mantenere il bilancio termico indipendente dagli altri
      carichi elettrici. La funzione resta disponibile per esperimenti/varianti.
    """
    parts = [
        _load_cached_curve(session_dir, consumer_id, "baseload.csv", index),
        _load_cached_curve(session_dir, consumer_id, "kitchen.csv", index),
        _load_cached_curve(session_dir, consumer_id, "laundry_total.csv", index),
        _load_cached_curve(session_dir, consumer_id, "occupancy_15min.csv", index),
    ]
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.Series(0.0, index=index, name="internal_gains_kw_el")
    tot = parts[0]
    for s in parts[1:]:
        tot = tot.add(s, fill_value=0.0)
    return tot.rename("internal_gains_kw_el").astype(float)

def laundry_device_config(device_name: str, title: str, defaults: dict):
    """
    UI di configurazione per lavatrice/asciugatrice.
    NON simula, NON plotta. Ritorna (present, dev_inputs) pronti per L.simulate.
    """
    consumer_id = consumer.get("id")
    dev = ensure_device(consumer, device_name, defaults)

    present = st.checkbox(
        f"Possiede {title.lower()}",
        value=bool(dev.get("present", False)),
        key=f"c{consumer_id}_{device_name}_present",
    )
    dev["present"] = bool(present)

    if not dev["present"]:
        if st.button(f"Abilita {title.lower()}", key=f"c{consumer_id}_add_{device_name}"):
            dev.update(defaults)
            dev["present"] = True
            save_consumers_json(SESSION_DIR, consumers)
            st.success(f"{title} aggiunta al consumatore.")
        return False, None

    # ---- qui riusa i tuoi stessi campi del device_panel ----
    cols = st.columns(3)
    dev["n_devices"] = cols[0].number_input(
        "Numero dispositivi", min_value=1, value=int(dev.get("n_devices", 1)), step=1,
        key=f"c{consumer_id}_{device_name}_n"
    )

    dev["cycles_per_week"] = cols[2].number_input(
        "Cicli a settimana (0 = nessun utilizzo)", min_value=0, value=int(dev.get("cycles_per_week") or 0), step=1,
        key=f"c{consumer_id}_{device_name}_cpw"
    )

    cols2 = st.columns(3)

    if device_name == "dryer":
        season_opts = ["tutto_anno", "inverno"]
        current_season = dev.get("seasonality", "tutto_anno")
        if current_season not in season_opts:
            current_season = "tutto_anno"
        dev["seasonality"] = cols2[0].selectbox(
            "Stagionalità",
            season_opts,
            index=season_opts.index(current_season),
            key=f"c{consumer_id}_{device_name}_season"
        )
    else:
        # per washer non vogliamo questo parametro
        dev.pop("seasonality", None)

    options = list(L.ENERGY_CLASS_OPTIONS)
    current_class = dev.get("energy_class", None)
    if current_class not in options:
        current_class = options[0] if options else None
    dev["energy_class"] = cols2[1].selectbox(
        "Classe energetica", options,
        index=options.index(current_class),
        key=f"c{consumer_id}_{device_name}_class"
    )

    dev["P_nominal_W"] = cols2[2].number_input(
        "Potenza nominale (W)", min_value=0, value=int(dev.get("P_nominal_W") or 0), step=50,
        key=f"c{consumer_id}_{device_name}_pnom"
    )

    if device_name in ("washer", "dryer"):
        dev["modes_selected"] = st.multiselect(
            "Modalità di utilizzo",
            L.WASHER_MODE_OPTIONS,
            default=dev.get("modes_selected") or ["standard"],
            key=f"c{consumer_id}_{device_name}_modes",
        )

    st.markdown("#### Finestre di utilizzo (griglia)")
    parsed_matrix = L.parse_start_matrix(dev.get("start_matrix"))
    init_matrix = dict_to_matrix(parsed_matrix)
    grid = start_matrix_editor(f"c{consumer_id}_{device_name}_grid", init_matrix)

    available = count_weekly_slots_from_grid(grid)
    cpw = int(dev.get("cycles_per_week") or 0)

    if cpw > 0 and available < cpw:
        st.error(
            f"Slot disponibili a settimana ({available}) < cicli richiesti ({cpw}). "
            "Aumenta le finestre ammissibili nella griglia o riduci i cicli/settimana."
        )

    if st.button(f"💾 Salva parametri {title.lower()}", key=f"c{consumer_id}_{device_name}_save"):
        dev["start_matrix"] = matrix_to_dict(grid)
        save_consumers_json(SESSION_DIR, consumers)
        st.success("Parametri salvati (inclusa la griglia).")

    # prepara dict inputs per la simulazione
    dev_inputs = dict(dev)
    if device_name == "washer":
        dev_inputs.pop("cycle_duration_min", None)
        dev_inputs.pop("seasonality", None)
    dev_inputs["start_matrix"] = matrix_to_dict(grid)

    return True, dev_inputs

def laundry_panel():
    consumer_id = consumer.get("id")

    with st.expander("🧺 Lavanderia (totale: lavatrice + asciugatrice)", expanded=True):
        tab_w, tab_d = st.tabs(["Lavatrice", "Asciugatrice"])

        with tab_w:
            w_present, w_inputs = laundry_device_config("washer", "Lavatrice", W_DEF)

        with tab_d:
            d_present, d_inputs = laundry_device_config("dryer", "Asciugatrice", D_DEF)

        st.markdown("---")

        # Bottone unico: calcola TOTALE
        laundry_curve = None
        if st.button("⚙️ Calcola curva LAVANDERIA (totale)", key=f"c{consumer_id}_laundry_total_calc"):
            try:
                curves = []

                if w_present and w_inputs is not None:
                    w_curve = L.simulate(
                        "washer",
                        index=INDEX,
                        inputs=w_inputs,
                        temp=T_AIR_15,
                        seed=derive_seed(BASE_SEED, "consumer", consumer_id, "washer"),
                    ).rename("washer").astype(float)
                    curves.append(w_curve)

                if d_present and d_inputs is not None:
                    d_curve = L.simulate(
                        "dryer",
                        index=INDEX,
                        inputs=d_inputs,
                        temp=T_AIR_15,
                        seed=derive_seed(BASE_SEED, "consumer", consumer_id, "dryer"),
                    ).rename("dryer").astype(float)
                    curves.append(d_curve)

                if not curves:
                    st.info("Nessun dispositivo lavanderia attivo per questo consumatore.")
                    return None

                # SOMMA totale
                laundry_curve = curves[0]
                for c in curves[1:]:
                    laundry_curve = laundry_curve.add(c, fill_value=0.0)
                laundry_curve = laundry_curve.rename("laundry_total").astype(float)

                # cache CSV totale
                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer_id}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                laundry_curve.to_csv(cache_dir / "laundry_total.csv", index=True, index_label="timestamp")

                # plot (1 solo)
                laundry_plot = prepare_curve_for_plot(laundry_curve, curve_view_mode, curve_view_month)
                st.line_chart(laundry_plot)

                # KPI
                kwh_periodo, kwh_giorno, kwh_annuo = compute_kpis(laundry_curve, INDEX)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lavanderia: kWh periodo", f"{kwh_periodo:.1f}")
                c2.metric("Lavanderia: kWh/anno stimato", f"{kwh_annuo:.0f}")
                c3.metric("Lavanderia: media giornaliera", f"{kwh_giorno:.2f} kWh/g")

                # download CSV totale 15 min
                csv_buf = laundry_curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
                c4.download_button(
                    "⬇️ CSV 15 min (lavanderia totale – anno)",
                    data=csv_buf,
                    file_name=f"consumer_{consumer_id}_laundry_total_15min.csv",
                    mime="text/csv",
                    key=f"dl_{consumer_id}_laundry_total",
                )

            except Exception as e:
                st.error(f"Errore simulazione lavanderia totale: {e}")

        return laundry_curve

def climate_panel(consumer: dict, INDEX: pd.DatetimeIndex, T_AIR_15: pd.Series):
    """
    Renderizza la scheda Streamlit “Clima” per un singolo consumatore e (opzionalmente) esegue la simulazione.

    Parametri
    - consumer: dizionario dal registry ``consumers.json`` (modificato in-place).
    - INDEX: indice temporale della simulazione (UTC, tipicamente 15 minuti).
    - T_AIR_15: temperatura esterna interpolata su INDEX (°C).

    Side effects
    - Aggiorna ``consumer["devices"]["climate"]`` con i parametri impostati dall’utente.
    - Quando l’utente preme “Calcola curva clima”, salva:
      - ``cache/consumer_<id>/climate.csv``: curva aggregata elettrica in kW.
      - ``cache/consumer_<id>/climate_debug.csv``: diagnostica multi-colonna (se disponibile).

    Dipendenze (iniettate da ``pages/2_Consumatori.py``)
    La funzione usa variabili/global helper iniettati nel modulo, in particolare:
    ``SESSION_DIR``, ``consumers``, ``save_consumers_json``, ``curve_view_mode``, ``curve_view_month``.

    Failure modes
    - Errori nel core o in I/O vengono mostrati come messaggi Streamlit e non devono interrompere
      l’esecuzione dell’app.
    """
    dev_climate = get_climate_device(consumer)
    consumer_id = consumer.get("id")

    with st.expander("🌡️ Clima (riscaldamento + raffrescamento + ACS)", expanded=False):


        # =========================================================
        # UI BASE (leggera): Comfort + Abitazione
        # =========================================================
        st.markdown("### Base: Comfort e Abitazione")

        housing = dev_climate.setdefault("housing", {})
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            area_m2 = st.number_input("Superficie [m²]", min_value=20.0, max_value=400.0, value=float(housing.get("area_m2", 80.0)), step=5.0)
        with col2:
            insul = st.selectbox("Isolamento", ["scarso", "medio", "buono", "ottimo"],
                                 index=["scarso","medio","buono","ottimo"].index(str(housing.get("insulation_level","medio")).lower() if str(housing.get("insulation_level","medio")).lower() in ["scarso","medio","buono","ottimo"] else "medio"))
        with col3:
            mass_level = st.selectbox("Massa termica", ["light", "medium", "heavy"],
                                      index=["light","medium","heavy"].index(str(housing.get("thermal_mass_level","medium")).lower() if str(housing.get("thermal_mass_level","medium")).lower() in ["light","medium","heavy"] else "medium"))

        with st.expander("Opzioni abitazione (raramente necessarie)"):
            height_m = st.number_input("Altezza interna [m]", min_value=2.0, max_value=4.0, value=float(housing.get("ceiling_height_m", 2.7)), step=0.1)

        # Persistenza
        housing.update({
            "area_m2": float(area_m2),
            "insulation_level": str(insul),
            "thermal_mass_level": str(mass_level),
            "ceiling_height_m": float(height_m),
        })

        # Propaga ai campi legacy per retro-compatibilità e stime esistenti
        dev_climate.setdefault("heating", {})["floor_area_m2"] = float(area_m2)
        dev_climate.setdefault("heating", {})["insulation_level"] = str(insul)
        dev_climate.setdefault("thermal", {})["ceiling_height_m"] = float(height_m)
        dev_climate.setdefault("thermal", {})["thermal_mass_level"] = str(mass_level)

        st.markdown("### Base: Comfort, stagionalità e impianti")

        # --- Comfort / stagionalità ---
        heating = dev_climate.setdefault("heating", {})
        cooling = dev_climate.setdefault("cooling", {})

        colh1, colh2 = st.columns(2)
        with colh1:
            heating["t_set_heating_c"] = st.number_input(
                "Setpoint riscaldamento [°C]",
                min_value=16.0, max_value=24.0, step=0.5,
                value=float(heating.get("t_set_heating_c", 20.0)),
                key=f"c{consumer_id}_cl_base_tset_h",
            )
        with colh2:
            heating["heating_enable_temp_c"] = st.number_input(
                "Abilita riscaldamento se T esterna < [°C]",
                min_value=-5.0, max_value=25.0, step=0.5,
                value=float(heating.get("heating_enable_temp_c", 15.0)),
                key=f"c{consumer_id}_cl_base_enable_h",
            )

        colc1, colc2 = st.columns(2)
        with colc1:
            cooling["enabled"] = st.checkbox(
                "Raffrescamento attivo",
                value=bool(cooling.get("enabled", False)),
                key=f"c{consumer_id}_cl_base_cool_en",
            )
            cooling["t_set_cooling_c"] = st.number_input(
                "Setpoint raffrescamento [°C]",
                min_value=22.0, max_value=30.0, step=0.5,
                value=float(cooling.get("t_set_cooling_c", 26.0)),
                key=f"c{consumer_id}_cl_base_tset_c",
            )
        with colc2:
            cooling["cooling_enable_temp_c"] = st.number_input(
                "Abilita raffrescamento se T esterna > [°C]",
                min_value=15.0, max_value=35.0, step=0.5,
                value=float(cooling.get("cooling_enable_temp_c", 24.0)),
                key=f"c{consumer_id}_cl_base_enable_c",
            )

        st.markdown("#### Finestre di comfort")
        hw = _day_profile_editor(f"c{consumer_id}_cl_hw_base", "Riscaldamento (comfort)", heating)
        heating.update(hw)
        heating["comfort_windows"] = {}  # legacy (non usato in UI semplificata)

        cw = _day_profile_editor(f"c{consumer_id}_cl_cw_base", "Raffrescamento (comfort)", cooling)
        cooling.update(cw)
        cooling["comfort_windows"] = {}  # legacy (non usato in UI semplificata)

        st.markdown("#### Impianto (configurazione rapida)")
        st.caption("Scegli una tecnologia e una taglia. I dettagli (COP, priorità, ecc.) sono in Avanzato.")        # --- Quick sizing helper ---
        SIZE_MAP = {
            "Piccolo": 2.0,
            "Medio": 3.5,
            "Grande": 6.0,
        }

        def _infer_heat_tech_from_devices(dc: dict) -> str:
            if bool(dc.get("air_to_air_hp", {}).get("present", False)):
                return "PDC aria-aria (split)"
            if bool(dc.get("air_to_water_hp", {}).get("present", False)):
                return "PDC aria-acqua"
            if bool(dc.get("direct_heating", {}).get("present", False)):
                return "Resistenza elettrica"
            if bool(dc.get("floor_heating", {}).get("present", False)):
                return "Pavimento elettrico"
            return "Nessuno"

        def _infer_size_from_power_kw(p_kw: float) -> str:
            # mappa la potenza al bucket piu' vicino
            try:
                p = float(p_kw)
            except Exception:
                p = 0.0
            best = "Medio"
            best_err = 1e9
            for k, v in SIZE_MAP.items():
                err = abs(float(v) - p)
                if err < best_err:
                    best, best_err = k, err
            return best

        # UI state persistence: memorizza la scelta in dev_climate["_ui"] per ricaricarla a sessione successiva
        ui_state = dev_climate.setdefault("_ui", {})

        heat_tech_options = ["Nessuno", "PDC aria-aria (split)", "PDC aria-acqua", "Resistenza elettrica", "Pavimento elettrico"]
        heat_tech_default = str(ui_state.get("heat_tech") or "").strip()
        if heat_tech_default not in heat_tech_options:
            heat_tech_default = _infer_heat_tech_from_devices(dev_climate)

        # determina una taglia coerente con la potenza nominale attuale del device selezionato
        if heat_tech_default == "PDC aria-aria (split)":
            p0 = float(dev_climate.get("air_to_air_hp", {}).get("p_heat_nom_kw", SIZE_MAP["Medio"]) or SIZE_MAP["Medio"])
        elif heat_tech_default == "PDC aria-acqua":
            p0 = float(dev_climate.get("air_to_water_hp", {}).get("p_heat_nom_kw", SIZE_MAP["Grande"]) or SIZE_MAP["Grande"])
        elif heat_tech_default == "Resistenza elettrica":
            p0 = float(dev_climate.get("direct_heating", {}).get("p_el_nom_kw", 3.0) or 3.0)
        elif heat_tech_default == "Pavimento elettrico":
            p0 = float(dev_climate.get("floor_heating", {}).get("p_el_nom_kw", 3.0) or 3.0)
        else:
            p0 = float(SIZE_MAP["Medio"])

        heat_size_default = str(ui_state.get("heat_size") or "").strip()
        if heat_size_default not in list(SIZE_MAP.keys()):
            heat_size_default = _infer_size_from_power_kw(p0)

        heat_tech = st.selectbox(
            "Tecnologia principale riscaldamento",
            options=heat_tech_options,
            index=heat_tech_options.index(heat_tech_default),
            key=f"c{consumer_id}_cl_base_heat_tech",
        )
        heat_size = st.selectbox(
            "Taglia riscaldamento (potenza termica nominale)",
            options=list(SIZE_MAP.keys()),
            index=list(SIZE_MAP.keys()).index(heat_size_default),
            key=f"c{consumer_id}_cl_base_heat_size",
        )
        heat_cop = st.slider(
            "COP medio riscaldamento",
            min_value=2.0, max_value=5.0, step=0.1,
            value=float(ui_state.get("heat_cop", heating.get("_cop_mean", 3.2))),
            key=f"c{consumer_id}_cl_base_heat_cop",
        )
        heating["_cop_mean"] = float(heat_cop)

        # persisti selezioni UI (verranno salvate su consumers.json quando premi salva o calcola)
        ui_state["heat_tech"] = str(heat_tech)
        ui_state["heat_size"] = str(heat_size)
        ui_state["heat_cop"] = float(heat_cop)

        # reset presence
        dev_climate.setdefault("air_to_air_hp", {})["present"] = False
        dev_climate.setdefault("air_to_water_hp", {})["present"] = False
        dev_climate.setdefault("direct_heating", {})["present"] = False
        dev_climate.setdefault("floor_heating", {})["present"] = False

        p_th = float(SIZE_MAP[heat_size])
        if heat_tech == "PDC aria-aria (split)":
            hp = dev_climate.setdefault("air_to_air_hp", {})
            hp["present"] = True
            hp["p_heat_nom_kw"] = p_th
            hp["cop_at_7c"] = min(6.0, float(heat_cop) + 0.5)
            hp["cop_at_minus7c"] = max(1.5, float(heat_cop) - 0.7)
        elif heat_tech == "PDC aria-acqua":
            hp = dev_climate.setdefault("air_to_water_hp", {})
            hp["present"] = True
            hp["p_heat_nom_kw"] = p_th
            hp["cop_at_7c"] = min(5.5, float(heat_cop) + 0.4)
            hp["cop_at_minus7c"] = max(1.3, float(heat_cop) - 0.8)
        elif heat_tech == "Resistenza elettrica":
            dh = dev_climate.setdefault("direct_heating", {})
            dh["present"] = True
            dh["p_el_nom_kw"] = max(1.0, p_th / max(1.0, float(heat_cop)))  # approx
        elif heat_tech == "Pavimento elettrico":
            fh = dev_climate.setdefault("floor_heating", {})
            fh["present"] = True
            fh["p_el_nom_kw"] = max(1.0, p_th / max(1.0, float(heat_cop)))  # approx

        # Cooling quick config
        st.markdown("##### Raffrescamento (split)")
        ac_present = st.checkbox(
            "Split per raffrescamento presente",
            value=bool(dev_climate.get("air_to_air_ac", {}).get("present", False)),
            key=f"c{consumer_id}_cl_base_ac_present",
        )
        ac_size = st.selectbox(
            "Taglia raffrescamento (potenza frigorifera nominale)",
            options=list(SIZE_MAP.keys()),
            index=1,
            key=f"c{consumer_id}_cl_base_ac_size",
        )
        ac_eer = st.slider(
            "EER medio raffrescamento",
            min_value=2.0, max_value=5.0, step=0.1,
            value=float(cooling.get("_eer_mean", 3.2)),
            key=f"c{consumer_id}_cl_base_ac_eer",
        )
        cooling["_eer_mean"] = float(ac_eer)

        ac = dev_climate.setdefault("air_to_air_ac", {})
        ac["present"] = bool(ac_present)
        if ac_present:
            ac["p_cool_nom_kw"] = float(SIZE_MAP[ac_size])
            ac["eer_at_27c"] = min(6.0, float(ac_eer) + 0.5)
            ac["eer_at_35c"] = max(1.5, float(ac_eer) - 0.7)

        
        # =========================
        # Base: ACS (semplificata)
        # =========================
        st.markdown("#### Base: ACS (acqua calda sanitaria)")
        st.caption("Configurazione essenziale: persone, tecnologia e fasce mattina/sera. I parametri tecnici sono stimati dai default.")

        dhw_boiler = dev_climate.setdefault("dhw_electric_boiler", {})
        dhw_hp = dev_climate.setdefault("dhw_hp", {})

        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        with col_d1:
            dhw_enabled = st.checkbox(
                "ACS attiva",
                value=bool(dhw_boiler.get("present", True) or dhw_hp.get("present", False)),
                key=f"c{consumer_id}_cl_base_dhw_enabled",
            )
        with col_d2:
            dhw_people = st.number_input(
                "Persone",
                min_value=0, max_value=10, step=1,
                value=int(dhw_boiler.get("people", dhw_hp.get("people", 2)) or 0),
                key=f"c{consumer_id}_cl_base_dhw_people",
            )
        with col_d3:
            dhw_tech = st.selectbox(
                "Tecnologia ACS",
                options=["Boiler elettrico", "PDC ACS"],
                index=0 if bool(dhw_boiler.get("present", True)) else 1,
                key=f"c{consumer_id}_cl_base_dhw_tech",
            )

        # Fasce orarie (ore intere per compatibilità con struttura esistente)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            am_start = st.number_input("Mattina - inizio [h]", min_value=0, max_value=23, step=1,
                                       value=int((dhw_boiler.get("morning_window") or dhw_hp.get("morning_window") or [6, 9])[0]),
                                       key=f"c{consumer_id}_cl_base_dhw_am_s")
            am_end = st.number_input("Mattina - fine [h]", min_value=0, max_value=24, step=1,
                                     value=int((dhw_boiler.get("morning_window") or dhw_hp.get("morning_window") or [6, 9])[1]),
                                     key=f"c{consumer_id}_cl_base_dhw_am_e")
        with col_w2:
            pm_start = st.number_input("Sera - inizio [h]", min_value=0, max_value=23, step=1,
                                       value=int((dhw_boiler.get("evening_window") or dhw_hp.get("evening_window") or [18, 22])[0]),
                                       key=f"c{consumer_id}_cl_base_dhw_pm_s")
            pm_end = st.number_input("Sera - fine [h]", min_value=0, max_value=24, step=1,
                                     value=int((dhw_boiler.get("evening_window") or dhw_hp.get("evening_window") or [18, 22])[1]),
                                     key=f"c{consumer_id}_cl_base_dhw_pm_e")

        # Applica scelta in modo consistente
        if not dhw_enabled or dhw_people <= 0:
            dhw_boiler["present"] = False
            dhw_hp["present"] = False
        else:
            if dhw_tech == "PDC ACS":
                dhw_boiler["present"] = False
                dhw_hp["present"] = True
            else:
                dhw_boiler["present"] = True
                dhw_hp["present"] = False

        # Persisti campi minimi
        for target in (dhw_boiler, dhw_hp):
            target["people"] = int(dhw_people)
            target["morning_window"] = [int(am_start), int(am_end)]
            target["evening_window"] = [int(pm_start), int(pm_end)]

        st.divider()

        # Avanzato rimosso: i parametri tecnici vengono stimati automaticamente a partire dalla UI Base.
        ac1, ac2 = st.columns(2)
        if ac1.button("💾 Salva parametri clima", key=f"c{consumer_id}_climate_save"):
            save_consumers_json(SESSION_DIR, consumers)
            st.success("Parametri clima salvati.")

        climate_curve = None
        if ac2.button("⚙️ Calcola curva clima", key=f"c{consumer_id}_climate_calc"):
            try:
                climate_cfg = build_climate_config_from_device_dict(dev_climate)

                # IMPORTANT: il modello clima deve essere indipendente dagli altri carichi
                # elettrici del consumatore (baseload/cucina/lavanderia/occupancy). Quindi
                # NON passiamo internal_gains_kw_el derivati da curve elettriche cache.
                profiles_clima = CL.build_climate_profiles(INDEX, climate_cfg, T_AIR_15, internal_gains_kw_el=None)

                climate_curve = profiles_clima["aggregated"].rename("climate").astype(float)
                if float(climate_curve.abs().sum()) == 0.0:
                    st.warning("Curva clima pari a zero: verifica che sia presente almeno un impianto (riscaldamento/raffrescamento/ACS) e che le soglie stagionali e le fasce orarie abilitino il servizio nel periodo simulato.")

                # cache CSV 15 min
                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer_id}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                climate_curve.to_csv(
                    cache_dir / "climate.csv",
                    index=True,
                    index_label="timestamp",
                )

                # Salva anche un CSV di diagnostica completo (include dbg_* e temperature interne)
                try:
                    df_dbg = pd.DataFrame({k: v for k, v in profiles_clima.items()})
                    df_dbg.to_csv(cache_dir / "climate_debug.csv", index=True, index_label="timestamp")
                except Exception:
                    # diagnostica non bloccante
                    pass

                # visualizzazione come le altre schede: annuale / mensile
                if curve_view_mode == "annuale" or curve_view_month is None:
                    curve_plot = climate_curve.resample("H").mean()
                else:
                    year, month = curve_view_month
                    mask_month = (climate_curve.index.year == year) & (climate_curve.index.month == month)
                    curve_plot = climate_curve[mask_month]

                st.line_chart(curve_plot)

                # KPI sul periodo simulato (conversione kW -> kWh)
                if len(climate_curve) > 1:
                    dt_hours = (climate_curve.index[1] - climate_curve.index[0]).total_seconds() / 3600.0
                else:
                    dt_hours = 0.0

                # ----------------------
                # Diagnostica dettagliata
                # ----------------------
                with st.expander("🔎 Diagnostica clima (debug)", expanded=False):
                    try:
                        dbg = profiles_clima

                        stab = float(dbg.get("dbg_stability_coef", pd.Series([float("nan")])).iloc[0])
                        st.write(f"**Stabilità numerica 2R2C** (dt/(C_air·R_am)): {stab:.2f} (target < 2.0)")
                        if np.isfinite(stab) and stab >= 1.95:
                            st.warning("Stabilità al limite: in alcune configurazioni la domanda può collassare a 0. Se vedi curve anomale/zero, prova a ridurre il timestep oppure aumenta l'inerzia (massa) / riduci il coupling aria-massa.")

                        heat_allowed_h = float(dbg.get("dbg_heat_allowed", pd.Series(0.0, index=climate_curve.index)).sum() * dt_hours)
                        heating_on_h = float(dbg.get("dbg_heating_on", pd.Series(0.0, index=climate_curve.index)).sum() * dt_hours)
                        cool_allowed_h = float(dbg.get("dbg_cool_allowed", pd.Series(0.0, index=climate_curve.index)).sum() * dt_hours)
                        cooling_on_h = float(dbg.get("dbg_cooling_on", pd.Series(0.0, index=climate_curve.index)).sum() * dt_hours)

                        q_heat_req_max = float(dbg.get("dbg_q_heat_req_th", pd.Series(0.0, index=climate_curve.index)).max())
                        q_cool_req_max = float(dbg.get("dbg_q_cool_req_th", pd.Series(0.0, index=climate_curve.index)).max())

                        p_heat_max = float(dbg.get("space_heating_total", pd.Series(0.0, index=climate_curve.index)).max())
                        p_cool_max = float(dbg.get("space_cooling_total", pd.Series(0.0, index=climate_curve.index)).max())

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Heat allowed (h)", f"{heat_allowed_h:.0f}")
                        m2.metric("Heating ON (h)", f"{heating_on_h:.0f}")
                        m3.metric("Cool allowed (h)", f"{cool_allowed_h:.0f}")
                        m4.metric("Cooling ON (h)", f"{cooling_on_h:.0f}")

                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("max Q_heat_req (kWth)", f"{q_heat_req_max:.2f}")
                        d2.metric("max Q_cool_req (kWth)", f"{q_cool_req_max:.2f}")
                        d3.metric("max P_heating (kWel)", f"{p_heat_max:.2f}")
                        d4.metric("max P_cooling (kWel)", f"{p_cool_max:.2f}")

                        # Quick checks for the typical "all zero" cases
                        if heat_allowed_h > 0 and heating_on_h == 0:
                            st.error("Il riscaldamento è 'allowed' ma non va mai in ON: tipicamente setpoint troppo basso/alto rispetto alla dinamica, o problema di stabilità numerica.")
                        if heating_on_h > 0 and p_heat_max == 0:
                            st.error("Riscaldamento ON ma potenza elettrica = 0: nessun generatore presente o potenza nominale = 0.")
                        if cool_allowed_h > 0 and cooling_on_h == 0:
                            st.warning("Raffrescamento allowed ma non va mai in ON: verifica soglie, setpoint, e deadband.")

                        st.caption("Cache diagnostica salvata in: cache/consumer_<id>/climate_debug.csv")
                    except Exception as e:
                        st.warning(f"Diagnostica non disponibile: {e}")
                kwh_periodo = float(climate_curve.sum() * dt_hours)

                giorni = (INDEX.max() - INDEX.min()).days + 1
                kwh_giorno = kwh_periodo / max(giorni, 1)
                kwh_annuo = kwh_giorno * 365

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Clima: kWh periodo", f"{kwh_periodo:.1f}")
                c2.metric("Clima: kWh/anno stimato", f"{kwh_annuo:.0f}")
                c3.metric("Clima: media giornaliera", f"{kwh_giorno:.2f} kWh/g")

                csv_buf = climate_curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
                c4.download_button(
                    "⬇️ CSV 15 min (clima – anno)",
                    data=csv_buf,
                    file_name=f"consumer_{consumer_id}_climate_15min.csv",
                    mime="text/csv",
                    key=f"dl_{consumer_id}_climate",
                )

            except Exception as e:
                st.error(f"Errore nella simulazione clima: {e}")

    return climate_curve