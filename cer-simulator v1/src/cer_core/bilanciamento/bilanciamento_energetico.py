"""cer_core.bilanciamento.bilanciamento_energetico
===============================================

Modulo di dominio per il **bilanciamento energetico** di una Comunità Energetica
Rinnovabile (CER).

Il modulo implementa la catena di trasformazione dai dati di input (CSV di
consumo e produzione) fino agli output orari per singolo membro e per la CER.

Responsabilità principali
-------------------------
- Definizione dei modelli dati di simulazione: :class:`PeriodConfig`,
  :class:`MemberSpec`, :class:`ProductionSpec`, :class:`BatterySpec`.
- Parsing e validazione dei CSV di input:
  - consumi a 15 minuti (potenza in kW);
  - produzione oraria (potenza in kW, formato wide PVGIS-like).
- Calcolo dei flussi energetici:
  - per membro (autoconsumo, prelievo, immissione) con o senza BESS;
  - aggregazione a livello comunità (energia condivisa e bilanci CER).
- Persistenza su filesystem degli output in formato CSV (`save_outputs`).

Convenzioni e invarianti
------------------------
Unità
  - Potenza: **kW**
  - Energia: **kWh**
  - Integrazione temporale: 15 minuti = 0.25 h

Tempo e timezone
  - Tutti gli indici temporali sono `DatetimeIndex` timezone-aware.
  - I timestamp letti dai CSV vengono interpretati in base al parametro ``tz``:
    - se i timestamp sono *naive*, vengono localizzati in ``tz``;
    - se sono *tz-aware*, vengono convertiti in ``tz``.
  - Internamente, i dati vengono **normalizzati in UTC** per coerenza tra
    consumi e produzione e per interoperabilità con la UI Streamlit.

Frequenze attese
  - Consumi: 15 minuti.
  - Produzione: oraria.

Output per membro (orario)
  Le colonne standard prodotte da :func:`compute_member_energy_hourly` includono:
  ``E_load_kWh``, ``E_prod_kWh``, ``E_aut_kWh``, ``E_prel_kWh``, ``E_imm_kWh``.

Side effects
------------
L'unica funzione con effetti collaterali è :func:`save_outputs`, che scrive
su filesystem gli output di una run in una directory target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple, List

import logging
import math

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


# =========================
# 1) Data structures
# =========================

@dataclass(frozen=True)
class PeriodConfig:
    tz: str
    t0: pd.Timestamp            # inclusive
    t1: pd.Timestamp            # exclusive
    idx_hourly: pd.DatetimeIndex
    idx_15min: pd.DatetimeIndex

    @property
    def expected_hours(self) -> int:
        return len(self.idx_hourly)

    @property
    def expected_15min(self) -> int:
        return len(self.idx_15min)


@dataclass(frozen=True)
class ProductionSpec:
    enabled: bool
    mode: str                   # "totale" | "aree"
    selected_areas: Tuple[str, ...] = ()

@dataclass(frozen=True)
class BatterySpec:
    """Specifiche batteria (BESS) per replica logica RSE.

    capacity_kwh: capacità nominale (kWh) a inizio vita.
    dod: Depth of Discharge (0-1). Es. 0.8 => SOC minimo = 20%.
    roundtrip_eff: efficienza round-trip (0-1). Nel modello RSE si usa ε_halfcycle = sqrt(roundtrip_eff).
    derating_factor: perdita di capacità per ciclo equivalente (0-1). Se 0 => nessun degrado.
    init_soc_perc: SOC iniziale in percentuale della capacità massima disponibile (0-1).
    """
    capacity_kwh: float
    dod: float = 0.8
    roundtrip_eff: float = 0.9
    derating_factor: float = 0.0
    init_soc_perc: float = 0.2

@dataclass(frozen=True)
class MemberSpec:
    member_id: str
    name: str
    consumption_csv: Path
    production_csv: Optional[Path]   # None => produzione 0
    production_spec: ProductionSpec
    battery: Optional[BatterySpec] = None


# =========================
# 1b) CSV header helpers (per UI)
# =========================

def production_available_columns(csv_path: Path) -> List[str]:
    """Ritorna le colonne del CSV produzione (solo header)."""
    df0 = pd.read_csv(csv_path, nrows=0)
    return list(df0.columns)


def production_area_columns(csv_path: Path) -> List[str]:
    """Ritorna le colonne Area_* presenti nel CSV produzione."""
    cols = production_available_columns(csv_path)
    return [c for c in cols if str(c).lower().startswith("area_")]


# =========================
# 2) Public API (what Streamlit calls)
# =========================

def infer_period_from_consumption(consumption_csv: Path, tz: str = "UTC") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Inferisce il periodo di simulazione a partire dai consumi a 15 minuti.

    La funzione legge la sola colonna ``timestamp`` dal CSV consumi e determina
    l'intervallo [t0, t1) che copre tutti i campioni, arrotondato a giorno.

    Regole di calcolo:
    - `t0` è il floor a giorno (`.floor('D')`) del primo timestamp disponibile;
    - `t1` è il ceil a giorno del *timestamp massimo + 15 minuti* (per includere
      l'ultimo intervallo a 15 minuti).

    Convenzione timezone:
    - i timestamp nel CSV sono parsati con :func:`_parse_datetime_utc`, che
      interpreta i timestamp naive come locali in ``tz`` e normalizza in UTC.
    - l'output (`t0`, `t1`) è quindi in UTC (timezone-aware).

    Parameters
    ----------
    consumption_csv : pathlib.Path
        CSV consumi con colonna ``timestamp``.
    tz : str
        Timezone di riferimento per timestamp naive (es. "UTC", "Europe/Rome").

    Returns
    -------
    (pandas.Timestamp, pandas.Timestamp)
        (t0, t1) timezone-aware in UTC, con t0 incluso e t1 escluso.
    """
    df = pd.read_csv(consumption_csv, usecols=["timestamp"])
    ts_utc = _parse_datetime_utc(df["timestamp"], tz=tz)  # DatetimeIndex (UTC)

    t_min = pd.Timestamp(ts_utc.min())
    t_max = pd.Timestamp(ts_utc.max())

    t0 = t_min.floor("D")
    t1_raw = t_max + pd.Timedelta(minutes=15)
    t1 = t1_raw.ceil("D")

    return t0, t1


def build_period_config(t0: pd.Timestamp, t1: pd.Timestamp, tz: str = "UTC") -> PeriodConfig:
    """Costruisce la configurazione di periodo e le griglie temporali attese.

    La :class:`PeriodConfig` contiene due griglie di riferimento:
    - `idx_hourly`: indice orario (left-inclusive, right-exclusive);
    - `idx_15min`: indice a 15 minuti (left-inclusive, right-exclusive).

    Queste griglie definiscono la copertura attesa degli input e rappresentano
    la base per reindex/validazione in :func:`load_and_validate_member`.

    Parameters
    ----------
    t0, t1 : pandas.Timestamp
        Estremi del periodo con convenzione [t0, t1). Se naive, vengono localizzati
        in `tz`; se timezone-aware, vengono convertiti in `tz`.
    tz : str
        Timezone del periodo (tipicamente "UTC").

    Returns
    -------
    PeriodConfig
        Configurazione completa di periodo.

    Raises
    ------
    ValueError
        Se t0 non è strettamente precedente a t1.
    """
    t0 = _ensure_tz(t0, tz)
    t1 = _ensure_tz(t1, tz)

    if not (t0 < t1):
        raise ValueError("Periodo non valido: t0 deve essere < t1")

    # Indice orario (exclusive right)
    idx_hourly = pd.date_range(start=t0, end=t1, freq="h", inclusive="left", tz=tz)
    # Indice 15 min
    idx_15min = pd.date_range(start=t0, end=t1, freq="15min", inclusive="left", tz=tz)

    return PeriodConfig(tz=tz, t0=t0, t1=t1, idx_hourly=idx_hourly, idx_15min=idx_15min)


def load_and_validate_member(member: MemberSpec, period: PeriodConfig) -> Dict[str, pd.Series]:
    """Carica e valida i dati di un membro su una griglia temporale di riferimento.

    La funzione:
    1) legge e valida i consumi a 15 minuti (kW) dal CSV;
    2) legge e valida la produzione oraria (kW) dal CSV (se abilitata), altrimenti
       crea una serie nulla su `period.idx_hourly`;
    3) reindicizza e valida copertura completa su `period.idx_15min` e
       `period.idx_hourly` tramite :func:`validate_and_clip_series`.

    Parameters
    ----------
    member : MemberSpec
        Specifiche membro (path CSV + configurazione produzione + batteria).
    period : PeriodConfig
        Periodo e griglie temporali attese.

    Returns
    -------
    Dict[str, pandas.Series]
        Dizionario con chiavi:
        - ``P_load_15min_kW``: serie kW su `period.idx_15min`;
        - ``P_prod_hourly_kW``: serie kW su `period.idx_hourly`.

    Raises
    ------
    ValueError
        Se i dati non coprono il periodo, la frequenza non è regolare o sono
        presenti buchi interni.

    Notes
    -----
    La normalizzazione timezone dei timestamp è gestita dalle funzioni di parsing
    :func:`parse_consumption_15min_kw` e :func:`parse_production_hourly_kw`.
    """
    P_load_15 = parse_consumption_15min_kw(member.consumption_csv, tz=period.tz)
    P_load_15 = validate_and_clip_series(P_load_15, period.idx_15min, name=f"consumi:{member.member_id}")

    if member.production_csv is None or not member.production_spec.enabled:
        P_prod_h = pd.Series(0.0, index=period.idx_hourly, name="P_prod_kW")
    else:
        P_prod_h = parse_production_hourly_kw(
            member.production_csv,
            production_mode=member.production_spec.mode,
            selected_areas=list(member.production_spec.selected_areas),
            tz=period.tz,
        )
        P_prod_h = validate_and_clip_series(P_prod_h, period.idx_hourly, name=f"produzione:{member.member_id}")

    return {"P_load_15min_kW": P_load_15, "P_prod_hourly_kW": P_prod_h}


def expand_hourly_to_15min_kw(P_prod_hourly_kW: pd.Series, idx_15min: pd.DatetimeIndex) -> pd.Series:
    """Espande una serie oraria di potenza su una griglia a 15 minuti.

    Il valore orario viene replicato sui quattro intervalli da 15 minuti
    dell'ora (forward-fill). Esempio: il valore alle 10:00 viene copiato su
    10:00, 10:15, 10:30, 10:45.

    Parameters
    ----------
    P_prod_hourly_kW : pandas.Series
        Potenza oraria (kW) indicizzata per timestamp (timezone-aware).
    idx_15min : pandas.DatetimeIndex
        Griglia a 15 minuti (timezone-aware) su cui espandere.

    Returns
    -------
    pandas.Series
        Potenza a 15 minuti (kW) su `idx_15min`.

    Raises
    ------
    ValueError
        Se l'espansione produce NaN (tipicamente quando `idx_15min` inizia prima
        del primo timestamp orario o il periodo non è allineato).
    """
    if P_prod_hourly_kW.empty:
        return pd.Series(0.0, index=idx_15min, name="P_prod_15min_kW")

    P = P_prod_hourly_kW.sort_index()

    # Forward-fill su indice 15 min: replica il valore dell'ora sui quarti d'ora successivi
    P15 = P.reindex(idx_15min, method="ffill")

    # Se ci sono NaN, tipicamente significa che idx_15min inizia prima del primo timestamp orario
    if P15.isna().any():
        missing_examples = P15[P15.isna()].index[:5].astype(str).tolist()
        raise ValueError(
            "Produzione: impossibile espandere l'orario su 15min (NaN dopo reindex/ffill). "
            f"Esempi missing: {missing_examples}. "
            "Verifica che il periodo consumi e produzione coincidano e che t0 sia allineato all'ora."
        )

    P15.name = "P_prod_15min_kW"
    return P15


def _bess_step_rse(
    E_terminal_theor_kWh: float,
    SOCkWh_tm1: float,
    eps_roundtrip_halfcycle: float,
    battery_min_kwh: float,
    battery_max_kwh: float,
    flag_battery_to_grid: int = 0,
    battery_to_grid_capacity_kwh: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """Step BESS identico alla funzione BESS del simulatore RSE (CACER).

    Sign convention:
      - E_terminal_theor_kWh > 0: carica (energia in ingresso alla batteria)
      - E_terminal_theor_kWh < 0: scarica (energia erogata dalla batteria al sistema)

    Ritorna:
      E_terminal_real_kWh, E_loss_kWh, E_discharge_real_net_kWh, SOCkWh, SOCperc

    Note
    ----
    `E_discharge_real_net_kWh` è **negativa** in fase di scarica (sign convention
    coerente con il codice originale RSE); il chiamante gestisce il segno.
    """
    E_charge_theor_gross = max(0.0, E_terminal_theor_kWh)
    E_discharge_theor_gross = min(
        0.0,
        E_terminal_theor_kWh - battery_to_grid_capacity_kwh * float(flag_battery_to_grid),
    )

    E_halfcycle_theor = (
        E_charge_theor_gross * eps_roundtrip_halfcycle
        + E_discharge_theor_gross / eps_roundtrip_halfcycle
    )

    # Update SOC con limiti min/max
    SOC_candidate = SOCkWh_tm1 + E_halfcycle_theor
    if SOC_candidate > battery_max_kwh:
        SOCkWh = battery_max_kwh
    elif SOC_candidate <= battery_min_kwh:
        SOCkWh = battery_min_kwh
    else:
        SOCkWh = SOC_candidate

    SOCperc = (SOCkWh / battery_max_kwh) if battery_max_kwh > 0 else 0.0

    E_halfcycle_real = SOCkWh - SOCkWh_tm1
    E_charge_real_net = max(0.0, E_halfcycle_real)
    E_charge_real_brut = E_charge_real_net / eps_roundtrip_halfcycle
    E_discharge_real_brut = min(0.0, E_halfcycle_real)
    E_discharge_real_net = E_discharge_real_brut * eps_roundtrip_halfcycle

    E_terminal_real = E_charge_real_brut + E_discharge_real_net
    E_loss = abs(E_charge_real_net - E_charge_real_brut) + abs(E_discharge_real_brut - E_discharge_real_net)

    return E_terminal_real, E_loss, E_discharge_real_net, SOCkWh, SOCperc


def compute_member_energy_hourly(
    P_load_15min_kW: pd.Series,
    P_prod_hourly_kW: pd.Series,
    battery: Optional[BatterySpec] = None,
) -> pd.DataFrame:
    """Calcola i flussi energetici orari di un membro (kWh) da potenze input.

    La funzione usa una discretizzazione a 15 minuti come timebase di calcolo:

    - i consumi sono già su 15 minuti (kW);
    - la produzione oraria (kW) viene espansa a 15 minuti assumendo potenza
      costante nell'ora (:func:`expand_hourly_to_15min_kw`);
    - se una batteria è configurata, il bilancio viene calcolato su ogni step
      a 15 minuti replicando la logica RSE (:func:`compute_member_energy_15min_rse_with_battery`);
    - i risultati vengono poi aggregati a 1 ora mediante somma delle energie.

    Parameters
    ----------
    P_load_15min_kW : pandas.Series
        Potenza di carico a 15 minuti (kW), indicizzata (timezone-aware).
    P_prod_hourly_kW : pandas.Series
        Potenza di produzione oraria (kW), indicizzata (timezone-aware).
        Nella pipeline standard viene già reindicizzata su `PeriodConfig.idx_hourly`.
    battery : Optional[BatterySpec]
        Specifiche BESS. Se None o con `capacity_kwh <= 0`, la batteria non è applicata.

    Returns
    -------
    pandas.DataFrame
        DataFrame orario (index = `P_prod_hourly_kW.index`) con colonne energetiche (kWh):

        - Sempre presenti: ``E_load_kWh``, ``E_prod_kWh``, ``E_aut_kWh``, ``E_imm_kWh``, ``E_prel_kWh``
        - Sempre presenti (dettaglio): ``E_aut_PV_kWh``, ``E_aut_batt_kWh``
        - Se batteria attiva: ``E_batt_charge_kWh``, ``E_batt_discharge_kWh``, ``E_batt_loss_kWh``,
          ``E_batt_terminal_kWh``, ``SOC_kWh``, ``SOC_perc``.

    Raises
    ------
    ValueError
        Se dopo l'aggregazione oraria compaiono NaN nelle colonne energetiche
        (tipicamente per mismatch di copertura tra consumi e produzione).

    Notes
    -----
    - L'indice di output è derivato dalla serie di produzione oraria. Nella
      pipeline UI, `P_prod_hourly_kW` è sempre definita sulla griglia di periodo.
    - Le convenzioni di segno della batteria seguono la logica RSE; in output le
      colonne ``E_batt_charge_kWh`` e ``E_batt_discharge_kWh`` sono non-negative,
      mentre ``E_batt_terminal_kWh`` è positiva in carica e negativa in scarica.
    """
    idx_hourly = P_prod_hourly_kW.index.sort_values()

    if battery is not None and float(battery.capacity_kwh) > 0:
        df15 = compute_member_energy_15min_rse_with_battery(
            P_load_15min_kW=P_load_15min_kW,
            P_prod_hourly_kW=P_prod_hourly_kW,
            battery=battery,
        )

        # Flussi principali (somma su ora)
        E_load_h = df15["E_load_kWh"].resample("h").sum()
        E_prod_h = df15["E_prod_kWh"].resample("h").sum()
        E_aut_h  = df15["E_aut_kWh"].resample("h").sum()
        E_imm_h  = df15["E_imm_kWh"].resample("h").sum()
        E_prel_h = df15["E_prel_kWh"].resample("h").sum()

        # Dettaglio autoconsumo (somma su ora)
        E_aut_pv_h = df15["E_aut_PV_kWh"].resample("h").sum()
        E_aut_batt_h = df15["E_aut_batt_kWh"].resample("h").sum()

        # Batteria (energia su ora + SOC a fine ora)
        E_batt_terminal_h = df15["E_terminal_real_kWh"].resample("h").sum()
        E_batt_loss_h = df15["E_loss_kWh"].resample("h").sum()

        E_batt_charge_h = df15["E_terminal_real_kWh"].clip(lower=0).resample("h").sum()
        E_batt_discharge_h = (-df15["E_terminal_real_kWh"].clip(upper=0)).resample("h").sum()

        SOC_kWh_h = df15["SOC_kWh"].resample("h").last()
        SOC_perc_h = df15["SOC_perc"].resample("h").last()

    else:
        # --- Senza batteria ---
        idx_15min = P_load_15min_kW.index
        P_prod_15min_kW = expand_hourly_to_15min_kw(P_prod_hourly_kW, idx_15min)

        E_load_step = (P_load_15min_kW * 0.25).astype(float)
        E_prod_step = (P_prod_15min_kW * 0.25).astype(float)

        E_aut_step = pd.concat([E_load_step, E_prod_step], axis=1).min(axis=1)
        E_imm_step = E_prod_step - E_aut_step
        E_prel_step = E_load_step - E_aut_step

        E_load_h = E_load_step.resample("h").sum()
        E_prod_h = E_prod_step.resample("h").sum()
        E_aut_h  = E_aut_step.resample("h").sum()
        E_imm_h  = E_imm_step.resample("h").sum()
        E_prel_h = E_prel_step.resample("h").sum()

        # Dettaglio coerente (PV-only)
        E_aut_pv_h = E_aut_h.copy()
        E_aut_batt_h = pd.Series(0.0, index=E_aut_h.index)

        E_batt_terminal_h = pd.Series(0.0, index=E_aut_h.index)
        E_batt_loss_h = pd.Series(0.0, index=E_aut_h.index)
        E_batt_charge_h = pd.Series(0.0, index=E_aut_h.index)
        E_batt_discharge_h = pd.Series(0.0, index=E_aut_h.index)
        SOC_kWh_h = pd.Series(np.nan, index=E_aut_h.index)
        SOC_perc_h = pd.Series(np.nan, index=E_aut_h.index)

    out = pd.DataFrame(
        {
            "E_load_kWh": E_load_h.reindex(idx_hourly),
            "E_prod_kWh": E_prod_h.reindex(idx_hourly),
            "E_aut_kWh":  E_aut_h.reindex(idx_hourly),
            "E_imm_kWh":  E_imm_h.reindex(idx_hourly),
            "E_prel_kWh": E_prel_h.reindex(idx_hourly),

            # dettaglio
            "E_aut_PV_kWh": E_aut_pv_h.reindex(idx_hourly),
            "E_aut_batt_kWh": E_aut_batt_h.reindex(idx_hourly),

            # batteria
            "E_batt_charge_kWh": E_batt_charge_h.reindex(idx_hourly),
            "E_batt_discharge_kWh": E_batt_discharge_h.reindex(idx_hourly),
            "E_batt_loss_kWh": E_batt_loss_h.reindex(idx_hourly),
            "E_batt_terminal_kWh": E_batt_terminal_h.reindex(idx_hourly),
            "SOC_kWh": SOC_kWh_h.reindex(idx_hourly),
            "SOC_perc": SOC_perc_h.reindex(idx_hourly),
        },
        index=idx_hourly,
    )

    # Validazioni: solo colonne "energetiche" devono essere complete
    mandatory = ["E_load_kWh", "E_prod_kWh", "E_aut_kWh", "E_imm_kWh", "E_prel_kWh"]
    if out[mandatory].isna().any().any():
        nan_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(
            f"Aggregazione a ora: NaN dopo reindex su idx_hourly. Colonne con NaN: {nan_cols}. "
            "Verifica che consumi (15min) e produzione (hourly) coprano lo stesso periodo."
        )

    # Non negatività per flussi che devono essere >= 0
    nonneg_cols = [
        "E_aut_kWh", "E_imm_kWh", "E_prel_kWh",
        "E_aut_PV_kWh", "E_aut_batt_kWh",
        "E_batt_charge_kWh", "E_batt_discharge_kWh", "E_batt_loss_kWh",
    ]
    for c in nonneg_cols:
        if c in out.columns and (out[c] < -1e-9).any():
            raise ValueError(f"Flussi negativi in colonna {c}: controllo unità/parse fallito.")

    return out


def compute_member_energy_15min_rse_with_battery(
    P_load_15min_kW: pd.Series,
    P_prod_hourly_kW: pd.Series,
    battery: BatterySpec,
) -> pd.DataFrame:
    """Calcola i flussi energetici a 15 minuti con batteria (logica RSE/CACER).

    La funzione replica la logica implementata nel simulatore RSE per il comportamento
    del BESS su passo a 15 minuti. Le potenze in ingresso vengono convertite in energia
    (kWh) per step e viene calcolata la gestione del SOC con limiti min/max e degradazione.

    Convenzioni di segno (coerenti con il codice originale RSE):
    - ``E_terminal_theor_kWh`` > 0: energia teorica verso la batteria (carica)
    - ``E_terminal_theor_kWh`` < 0: energia teorica dalla batteria (scarica)
    - ``E_discharge_real_net_kWh`` restituita da :func:`_bess_step_rse` è **negativa** in scarica.

    Parameters
    ----------
    P_load_15min_kW : pandas.Series
        Potenza di carico a 15 minuti (kW) indicizzata.
    P_prod_hourly_kW : pandas.Series
        Potenza di produzione oraria (kW) indicizzata.
    battery : BatterySpec
        Parametri del BESS (capacità, DoD, efficienza, degrado, SOC iniziale).

    Returns
    -------
    pandas.DataFrame
        DataFrame indicizzato su `idx_15min = P_load_15min_kW.index` con colonne
        in kWh per step:

        - ``E_load_kWh``, ``E_prod_kWh``
        - ``E_aut_PV_kWh``, ``E_aut_batt_kWh``, ``E_aut_kWh``
        - ``E_terminal_real_kWh``, ``E_loss_kWh``
        - ``SOC_kWh``, ``SOC_perc``
        - ``E_prel_kWh``, ``E_imm_kWh``

    Raises
    ------
    ValueError
        Se i parametri della batteria non rispettano i range ammessi o se l'espansione
        della produzione oraria su 15 minuti fallisce.
    """
    idx_15min = P_load_15min_kW.index
    P_prod_15min_kW = expand_hourly_to_15min_kw(P_prod_hourly_kW, idx_15min)

    # kW -> kWh per step 15 minuti
    Eut = (P_load_15min_kW * 0.25).astype(float)
    Eprod = (P_prod_15min_kW * 0.25).astype(float)

    # Parametri batteria (RSE)
    capacity = float(battery.capacity_kwh)
    dod = float(battery.dod)
    roundtrip_eff = float(battery.roundtrip_eff)
    derating_factor = float(battery.derating_factor)
    init_soc_perc = float(battery.init_soc_perc)

    if capacity <= 0:
        raise ValueError("BatterySpec.capacity_kwh deve essere > 0")
    if not (0 < dod <= 1):
        raise ValueError("BatterySpec.dod deve essere in (0, 1].")
    if not (0 < roundtrip_eff <= 1):
        raise ValueError("BatterySpec.roundtrip_eff deve essere in (0, 1].")
    if not (0 <= derating_factor < 1):
        raise ValueError("BatterySpec.derating_factor deve essere in [0, 1).")
    if not (0 <= init_soc_perc <= 1):
        raise ValueError("BatterySpec.init_soc_perc deve essere in [0, 1].")

    eps_half = math.sqrt(roundtrip_eff)

    # Stato iniziale (coerente con commento RSE: start at 20% SOC)
    SOC_tm1 = init_soc_perc * capacity
    battery_cum_charge = 0.0

    n = len(idx_15min)

    E_terminal_real = np.zeros(n, dtype=float)
    E_loss = np.zeros(n, dtype=float)
    E_discharge_real_net = np.zeros(n, dtype=float)
    SOC_kWh = np.zeros(n, dtype=float)
    SOC_perc = np.zeros(n, dtype=float)

    E_aut_PV = np.zeros(n, dtype=float)
    E_aut_batt = np.zeros(n, dtype=float)
    E_aut = np.zeros(n, dtype=float)
    E_prel = np.zeros(n, dtype=float)
    E_imm = np.zeros(n, dtype=float)

    Eut_values = Eut.to_numpy(dtype=float)
    Eprod_values = Eprod.to_numpy(dtype=float)

    for i in range(n):
        # Degrado capacità (RSE)
        cycles_number = battery_cum_charge / (capacity * dod) if (capacity * dod) > 0 else 0.0
        derating_index = (1.0 - derating_factor) ** cycles_number if derating_factor > 0 else 1.0
        batt_max = derating_index * capacity
        batt_min = batt_max * (1.0 - dod)

        E_terminal_theor = Eprod_values[i] - Eut_values[i]

        et_real, eloss, edis_net, soc, socp = _bess_step_rse(
            E_terminal_theor_kWh=E_terminal_theor,
            SOCkWh_tm1=SOC_tm1,
            eps_roundtrip_halfcycle=eps_half,
            battery_min_kwh=batt_min,
            battery_max_kwh=batt_max,
            flag_battery_to_grid=0,
            battery_to_grid_capacity_kwh=0.0,
        )

        # Update cumulative charge (RSE: solo quando carica ai morsetti > 0)
        if et_real > 0:
            battery_cum_charge += (soc - SOC_tm1)

        SOC_tm1 = soc

        E_terminal_real[i] = et_real
        E_loss[i] = eloss
        E_discharge_real_net[i] = edis_net
        SOC_kWh[i] = soc
        SOC_perc[i] = socp

        # Bilancio energia RSE
        E_aut_PV[i] = min(Eprod_values[i], Eut_values[i])
        E_aut_batt[i] = min(-edis_net, Eut_values[i] - E_aut_PV[i])
        E_aut[i] = E_aut_PV[i] + E_aut_batt[i]

        interscambio = Eprod_values[i] - Eut_values[i] - et_real
        E_prel[i] = -min(0.0, interscambio)
        E_imm[i] = max(0.0, interscambio)

    out = pd.DataFrame(
        {
            "E_load_kWh": Eut_values,
            "E_prod_kWh": Eprod_values,
            "E_aut_PV_kWh": E_aut_PV,
            "E_aut_batt_kWh": E_aut_batt,
            "E_aut_kWh": E_aut,
            "E_terminal_real_kWh": E_terminal_real,
            "E_loss_kWh": E_loss,
            "SOC_kWh": SOC_kWh,
            "SOC_perc": SOC_perc,
            "E_prel_kWh": E_prel,
            "E_imm_kWh": E_imm,
        },
        index=idx_15min,
    )

    return out


def compute_cer_hourly(members_hourly: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggrega i bilanci orari dei membri e calcola il bilancio CER.

    L'aggregazione avviene per somma delle immissioni e dei prelievi orari dei
    singoli membri, e definizione dell'energia condivisa come:

    ``E_cond = min(E_imm_CER, E_prel_CER)`` (per ogni ora).

    Parameters
    ----------
    members_hourly : Dict[str, pandas.DataFrame]
        Mappa ``member_id -> DataFrame`` con indice orario e colonne almeno:
        ``E_imm_kWh`` e ``E_prel_kWh``.

    Returns
    -------
    pandas.DataFrame
        DataFrame orario con colonne:
        ``E_imm_CER_kWh``, ``E_prel_CER_kWh``, ``E_cond_kWh``, ``E_export_kWh``, ``E_import_kWh``.

    Raises
    ------
    ValueError
        Se `members_hourly` è vuoto.
    """

    if not members_hourly:
        raise ValueError("members_hourly is empty: impossibile calcolare il bilancio CER")
    # Somme comunitarie
    E_imm_CER = sum(df["E_imm_kWh"] for df in members_hourly.values())
    E_prel_CER = sum(df["E_prel_kWh"] for df in members_hourly.values())

    E_cond = pd.concat([E_imm_CER, E_prel_CER], axis=1).min(axis=1)
    E_export = E_imm_CER - E_cond
    E_import = E_prel_CER - E_cond

    out = pd.DataFrame(
        {
            "E_imm_CER_kWh": E_imm_CER,
            "E_prel_CER_kWh": E_prel_CER,
            "E_cond_kWh": E_cond,
            "E_export_kWh": E_export,
            "E_import_kWh": E_import,
        }
    )
    return out


def summarize_member(member_hourly: pd.DataFrame) -> Dict[str, float]:
    """Calcola KPI energetici aggregati di un singolo membro sul periodo.

    KPI principali:
    - energia totale di carico, produzione, autoconsumo, immissione, prelievo (kWh);
    - `self_consumption = E_aut_tot / E_prod_tot` (se produzione > 0);
    - `self_sufficiency = E_aut_tot / E_load_tot` (se carico > 0).

    Parameters
    ----------
    member_hourly : pandas.DataFrame
        Output orario del membro prodotto da :func:`compute_member_energy_hourly`.

    Returns
    -------
    Dict[str, float]
        Dizionario KPI (kWh e indici adimensionali).
    """
    load_tot = float(member_hourly["E_load_kWh"].sum())
    prod_tot = float(member_hourly["E_prod_kWh"].sum())
    aut_tot  = float(member_hourly["E_aut_kWh"].sum())
    imm_tot  = float(member_hourly["E_imm_kWh"].sum())
    prel_tot = float(member_hourly["E_prel_kWh"].sum())

    sc = (aut_tot / prod_tot) if prod_tot > 0 else float("nan")
    ss = (aut_tot / load_tot) if load_tot > 0 else float("nan")

    return {
        "E_load_tot_kWh": load_tot,
        "E_prod_tot_kWh": prod_tot,
        "E_aut_tot_kWh": aut_tot,
        "E_imm_tot_kWh": imm_tot,
        "E_prel_tot_kWh": prel_tot,
        "self_consumption": sc,
        "self_sufficiency": ss,
    }


def summarize_cer(cer_hourly: pd.DataFrame, members_hourly: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Calcola KPI aggregati a livello CER.

    KPI principali:
    - energia condivisa totale, export e import (kWh);
    - `share_on_production = E_cond_tot / E_prod_tot`;
    - `coverage_on_load = E_cond_tot / E_load_tot`.

    Parameters
    ----------
    cer_hourly : pandas.DataFrame
        Output orario della CER prodotto da :func:`compute_cer_hourly`.
    members_hourly : Dict[str, pandas.DataFrame]
        Mappa membri -> output orari (per calcolo totali di load/prod).

    Returns
    -------
    Dict[str, float]
        Dizionario KPI (kWh e indici adimensionali).
    """
    cond_tot = float(cer_hourly["E_cond_kWh"].sum())
    export_tot = float(cer_hourly["E_export_kWh"].sum())
    import_tot = float(cer_hourly["E_import_kWh"].sum())

    prod_tot = float(sum(df["E_prod_kWh"].sum() for df in members_hourly.values()))
    load_tot = float(sum(df["E_load_kWh"].sum() for df in members_hourly.values()))

    share_on_prod = (cond_tot / prod_tot) if prod_tot > 0 else float("nan")
    cover_on_load = (cond_tot / load_tot) if load_tot > 0 else float("nan")

    return {
        "E_cond_tot_kWh": cond_tot,
        "E_export_tot_kWh": export_tot,
        "E_import_tot_kWh": import_tot,
        "share_on_production": share_on_prod,
        "coverage_on_load": cover_on_load,
        "E_prod_tot_kWh": prod_tot,
        "E_load_tot_kWh": load_tot,
    }


def save_outputs(
    out_dir: Path,
    period: PeriodConfig,
    members_hourly: Dict[str, pd.DataFrame],
    cer_hourly: pd.DataFrame,
    members_summary: pd.DataFrame,
    cer_summary: pd.DataFrame,
) -> Dict[str, Path]:
    """Scrive su filesystem gli output energetici di una run.

    Vengono prodotti file CSV con indici timezone-aware (serializzati da pandas):

    - `cer_hourly.csv`: bilancio orario comunità;
    - `members_hourly_long.csv`: bilancio orario membri in formato *long*, con colonna `member_id`;
    - `members_summary.csv`: KPI per membro (una riga per membro);
    - `cer_summary.csv`: KPI aggregati CER;
    - `period_meta.csv`: metadata del periodo e cardinalità attese.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory target (creata se mancante).
    period : PeriodConfig
        Periodo di riferimento (tz, t0, t1, dimensioni attese).
    members_hourly : Dict[str, pandas.DataFrame]
        Output orari per membro.
    cer_hourly : pandas.DataFrame
        Output orario della CER.
    members_summary : pandas.DataFrame
        Tabella KPI membri.
    cer_summary : pandas.DataFrame
        Tabella KPI CER.

    Returns
    -------
    Dict[str, pathlib.Path]
        Mappa nome-logico -> path del file scritto.

    Side Effects
    ------------
    Scrittura di più file CSV su filesystem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CER hourly
    p_cer = out_dir / "cer_hourly.csv"
    cer_hourly.to_csv(p_cer, index=True)

    # 2) Members hourly in formato long
    rows = []
    for mid, df in members_hourly.items():
        tmp = df.copy()
        tmp.insert(0, "member_id", mid)
        rows.append(tmp)
    df_members_long = pd.concat(rows, axis=0)
    p_members = out_dir / "members_hourly_long.csv"
    df_members_long.to_csv(p_members, index=True)

    # 3) Summary
    p_msum = out_dir / "members_summary.csv"
    members_summary.to_csv(p_msum, index=False)

    p_csum = out_dir / "cer_summary.csv"
    cer_summary.to_csv(p_csum, index=False)

    # 4) Metadata periodo
    p_meta = out_dir / "period_meta.csv"
    pd.DataFrame([{
        "tz": period.tz,
        "t0": str(period.t0),
        "t1": str(period.t1),
        "expected_hours": period.expected_hours,
        "expected_15min": period.expected_15min,
    }]).to_csv(p_meta, index=False)

    return {
        "cer_hourly": p_cer,
        "members_hourly_long": p_members,
        "members_summary": p_msum,
        "cer_summary": p_csum,
        "period_meta": p_meta,
    }


# =========================
# 3) Parsers (CSV formats)
# =========================

def parse_consumption_15min_kw(csv_path: Path, tz: str = "UTC") -> pd.Series:
    """Parsa i consumi a 15 minuti da CSV (potenza in kW).

    Formato atteso (CSV):
    - colonna ``timestamp``: timestamp di inizio intervallo (15 min);
    - colonna ``total_load``: potenza media nell'intervallo, in kW.

    Il parsing del timestamp usa :func:`_parse_datetime_utc`:
    - timestamp naive -> localizzati in ``tz``;
    - timestamp tz-aware -> convertiti in ``tz``;
    - risultato normalizzato in UTC.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path al CSV consumi.
    tz : str
        Timezone di riferimento per timestamp naive.

    Returns
    -------
    pandas.Series
        Serie ``P_load_kW`` indicizzata da DatetimeIndex UTC.

    Raises
    ------
    ValueError
        Se i timestamp non sono monotoni, presentano duplicati o non rispettano
        la frequenza attesa (15 minuti).
    """
    df = pd.read_csv(csv_path)
    ts = _parse_datetime_utc(df["timestamp"], tz=tz)
    s = pd.Series(df["total_load"].astype(float).values, index=ts, name="P_load_kW")
    s = s.sort_index()
    _validate_monotonic_no_dupes(s, name="consumption")
    _validate_regular_frequency(s.index, expected="15min", name="consumption")
    return s


def parse_production_hourly_kw(
    csv_path: Path,
    production_mode: str,
    selected_areas: Sequence[str],
    tz: str = "UTC"
) -> pd.Series:
    """Parsa la produzione oraria da CSV (potenza in kW).

    Formato atteso (CSV "wide" PVGIS-like):
    - colonna temporale: ``time`` (timestamp di inizio ora);
    - colonne potenza (kW):
      - ``Totale`` (produzione complessiva)
      - una o più colonne ``Area_*`` (produzione per sotto-area).

    La selezione della serie dipende da `production_mode`:
    - ``totale``: usa la colonna ``Totale``;
    - ``aree``: somma le colonne indicate in `selected_areas`.

    Il parsing del timestamp usa :func:`_parse_datetime_utc` e normalizza in UTC.
    La funzione valida monotonicità, assenza duplicati e frequenza regolare oraria.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path al CSV produzione.
    production_mode : str
        Modalità di produzione: "totale" oppure "aree".
    selected_areas : Sequence[str]
        Colonne da sommare se `production_mode == "aree"`.
    tz : str
        Timezone di riferimento per timestamp naive.

    Returns
    -------
    pandas.Series
        Serie ``P_prod_kW`` indicizzata da DatetimeIndex UTC.

    Raises
    ------
    ValueError
        Se mancano le colonne richieste o la frequenza temporale non è regolare.
    """
    df = pd.read_csv(csv_path)
    ts = _parse_datetime_utc(df["time"], tz=tz)
    df = df.set_index(ts).sort_index()

    _validate_monotonic_no_dupes(df.index.to_series(), name="production_index")
    _validate_regular_frequency(df.index, expected="h", name="production")

    if production_mode.lower() == "totale":
        if "Totale" not in df.columns:
            raise ValueError("Produzione: colonna 'Totale' non trovata.")
        s = df["Totale"].astype(float)
    elif production_mode.lower() == "aree":
        if not selected_areas:
            raise ValueError("Produzione: modalità 'aree' richiede almeno una area selezionata.")
        missing = [c for c in selected_areas if c not in df.columns]
        if missing:
            raise ValueError(f"Produzione: aree mancanti nel CSV: {missing}")
        s = df[list(selected_areas)].astype(float).sum(axis=1)
    else:
        raise ValueError("Produzione: production_mode deve essere 'totale' o 'aree'.")

    s.name = "P_prod_kW"
    return s


# =========================
# 4) Validation utilities
# =========================

def validate_and_clip_series(
    s: pd.Series,
    expected_index: pd.DatetimeIndex,
    name: str,
    *,
    allow_boundary_fill: bool = True,
    max_boundary_missing: int = 4,
    fill_method: str = "ffill",
) -> pd.Series:
    """
    Ritaglia su expected_index e verifica copertura completa (nessun buco).

    Per robustezza nell'import da CSV, è possibile colmare piccoli buchi *solo ai bordi*
    (inizio/fine periodo) fino a `max_boundary_missing` campioni.

    - Buchi in testa: riempi con il primo valore disponibile (backfill del bordo)
    - Buchi in coda: riempi con l'ultimo valore disponibile (forward-fill del bordo) oppure 0 con fill_method="zero"

    I buchi interni al periodo restano un errore.
    """
    s_clip = s.reindex(expected_index)

    if not s_clip.isna().any():
        return s_clip

    na = s_clip.isna()
    n_missing = int(na.sum())

    if allow_boundary_fill and n_missing <= max_boundary_missing:
        first_valid = s_clip.first_valid_index()
        last_valid = s_clip.last_valid_index()

        if first_valid is None or last_valid is None:
            missing_examples = s_clip.index[:5].astype(str).tolist()
            raise ValueError(
                f"{name}: dati assenti nel periodo (tutti NaN). Esempi attesi: {missing_examples}"
            )

        pos_first = s_clip.index.get_loc(first_valid)
        pos_last = s_clip.index.get_loc(last_valid)

        # verifica che i NaN siano SOLO ai bordi (nessun buco interno)
        internal_na = na.iloc[pos_first:pos_last + 1].any()
        if not internal_na:
            # NaN in testa
            if pos_first > 0:
                s_clip.iloc[:pos_first] = s_clip.iloc[pos_first]
            # NaN in coda
            if pos_last < len(s_clip) - 1:
                if fill_method == "zero":
                    s_clip.iloc[pos_last + 1 :] = 0.0
                else:
                    s_clip.iloc[pos_last + 1 :] = s_clip.iloc[pos_last]

            # warning soft
            try:
                missing_idx = expected_index[na]
                examples = missing_idx[:5].astype(str).tolist()
                logger.warning(
                    "%s: colmati %s campioni mancanti ai bordi del periodo. Esempi: %s",
                    name,
                    n_missing,
                    examples,
                )
            except Exception:
                pass
            return s_clip

    missing_examples = s_clip[na].index[:5].astype(str).tolist()
    try:
        src_min = str(s.index.min())
        src_max = str(s.index.max())
    except Exception:
        src_min = src_max = "n/a"
    exp_min = str(expected_index.min())
    exp_max = str(expected_index.max())
    raise ValueError(
        f"{name}: dati incompleti nel periodo. Missing={n_missing}. "
        f"Esempi: {missing_examples}. Atteso=[{exp_min} .. {exp_max}], File=[{src_min} .. {src_max}]"
    )

def _parse_datetime_utc(series, tz: str) -> pd.DatetimeIndex:
    """Parse timestamp da CSV e normalizza in UTC.

    Policy di parsing (coerente con i casi reali di import da CSV):

    - Se il timestamp è *naive* (senza timezone), viene interpretato come ora locale
      nel timezone ``tz`` e quindi localizzato con ``tz_localize(tz)``.
    - Se il timestamp è *timezone-aware*, viene convertito nel timezone ``tz``.
    - In entrambi i casi, il risultato finale viene convertito in **UTC**.

    Parameters
    ----------
    series
        Serie/array-like di timestamp (stringhe o datetime) letto da CSV.
    tz : str
        Timezone di riferimento dello scenario (es. "UTC", "Europe/Rome").

    Returns
    -------
    pandas.DatetimeIndex
        Indice temporale timezone-aware in UTC.

    Raises
    ------
    Exception
        Se il parsing dei timestamp fallisce (errors="raise").
    """
    dt_raw = pd.to_datetime(series, errors="raise")

    # `pd.to_datetime` può tornare Series o DatetimeIndex; convertiamo a DatetimeIndex.
    dt = pd.DatetimeIndex(dt_raw)

    if dt.tz is None:
        # Timestamp naive: interpretazione come ora locale in `tz`.
        dt = dt.tz_localize(tz)
    else:
        # Timestamp tz-aware: portiamo nel timezone di scenario prima della normalizzazione.
        dt = dt.tz_convert(tz)

    return dt.tz_convert("UTC")


def _ensure_tz(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def _validate_monotonic_no_dupes(s: pd.Series, name: str) -> None:
    idx = s.index
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name}: timestamp non monotoni.")
    if idx.has_duplicates:
        raise ValueError(f"{name}: timestamp duplicati.")


def _validate_regular_frequency(idx: pd.DatetimeIndex, expected: str, name: str) -> None:
    if len(idx) < 2:
        raise ValueError(f"{name}: serie troppo corta.")
    diffs = idx[1:] - idx[:-1]

    try:
        off = pd.tseries.frequencies.to_offset(expected)  # accetta "h", "H", "15min"
        expected_delta = pd.to_timedelta(off)
    except Exception:
        raise ValueError(f"{name}: frequenza attesa non supportata: {expected}")

    if not (diffs == expected_delta).all():
        raise ValueError(f"{name}: frequenza non regolare. Attesa={expected_delta}.")

