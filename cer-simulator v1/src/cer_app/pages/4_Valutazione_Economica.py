# -*- coding: utf-8 -*-
"""cer_app.pages.4_Valutazione_Economica

Pagina Streamlit per la valutazione economico-finanziaria di una Comunità Energetica
Rinnovabile (CER) a partire dagli output del bilanciamento energetico.

Ruolo del modulo
----------------
Questo modulo è parte del layer ``cer_app`` (orchestrazione UI) e si occupa di:

1) **Selezione scenario e run energetica** (coerente con la pagina Bilanciamento).
   - Nuova UX a scenari: ``data/sessions/<session>/bilanciamento/scenarios/<scenario>/outputs/run_*``.
   - Fallback legacy: ``data/sessions/<session>/bilanciamento/outputs/run_*``.

2) **Gestione delle assunzioni economiche** tramite tabelle/controlli Streamlit.
   Le assunzioni vengono serializzate come "scenario economico" (assumption pack)
   in ``data/sessions/<session>/economics/scenarios/<slug>/``.

3) **Esecuzione della valutazione economica** invocando il layer ``cer_core``:
   ``cer_core.economics.economic_model.evaluate_economics``.
   Il calcolo produce: conto economico (PnL), cash-flow e KPI (NPV/IRR/Payback)
   per membro e aggregato.

4) **Persistenza risultati e audit trail**:
   - Output economici in ``data/sessions/<session>/economics/outputs/run_<UTCtag>/``.
   - Registry run economici: ``economics/runs_index.jsonl``.
   - Puntatore run attivo: ``economics/active_run.txt``.

Input / Output (filesystem)
---------------------------
Input (da run energetico selezionato):
  - ``run_*/cer_hourly.csv``: serie orarie CER (indice temporale orario).
  - ``run_*/members_hourly_long.csv``: serie orarie per membro (long format).
  - ``run_*/run_config.json``: configurazione del run energetico (membri, asset, ecc.).

Output (run economico):
  - ``pnl_by_member.csv`` / ``cashflow_by_member.csv`` / ``kpis_by_member.csv``
  - ``pnl_total.csv`` / ``cashflow_total.csv`` / ``kpis_total.csv``
  - ``assumptions.json`` (o equivalente, gestito da ``save_economic_outputs``)

Assunzioni e invarianti
-----------------------
- **Unità**: energia in kWh; prezzi e tariffe in €/kWh; CAPEX in € (o €/kW, €/kWh);
  tassi (discount, inflation, tax) espressi come frazioni (0.06 = 6%).
- **Indice temporale**: le serie orarie del run energetico sono trattate come UTC
  (coerentemente con il bilanciamento). I profili caricati (es. PZO) vengono
  normalizzati con ``pd.to_datetime(..., utc=True)``.
- **Compatibilità UI/Core**: questo modulo non deve cambiare nomi file di output,
  colonne CSV o struttura directory attesa dalla UI.

Side effects
------------
- Creazione directory scenario/run se mancanti.
- Scrittura di file CSV/JSON nella cartella output economica.
- Append su file ``runs_index.jsonl`` e aggiornamento del file ``active_run.txt``.

Nota sull'ottimizzazione BESS
-----------------------------
L'eventuale sweep sulla taglia batteria è "ceteris paribus": varia la BESS di un
membro alla volta mantenendo invariati i flussi degli altri membri del run energetico
selezionato, e ricalcola ``E_cond`` della CER sulla base dei nuovi flussi.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from cer_app.session_paths import get_paths

from cer_core.bilanciamento.bilanciamento_energetico import (
    BatterySpec,
    MemberSpec,
    ProductionSpec,
    build_period_config,
    compute_member_energy_hourly,
    load_and_validate_member,
)
from cer_core.economics.economic_model import (
    EconomicsAssumptions,
    EnergyRunData,
    evaluate_economics,
    load_energy_run,
    save_economic_outputs,
    load_economic_result,
)

from cer_core.optimization.bess_greedy import run_bess_greedy, GreedyConfig

from cer_core.bilanciamento.run_registry import (
    get_active_energy_run,
    read_energy_run_registry,
    append_economic_run_record,
    read_economic_run_registry,
    set_active_economic_run,
    get_active_economic_run,
)

from cer_core.bilanciamento.fingerprint import sha256_file
from cer_core.bilanciamento.scenario import scenario_file_content_fingerprint_sha256


from cer_core.economics.econ_scenario import (
    get_active_econ_scenario,
    list_econ_scenarios,
    load_econ_scenario,
    save_econ_scenario,
    set_active_econ_scenario,
    get_econ_scenario_content_fingerprint_sha256,
)


# =============================================================================
# Page config
# =============================================================================

PAGE_TITLE = "Valutazione economica"
st.set_page_config(page_title=f"CER - {PAGE_TITLE}", page_icon="💶", layout="wide")
st.title("Valutazione economica CER")

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
}


# =============================================================================
# Session paths (coerenti con le altre pagine)
# =============================================================================

PATHS = get_paths()
SESSION_DIR = PATHS.session_dir

ECON_DIR = PATHS.econ_dir
ECON_OUTPUTS_DIR = PATHS.econ_outputs_dir

ECON_SCENARIOS_DIR = PATHS.econ_scenarios_dir
ECON_ACTIVE_SCENARIO_TXT = PATHS.econ_active_scenario_txt

ECON_RUNS_INDEX_JSONL = PATHS.econ_runs_index_jsonl
ECON_ACTIVE_RUN_TXT = PATHS.econ_active_run_txt

BIL_DIR = PATHS.bil_dir

# -----------------------------------------------------------------------------
# Bilanciamento: supporto alla nuova UX a scenari (come nella pagina
# "Bilanciamento CER").
#
# La pagina Bilanciamento salva i run sotto:
#   data/sessions/<session>/bilanciamento/scenarios/<scenario>/outputs/run_*/
# e mantiene un registry per-scenario.
#
# Qui replichiamo lo stesso schema per:
#   scenario energetico -> selezione run energetica -> valutazione economica.
# -----------------------------------------------------------------------------

SCENARIOS_DIR = BIL_DIR / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers (UI state sync)
# =============================================================================

def _safe_first_row_value(df: Optional[pd.DataFrame], col: str, default=None):
    """Ritorna df.loc[0, col] in modo safe."""
    try:
        if df is None or df.empty:
            return default
        return df.loc[0, col]
    except Exception:
        return default


def _sync_dashboard_mode_widgets_from_tables(*, force: bool = False) -> None:
    """Allinea i widget di scelta modalità (BUY/SELL/TIP/TIAD) ai dati caricati.

    Se carichi uno scenario economico, le tabelle in ``st.session_state`` vengono
    aggiornate; però i widget Streamlit possono restare ai valori di default.
    Subito dopo, la logica della dashboard usa i valori dei widget per riscrivere
    le tabelle (es. ``_apply_global_buy_mode`` / ``_set_policy_modes``) e quindi
    **annulla** le assunzioni caricate.

    Questa funzione viene chiamata *prima* del rendering dei widget e forza i
    valori coerenti con le tabelle (una sola volta per coppia run/scenario).
    """
    run_key = st.session_state.get("econ_tables_key")
    active_slug = get_active_econ_scenario(ECON_ACTIVE_SCENARIO_TXT) or ""
    sync_key = (run_key, active_slug)
    if not force and st.session_state.get("econ_ui_sync_key") == sync_key:
        return

    buy = st.session_state.get("econ_tariffs_buy_df")
    sell = st.session_state.get("econ_tariffs_sell_df")
    pol = st.session_state.get("econ_policy_df")

    # BUY
    buy_mode = "fixed"
    try:
        if buy is not None and not buy.empty and "buy_price_mode" in buy.columns:
            buy_mode = str(buy["buy_price_mode"].iloc[0]).strip().lower() or "fixed"
    except Exception:
        buy_mode = "fixed"

    buy_map = {
        "fixed": "Fisso",
        "f1f2f3": "Fasce F1/F2/F3",
        "pzo_plus_spread": "PZO + spread",
    }
    st.session_state["dash_buy_choice"] = buy_map.get(buy_mode, "Fisso")

    # SELL
    sell_mode = "fixed"
    try:
        if sell is not None and not sell.empty and "sell_price_mode" in sell.columns:
            sell_mode = str(sell["sell_price_mode"].iloc[0]).strip().lower() or "fixed"
    except Exception:
        sell_mode = "fixed"

    sell_map = {
        "fixed": "Fisso",
        "pzo": "PZO",
        "pzo_minus_fee": "PZO - fee",
    }
    st.session_state["dash_sell_choice"] = sell_map.get(sell_mode, "Fisso")

    # TIP/TIAD
    tip_mode = str(_safe_first_row_value(pol, "tip_mode", "fixed") or "fixed").strip().lower()
    tiad_mode = str(_safe_first_row_value(pol, "tiad_mode", "fixed") or "fixed").strip().lower()

    tip_map = {
        "fixed": "Fisso",
        "pzo_function": "f(PZO)",
        "rse_decree": "RSE (decreto)",
        "rse": "RSE (decreto)",
        "rse_tip": "RSE (decreto)",
    }
    tiad_map = {
        "fixed": "Fisso",
        "rse_arera": "RSE (ARERA)",
        "rse": "RSE (ARERA)",
        "rse_tiad": "RSE (ARERA)",
    }

    st.session_state["dash_tip_choice"] = tip_map.get(tip_mode, "Fisso")
    st.session_state["dash_tiad_choice"] = tiad_map.get(tiad_mode, "Fisso")

    # Parametri addizionali: evita override silenziosi al primo render
    if pol is not None and not pol.empty:
        if "tip_rse_macro_area" in pol.columns:
            st.session_state["dash_tip_rse_macro"] = str(_safe_first_row_value(pol, "tip_rse_macro_area", "NORD") or "NORD")
        if "cacer_type" in pol.columns:
            st.session_state["dash_cacer_type"] = str(_safe_first_row_value(pol, "cacer_type", "CER") or "CER")

    st.session_state["econ_ui_sync_key"] = sync_key


def _normalize_scenario_name(name: str) -> str:
    """Normalizza un nome scenario in uno slug filesystem-safe.

    Regole applicate:
      - strip e sostituzione spazi con underscore;
      - rimozione caratteri non alfanumerici (eccetto ``_`` e ``-``);
      - prevenzione path traversal (``..``, slash/backslash).

    Returns
    -------
    str
        Slug non vuoto; fallback a ``"default"``.
    """
    import re

    n = (name or "").strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^A-Za-z0-9_\-]", "", n)
    n = n.replace("..", "_").replace("/", "_").replace("\\", "_")
    return n or "default"


def _scenario_dir(scenario_name: str) -> Path:
    return SCENARIOS_DIR / _normalize_scenario_name(scenario_name)


def _scenario_paths(scenario_name: str) -> dict:
    """Costruisce (e crea) i path standard per uno scenario energetico.

    La pagina Bilanciamento salva run e file scenario sotto:
    ``bilanciamento/scenarios/<scenario>/``.

    Side effects
    ------------
    Crea le directory ``inputs`` e ``outputs`` se mancanti.
    """
    d = _scenario_dir(scenario_name)
    inputs_dir = d / "inputs"
    outputs_dir = d / "outputs"
    scenario_json = d / "scenario.json"
    runs_index_jsonl = d / "runs_index.jsonl"
    active_run_txt = d / "active_run.txt"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "inputs_dir": inputs_dir,
        "outputs_dir": outputs_dir,
        "scenario_json": scenario_json,
        "runs_index_jsonl": runs_index_jsonl,
        "active_run_txt": active_run_txt,
    }


def _list_energy_scenarios() -> List[str]:
    out: List[str] = []
    for p in sorted(SCENARIOS_DIR.glob("*")):
        if p.is_dir() and (p / "scenario.json").exists():
            out.append(p.name)
    if "default" not in out:
        out.insert(0, "default")
    return out


# Legacy fallback (vecchia struttura senza scenari)
LEGACY_INPUTS_DIR = PATHS.bil_inputs_dir
LEGACY_OUTPUTS_DIR = PATHS.bil_outputs_dir
LEGACY_RUNS_INDEX_JSONL = PATHS.bil_runs_index_jsonl
LEGACY_ACTIVE_RUN_TXT = PATHS.bil_active_run_txt

# Default iniziale: legacy (verrà sovrascritto dopo la selezione scenario energetico)
INPUTS_DIR = LEGACY_INPUTS_DIR
OUTPUTS_DIR = LEGACY_OUTPUTS_DIR
RUNS_INDEX_JSONL = LEGACY_RUNS_INDEX_JSONL
ACTIVE_RUN_TXT = LEGACY_ACTIVE_RUN_TXT

def consumption_path(member_id: int) -> Path:
    return INPUTS_DIR / f"member_{member_id}" / "consumption_15min.csv"


def production_path(member_id: int) -> Path:
    return INPUTS_DIR / f"member_{member_id}" / "production_hourly.csv"


# =============================================================================
# Helpers: defaults builders
# =============================================================================


def _members_from_run_cfg(run_cfg: dict) -> List[dict]:
    members = run_cfg.get("members", []) or []
    out = []
    for m in members:
        mm = dict(m)
        mm["id"] = int(mm.get("id"))
        mm["member_id"] = str(mm.get("id"))
        mm.setdefault("name", f"Membro {mm['id']}")
        mm.setdefault("role", "prosumer")
        mm.setdefault("num", 1)
        mm.setdefault("installed_capacity_kw", 0.0)
        mm.setdefault("new_plant", True)
        mm.setdefault("commissioning_month", "")
        mm.setdefault("production_mode", "totale")
        mm.setdefault("selected_areas", [])
        out.append(mm)
    return out


def _annual_energy_summaries(members_hourly_long: pd.DataFrame) -> pd.DataFrame:
    """Aggrega l'energia annua per membro dal run energetico.

    Parameters
    ----------
    members_hourly_long:
        Tabella oraria long-format prodotta dal bilanciamento, con almeno le colonne:
        ``member_id``, ``E_prel_kWh``, ``E_imm_kWh``, ``E_load_kWh``, ``E_prod_kWh``.

    Returns
    -------
    pandas.DataFrame
        Indicizzato per ``member_id`` (string), con colonne annue in kWh.

    Raises
    ------
    ValueError
        Se ``member_id`` non è presente.
    """
    df = members_hourly_long.copy()
    if "member_id" not in df.columns:
        raise ValueError("members_hourly_long: colonna member_id mancante")
    df["member_id"] = df["member_id"].astype(str)
    g = df.groupby("member_id", sort=False)
    out = pd.DataFrame(
        {
            "E_prel_year_kWh": g["E_prel_kWh"].sum().astype(float),
            "E_imm_year_kWh": g["E_imm_kWh"].sum().astype(float),
            "E_load_year_kWh": g["E_load_kWh"].sum().astype(float),
            "E_prod_year_kWh": g["E_prod_kWh"].sum().astype(float),
        }
    )
    out.index = out.index.astype(str)
    return out


def build_default_policy(energy_run: EnergyRunData) -> pd.DataFrame:
    """Costruisce la tabella *Policy CER* con valori di default.

    Note
    ----
    - ``alpha_consumers``: quota incentivi allocata ai consumatori.
    - ``tip_mode`` / ``tiad_mode``: modalità di valorizzazione ("fixed" o legate a PZO).
    - ``year0``: anno base per l'indicizzazione/interpolazione di profili annuali.
    """
    year0 = int(pd.Timestamp(energy_run.t0).year)
    return pd.DataFrame(
        [
            {
                "alpha_consumers": 0.45,
                # di default allineiamo la struttura degli incentivi a RSE
                "tip_mode": "rse_decree",
                "tiad_mode": "rse_arera",
                "tip_value_eur_kwh": 0.0,
                "tiad_value_eur_kwh": 0.0,
                "incentive_years": 20,
                "year0": year0,
                # ---- Parametri RSE-like (TIP)
                # Se tip_rse_power_kw è NaN o 0, il core usa come fallback la massima
                # 'installed_capacity_kw' tra i membri enabled con new_plant=True.
                "tip_rse_power_kw": float("nan"),
                "tip_rse_macro_area": "NORD",  # NORD/CENTRO/SUD
                "tip_rse_grant_intensity": 0.0,  # 0..1 (PNRR/grant)
                # ---- Parametri RSE-like (TIAD)
                "cacer_type": "CER",  # CER/AUC/NO_CACER
                "tiad_rse_TRASe_eur_mwh": 0.0,
                "tiad_rse_BTAU_eur_mwh": 0.0,
                "tiad_rse_Cpr_bt": 0.0,
                "tiad_rse_Cpr_mt": 0.0,
                "tiad_rse_share_bt": float("nan"),
                # ---- Escalation (separate da prezzi buy/sell)
                "tip_escalation_rate": 0.0,
                "tiad_escalation_rate": 0.0,
            }
        ]
    )


def build_default_users(energy_run: EnergyRunData) -> pd.DataFrame:
    """Crea l'anagrafica economica di default a partire dal ``run_config``.

    La tabella include, per ogni membro, ruolo (consumer/producer/prosumer), classe
    fiscale (family/business) e campi utili per scalare asset e costi (es. ``num``).

    La funzione calcola inoltre riepiloghi annui (kWh) da ``members_hourly_long``
    per fornire contesto all'utente.
    """
    members = _members_from_run_cfg(energy_run.run_config)
    sums = _annual_energy_summaries(energy_run.members_hourly_long)

    rows = []
    for m in members:
        mid = str(m["member_id"])
        role = str(m.get("role", "prosumer")).lower()
        # default classificazione (editabile)
        if role == "consumer":
            user_class = "family"
        elif role == "producer":
            user_class = "business"
        else:
            user_class = "family"

        rows.append(
            {
                "member_id": mid,
                "name": str(m.get("name", f"Membro {mid}")),
                "role": role,
                "user_class": user_class,
                "enabled": True,
                "voltage_level": str(m.get("voltage_level", "BT")),
                "num": int(m.get("num", 1) or 1),
                "installed_capacity_kw": float(m.get("installed_capacity_kw", 0.0) or 0.0),
                "new_plant": bool(m.get("new_plant", True)),
                "commissioning_month": str(m.get("commissioning_month", "")),
                "E_prel_year_kWh": float(sums.loc[mid, "E_prel_year_kWh"]) if mid in sums.index else 0.0,
                "E_imm_year_kWh": float(sums.loc[mid, "E_imm_year_kWh"]) if mid in sums.index else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_default_tariffs_buy(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in users.iterrows():
        uclass = str(r.get("user_class", "family")).lower()
        # default: famiglie a fasce, imprese fixed (utente poi può passare a PZO)
        mode = "f1f2f3" if uclass == "family" else "fixed"
        rows.append(
            {
                "member_id": str(r["member_id"]),
                "buy_price_mode": mode,
                "buy_fixed_eur_kwh": 0.25,
                "f1_eur_kwh": 0.30,
                "f2_eur_kwh": 0.28,
                "f3_eur_kwh": 0.25,
                "buy_spread_eur_kwh": 0.0,
                "buy_multiplier": 1.0,
                "annual_fixed_fee_eur": 0.0,
                "power_fee_eur_per_kw_year": 0.0,
                "contract_power_kw": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_default_tariffs_sell(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in users.iterrows():
        role = str(r.get("role", "prosumer")).lower()
        # abilita vendita per producer/prosumer di default
        sell_enabled = role in ("producer", "prosumer")
        rows.append(
            {
                "member_id": str(r["member_id"]),
                "sell_enabled": bool(sell_enabled),
                "sell_price_mode": "fixed",
                "sell_fixed_eur_kwh": 0.10,
                "sell_fee_eur_kwh": 0.0,
                "sell_multiplier": 1.0,
                "annual_rid_fee_eur": 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_default_assets_pv(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in users.iterrows():
        mid = str(r["member_id"])
        role = str(r.get("role", "prosumer")).lower()
        num = int(r.get("num", 1) or 1)

        pv_exists = role in ("producer", "prosumer") or float(r.get("E_imm_year_kWh", 0.0) or 0.0) > 0
        # default: impianto "sunk" se new_plant=False
        pv_is_sunk = not bool(r.get("new_plant", True))
        pv_kw = float(r.get("installed_capacity_kw", 0.0) or 0.0) * float(num)

        rows.append(
            {
                "member_id": mid,
                "pv_exists": bool(pv_exists),
                "pv_is_sunk": bool(pv_is_sunk),
                "pv_capex_eur_per_kw": 1200.0,
                "pv_installed_kw": pv_kw,
                "pv_capex_override_eur": np.nan,
                "pv_opex_eur_per_kw_year": 20.0,
                "pv_life_years": 25,
                "pv_inverter_repl_year": 12,
                "pv_inverter_repl_eur_per_kw": 120.0,
            }
        )
    return pd.DataFrame(rows)


def build_default_assets_wind(users: pd.DataFrame) -> pd.DataFrame:
    """Tabella default asset Eolico (per utente).

    In questa prima integrazione il bilanciamento energetico potrebbe non includere
    ancora la produzione eolica: qui gestiamo comunque i flussi economici (CAPEX/OPEX,
    vita utile e sostituzioni) per poter chiudere correttamente i cashflow.
    """
    rows = []
    for _, r in users.iterrows():
        mid = str(r["member_id"])
        num = int(r.get("num", 1) or 1)

        # default conservative: eolico non presente
        wind_exists = bool(r.get("wind_exists", False))
        wind_is_sunk = not bool(r.get("new_plant", True))

        # se l'utente ha fornito "wind_installed_kw" lo uso, altrimenti 0
        base_kw = float(r.get("wind_installed_kw", 0.0) or 0.0) * float(num)

        rows.append(
            {
                "member_id": mid,
                "wind_exists": bool(wind_exists),
                "wind_is_sunk": bool(wind_is_sunk),
                "wind_capex_eur_per_kw": 2000.0,
                "wind_installed_kw": float(base_kw),
                "wind_capex_override_eur": np.nan,
                "wind_opex_eur_per_kw_year": 60.0,
                "wind_life_years": 20,
                "wind_major_repl_year": 10,
                "wind_major_repl_eur_per_kw": 200.0,
            }
        )
    return pd.DataFrame(rows)



def _battery_defaults_from_run_cfg(run_cfg: dict, member_id: str) -> Dict[str, float]:
    # fallback coerenti con migrazione membri della pagina bilanciamento
    defaults = {
        "dod": 0.8,
        "roundtrip_eff": 0.9,
        "derating_factor": 0.0,
        "init_soc_perc": 0.2,
    }
    for m in _members_from_run_cfg(run_cfg):
        if str(m.get("member_id")) != str(member_id):
            continue
        batt = m.get("battery")
        if isinstance(batt, dict):
            for k in defaults:
                if k in batt and batt[k] not in (None, ""):
                    defaults[k] = float(batt[k])
        return defaults
    return defaults


def build_default_assets_bess(energy_run: EnergyRunData, users: pd.DataFrame) -> pd.DataFrame:
    members = {str(m["member_id"]): m for m in _members_from_run_cfg(energy_run.run_config)}
    rows = []
    for _, r in users.iterrows():
        mid = str(r["member_id"])
        num = int(r.get("num", 1) or 1)
        m = members.get(mid, {})
        batt = m.get("battery") or None
        # NB: in bilanciamento, i flussi vengono scalati per num; per coerenza,
        #     trattiamo qui bess_initial_kwh come taglia aggregata.
        base_kwh_per_user = float(batt.get("capacity_kwh")) if isinstance(batt, dict) else 0.0
        base_kwh_agg = base_kwh_per_user * float(num)
        rows.append(
            {
                "member_id": mid,
                "bess_optimize": True,
                "bess_initial_kwh": float(base_kwh_agg),
                "bess_candidate_min_kwh": 0.0,
                "bess_candidate_max_kwh": max(20.0, float(base_kwh_agg) if base_kwh_agg > 0 else 20.0),
                "bess_step_kwh": 1.0,
                "bess_capex_eur_per_kwh": 500.0,
                "bess_opex_pct_capex": 0.02,
                "bess_opex_eur_per_kwh_year": 0.0,
                "bess_life_years": 12,
                "bess_c_rate_kw_per_kwh": 0.5,
                "bess_replacement": True,
            }
        )
    return pd.DataFrame(rows)


def build_default_tax_by_class() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_class": "family",
                "tax_enabled": False,
                "tax_rate_effective": 0.0,
                "allow_tax_loss_carryforward": False,
                "loss_carry_years": 0,
                "depreciation_enabled": True,
            },
            {
                "user_class": "business",
                "tax_enabled": True,
                "tax_rate_effective": 0.28,
                "allow_tax_loss_carryforward": True,
                "loss_carry_years": 5,
                "depreciation_enabled": True,
            },
        ]
    )


def build_default_tax_overrides(users: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in users.iterrows():
        rows.append(
            {
                "member_id": str(r["member_id"]),
                "tax_enabled_override": np.nan,
                "tax_rate_override": np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_default_dcf_params() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon_years": 20,
                "discount_rate": 0.06,
                "inflation_rate": 0.0,
                "escalation_buy": 0.0,
                "escalation_sell": 0.0,
                "escalation_opex": 0.0,
                "normalize_to_annual": True,
                "apply_incentives_beyond_run_year": True,
                "working_capital_enabled": False,
                "dso_days": 0,
                "dpo_days": 0,
                "detraction_enabled": False,
                "detraction_rate": 0.0,
                "detraction_cap_eur": 96000.0,
                "detraction_years": 10,
            }
        ]
    )


def _sync_table_by_members(df: pd.DataFrame, member_ids: List[str], *, id_col: str = "member_id") -> pd.DataFrame:
    """Allinea una tabella di assunzioni ai ``member_id`` del run.

    - Aggiunge righe mancanti (con NaN) per nuovi membri.
    - Rimuove righe in eccesso per membri non presenti nel run.
    - Ordina le righe secondo l'ordine di ``member_ids``.

    Questo è fondamentale quando si carica uno scenario economico salvato su una
    configurazione membri diversa dal run attuale.
    """
    if df is None or df.empty:
        return df
    x = df.copy()
    if id_col not in x.columns:
        return x
    x[id_col] = x[id_col].astype(str)
    existing = set(x[id_col].astype(str).tolist())
    missing = [m for m in member_ids if m not in existing]
    if missing:
        # append righe vuote con NaN (poi verranno compilate dall'utente)
        add = pd.DataFrame([{id_col: m} for m in missing])
        x = pd.concat([x, add], ignore_index=True)
    # drop extra
    x = x[x[id_col].astype(str).isin(member_ids)].copy()
    # order
    x[id_col] = pd.Categorical(x[id_col].astype(str), categories=member_ids, ordered=True)
    x = x.sort_values(id_col).reset_index(drop=True)
    x[id_col] = x[id_col].astype(str)
    return x


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# =============================================================================
# UI: Run selection
# =============================================================================

st.subheader("Selezione scenario e run energetica")

# --- Scenario energetico (come nella pagina Bilanciamento) ---
energy_scenarios = _list_energy_scenarios()
active_energy_scenario = _normalize_scenario_name(st.session_state.get("active_scenario", "default"))

# Se non esistono scenari (solo legacy), mostra comunque un'opzione "legacy"
has_new_scenarios = any((_scenario_dir(s).exists() and (_scenario_dir(s) / "scenario.json").exists()) for s in energy_scenarios)
if not has_new_scenarios and not (LEGACY_OUTPUTS_DIR.exists() or LEGACY_RUNS_INDEX_JSONL.exists()):
    # nessun dato energetico in assoluto: lasciamo che il blocco run mostri l'avviso
    pass

col_sc, col_spacer = st.columns([1.2, 2.8])
with col_sc:
    picked_energy_scenario = st.selectbox(
        "Scenario energetico",
        options=energy_scenarios,
        index=energy_scenarios.index(active_energy_scenario) if active_energy_scenario in energy_scenarios else 0,
        key="econ_energy_scenario",
    )

picked_energy_scenario_norm = _normalize_scenario_name(picked_energy_scenario)
if picked_energy_scenario_norm != active_energy_scenario:
    st.session_state["active_scenario"] = picked_energy_scenario_norm
    # reset selezione run se cambio scenario
    st.session_state.pop("econ_selected_run", None)
    st.rerun()

# Applica i path per-scenario (nuova UX); se scenario non esiste, fallback legacy.
_sp = _scenario_paths(picked_energy_scenario_norm)
if _sp["scenario_json"].exists():
    INPUTS_DIR = _sp["inputs_dir"]
    OUTPUTS_DIR = _sp["outputs_dir"]
    RUNS_INDEX_JSONL = _sp["runs_index_jsonl"]
    ACTIVE_RUN_TXT = _sp["active_run_txt"]
else:
    INPUTS_DIR = LEGACY_INPUTS_DIR
    OUTPUTS_DIR = LEGACY_OUTPUTS_DIR
    RUNS_INDEX_JSONL = LEGACY_RUNS_INDEX_JSONL
    ACTIVE_RUN_TXT = LEGACY_ACTIVE_RUN_TXT


# --- Se disponibile, usa il registry dei run per avere un ordinamento stabile e default coerente ---
runs = []
_registry = read_energy_run_registry(RUNS_INDEX_JSONL)
if _registry:
    # Più recenti prima
    _registry_sorted = sorted(_registry, key=lambda r: r.get("created_at_utc", ""), reverse=True)
    for r in _registry_sorted:
        p = Path(str(r.get("run_dir", "")))
        if not p.is_absolute():
            p = SESSION_DIR / p
        if p.exists() and p.is_dir():
            runs.append(p)

# fallback: scansione directory (backward compatible)
if not runs:
    # nuova UX: i run vivono in OUTPUTS_DIR (per-scenario). Per compatibilità,
    # se OUTPUTS_DIR è legacy, questa scansione funziona comunque.
    runs = sorted([d for d in OUTPUTS_DIR.glob('run_*') if d.is_dir()], reverse=True)

if not runs:
    st.warning(
        "Nessun run energetico trovato. Vai prima in 'Bilanciamento CER' e premi 'Esegui simulazione' "
        "per generare gli output (run_*)."
    )
    st.stop()

run_labels = [p.name for p in runs]

# default: active_run (se presente) -> session_state -> primo
default_idx = 0
_active_id = get_active_energy_run(ACTIVE_RUN_TXT)
if _active_id:
    _active_name = f"run_{_active_id}"
    if _active_name in run_labels:
        default_idx = run_labels.index(_active_name)

if "econ_selected_run" in st.session_state and st.session_state["econ_selected_run"] in run_labels:
    default_idx = run_labels.index(st.session_state["econ_selected_run"])

sel_run = st.selectbox("Run energetico", options=run_labels, index=default_idx)
st.session_state["econ_selected_run"] = sel_run
@st.cache_data(show_spinner=False)
def _cached_load_run(run_dir_str: str) -> EnergyRunData:
    return load_energy_run(Path(run_dir_str))


run_dir = next(p for p in runs if p.name == sel_run)
energy_run = _cached_load_run(str(run_dir))

# Summary
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Periodo", f"{energy_run.t0.date()} → {energy_run.t1.date()}")
with colB:
    st.metric("Ore run", int(len(energy_run.cer_hourly.index)))
with colC:
    st.metric("E_cond tot (kWh)", f"{float(energy_run.cer_hourly['E_cond_kWh'].sum()):,.0f}")
with colD:
    st.metric("Membri", int(energy_run.members_hourly_long["member_id"].nunique()))


# =============================================================================
# Init state tables per-run
# =============================================================================


def _init_tables_if_needed(energy_run: EnergyRunData) -> None:
    """Inizializza le tabelle economiche in ``st.session_state``.

    L'inizializzazione è *per-run energetico*: la chiave ``econ_tables_key``
    include il nome della directory ``run_*``; se la chiave non cambia, la
    funzione non modifica lo stato.

    Side effects
    ------------
    Popola/override le seguenti chiavi di sessione:
      - ``econ_policy_df`` (policy CER)
      - ``econ_users_df`` (anagrafica e classi fiscali)
      - ``econ_tariffs_buy_df`` / ``econ_tariffs_sell_df``
      - ``econ_assets_pv_df`` / ``econ_assets_wind_df`` / ``econ_assets_bess_df``
      - ``econ_tax_by_class_df`` / ``econ_tax_overrides_df``
      - ``econ_dcf_df``
      - profili opzionali (PZO/TIP/TIAD) e cache risultati.
    """
    run_key = f"econ_tables::{energy_run.run_dir.name}"
    if st.session_state.get("econ_tables_key") == run_key:
        return

    users_df = build_default_users(energy_run)
    member_ids = users_df["member_id"].astype(str).tolist()

    st.session_state["econ_policy_df"] = build_default_policy(energy_run)
    st.session_state["econ_users_df"] = users_df
    st.session_state["econ_tariffs_buy_df"] = build_default_tariffs_buy(users_df)
    st.session_state["econ_tariffs_sell_df"] = build_default_tariffs_sell(users_df)
    st.session_state["econ_assets_pv_df"] = build_default_assets_pv(users_df)
    st.session_state["econ_assets_wind_df"] = build_default_assets_wind(users_df)
    st.session_state["econ_assets_bess_df"] = build_default_assets_bess(energy_run, users_df)
    st.session_state["econ_tax_by_class_df"] = build_default_tax_by_class()
    st.session_state["econ_tax_overrides_df"] = build_default_tax_overrides(users_df)
    st.session_state["econ_dcf_df"] = build_default_dcf_params()

    # profiles
    st.session_state["econ_pzo_profile"] = None
    st.session_state["econ_tip_profile"] = None
    st.session_state["econ_tiad_profile"] = None
    st.session_state["econ_buy_profiles"] = None
    st.session_state["econ_sell_profiles"] = None

    st.session_state["econ_result"] = None
    st.session_state["econ_last_out_dir"] = None
    st.session_state["econ_opt_results"] = None

    st.session_state["econ_tables_key"] = run_key

    # forza resync dei widget dashboard al prossimo render
    st.session_state.pop("econ_ui_sync_key", None)


_init_tables_if_needed(energy_run)

# Allinea i widget modalità (BUY/SELL/TIP/TIAD) ai valori attuali delle tabelle.
_sync_dashboard_mode_widgets_from_tables()



# =============================================================================
# Scenario economico: salvataggio / ricarica "assumption pack"
# =============================================================================

def _apply_loaded_assumptions(loaded: EconomicsAssumptions) -> None:
    """Applica un *assumption pack* allo stato Streamlit.

    La funzione sincronizza i ``member_id`` dello scenario economico con quelli del
    run energetico attualmente selezionato.

    Side effects
    ------------
    Aggiorna molte chiavi di ``st.session_state`` (tabelle e profili) e invalida
    i risultati economici precedentemente calcolati.
    """
    run_member_ids = [str(m["member_id"]) for m in _members_from_run_cfg(energy_run.run_config)]

    st.session_state["econ_policy_df"] = loaded.policy_cer.copy()

    users_df = _sync_table_by_members(loaded.users.copy(), run_member_ids)
    st.session_state["econ_users_df"] = users_df

    st.session_state["econ_tariffs_buy_df"] = _sync_table_by_members(loaded.tariffs_buy.copy(), run_member_ids)
    st.session_state["econ_tariffs_sell_df"] = _sync_table_by_members(loaded.tariffs_sell.copy(), run_member_ids)
    def _ensure_member_id_column_local(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "member_id" in df.columns:
            return df
        x = df.copy()
        if getattr(x.index, "name", None) == "member_id":
            return x.reset_index()
        if "Unnamed: 0" in x.columns and "member_id" not in x.columns:
            return x.rename(columns={"Unnamed: 0": "member_id"})
        if not isinstance(x.index, pd.RangeIndex):
            return x.reset_index().rename(columns={"index": "member_id"})
        return x

    st.session_state["econ_assets_pv_df"] = _sync_table_by_members(_ensure_member_id_column_local(loaded.assets_pv.copy()), run_member_ids)
    # wind support (backward compat: scenario might not have it)
    loaded_wind = getattr(loaded, "assets_wind", None)
    if loaded_wind is None:
        st.session_state["econ_assets_wind_df"] = build_default_assets_wind(users_df)
    else:
        st.session_state["econ_assets_wind_df"] = _sync_table_by_members(_ensure_member_id_column_local(loaded_wind.copy()), run_member_ids)
    st.session_state["econ_assets_bess_df"] = _sync_table_by_members(_ensure_member_id_column_local(loaded.assets_bess.copy()), run_member_ids)

    st.session_state["econ_tax_by_class_df"] = loaded.tax_by_class.copy()

    if loaded.tax_overrides is None:
        tax_over = build_default_tax_overrides(users_df)
    else:
        tax_over = loaded.tax_overrides.copy()
    st.session_state["econ_tax_overrides_df"] = _sync_table_by_members(_ensure_member_id_column_local(tax_over), run_member_ids)

    st.session_state["econ_dcf_df"] = loaded.dcf_params.copy()

    # profiles (optional)
    st.session_state["econ_pzo_profile"] = loaded.pzo_profile
    st.session_state["econ_tip_profile"] = loaded.tip_profile
    st.session_state["econ_tiad_profile"] = loaded.tiad_profile
    st.session_state["econ_buy_profiles"] = loaded.buy_profiles
    st.session_state["econ_sell_profiles"] = loaded.sell_profiles

    # invalidate results
    st.session_state["econ_result"] = None
    st.session_state["econ_last_out_dir"] = None
    st.session_state["econ_opt_results"] = None

    # dopo un load scenario, riallinea i widget dashboard alle tabelle appena caricate
    st.session_state.pop("econ_ui_sync_key", None)
    _sync_dashboard_mode_widgets_from_tables(force=True)



def _snapshot_current_assumptions() -> EconomicsAssumptions:
    """Cattura uno snapshot coerente delle assunzioni correnti dalla UI.

    Returns
    -------
    EconomicsAssumptions
        Oggetto dataclass usato dal core economico.

    Notes
    -----
    - Normalizza ``member_id`` a stringa.
    - Gestisce campi tri-state (es. override fiscali) traducendo input UI in
      ``NaN``/``True``/``False``.
    """
    def _ensure_member_id_column(df: pd.DataFrame) -> pd.DataFrame:
        """Garantisce la presenza della colonna 'member_id'.

        Alcuni scenari esportati/salvati (o CSV legacy) possono avere
        ``member_id`` come indice oppure in una colonna tipo 'Unnamed: 0'.
        Il core economico richiede sempre una colonna ``member_id``.
        """
        if df is None or df.empty:
            return df
        if "member_id" in df.columns:
            return df
        x = df.copy()
        # Caso 1: member_id è già l'indice nominato
        if getattr(x.index, "name", None) == "member_id":
            x = x.reset_index()
            return x
        # Caso 2: CSV salvato con index=True -> 'Unnamed: 0'
        if "Unnamed: 0" in x.columns and "member_id" not in x.columns:
            x = x.rename(columns={"Unnamed: 0": "member_id"})
            return x
        # Caso 3: indice non banale -> reset e rinomina
        if not isinstance(x.index, pd.RangeIndex):
            x = x.reset_index().rename(columns={"index": "member_id"})
        return x

    policy = st.session_state["econ_policy_df"].copy()
    users = st.session_state["econ_users_df"].copy()
    buy = st.session_state["econ_tariffs_buy_df"].copy()
    sell = st.session_state["econ_tariffs_sell_df"].copy()
    pv = st.session_state["econ_assets_pv_df"].copy()
    wind = st.session_state.get("econ_assets_wind_df")
    wind = wind.copy() if wind is not None else None
    bess = st.session_state["econ_assets_bess_df"].copy()
    tax_class = st.session_state["econ_tax_by_class_df"].copy()
    tax_over = st.session_state["econ_tax_overrides_df"].copy()
    dcf = st.session_state["econ_dcf_df"].copy()

    # ensure member_id column exists (legacy scenarios may store it as index)
    pv = _ensure_member_id_column(pv)
    if wind is not None:
        wind = _ensure_member_id_column(wind)
    bess = _ensure_member_id_column(bess)
    tax_over = _ensure_member_id_column(tax_over)

    # ensure member_id types
    for df in [users, buy, sell, pv, wind, bess, tax_over]:
        if df is not None and not df.empty and "member_id" in df.columns:
            df["member_id"] = df["member_id"].astype(str)

    # tri-state override: "" -> NaN, "True"/"False" -> bool
    if tax_over is not None and not tax_over.empty and "tax_enabled_override" in tax_over.columns:
        def _parse_tri_bool(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            s = str(v).strip()
            if s == "":
                return np.nan
            if s.lower() in ("true", "1", "yes"):
                return True
            if s.lower() in ("false", "0", "no"):
                return False
            return np.nan

        tax_over["tax_enabled_override"] = tax_over["tax_enabled_override"].map(_parse_tri_bool)

    return EconomicsAssumptions(
        policy_cer=policy,
        users=users,
        tariffs_buy=buy,
        tariffs_sell=sell,
        assets_pv=pv,
        assets_wind=wind,
        assets_bess=bess,
        tax_by_class=tax_class,
        tax_overrides=tax_over,
        dcf_params=dcf,
        pzo_profile=st.session_state.get("econ_pzo_profile"),
        tip_profile=st.session_state.get("econ_tip_profile"),
        tiad_profile=st.session_state.get("econ_tiad_profile"),
        buy_profiles=st.session_state.get("econ_buy_profiles"),
        sell_profiles=st.session_state.get("econ_sell_profiles"),
    )


st.subheader("Scenario economico (assunzioni)")

_infos = list_econ_scenarios(ECON_SCENARIOS_DIR)
_slugs = [i.scenario_dir.name for i in _infos]
_active_slug = get_active_econ_scenario(ECON_ACTIVE_SCENARIO_TXT)

# Auto-load dello scenario attivo (1 volta per run) se presente
_auto_key = (st.session_state.get("econ_tables_key"), _active_slug)
if _active_slug and st.session_state.get("econ_loaded_scenario") != _auto_key:
    p = ECON_SCENARIOS_DIR / _active_slug
    if p.exists():
        try:
            _loaded, _meta = load_econ_scenario(p)
            _apply_loaded_assumptions(_loaded)
            st.session_state["econ_loaded_scenario"] = _auto_key
        except Exception:
            # tolleranza: uno scenario corrotto non deve bloccare la UI
            pass

with st.expander("Salva / carica scenario economico", expanded=False):
    if not _slugs:
        st.info("Nessuno scenario economico salvato. Puoi salvarne uno con 'Salva scenario corrente'.")
        sel_slug = None
    else:
        def _fmt(slug: str) -> str:
            for ii in _infos:
                if ii.scenario_dir.name == slug:
                    return f"{ii.name}  [{slug}]"
            return slug

        default_idx = 0
        if _active_slug in _slugs:
            default_idx = _slugs.index(_active_slug)
        sel_slug = st.selectbox(
            "Scenari disponibili",
            options=_slugs,
            index=default_idx,
            format_func=_fmt,
            key="econ_sel_scenario_slug",
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Carica scenario selezionato", type="secondary", disabled=(sel_slug is None)):
            try:
                loaded, meta = load_econ_scenario(ECON_SCENARIOS_DIR / str(sel_slug))
                _apply_loaded_assumptions(loaded)
                set_active_econ_scenario(ECON_ACTIVE_SCENARIO_TXT, str(sel_slug))
                st.session_state["econ_loaded_scenario"] = (st.session_state.get("econ_tables_key"), str(sel_slug))
                st.success(f"Scenario caricato: {sel_slug}")
            except Exception as e:
                st.error(str(e))

    with c2:
        default_name = sel_slug or (_active_slug or "default")
        save_name = st.text_input("Nome scenario (verrà usato come slug)", value=default_name, key="econ_save_scenario_name")
        if st.button("Salva scenario corrente", type="primary"):
            try:
                assumptions_to_save = _snapshot_current_assumptions()
                scen_dir = save_econ_scenario(
                    ECON_SCENARIOS_DIR,
                    name=str(save_name),
                    assumptions=assumptions_to_save,
                    overwrite=True,
                    extra_meta={"source_energy_run": str(energy_run.run_dir.name)},
                )
                set_active_econ_scenario(ECON_ACTIVE_SCENARIO_TXT, scen_dir.name)
                st.session_state["econ_loaded_scenario"] = (st.session_state.get("econ_tables_key"), scen_dir.name)
                st.success(f"Scenario salvato: {scen_dir.name}")
                st.rerun()
            except Exception as e:
                st.error(str(e))



def _get_member_ids_from_users(users_df: pd.DataFrame) -> List[str]:
    if users_df is None or users_df.empty:
        return []
    return users_df["member_id"].astype(str).tolist()


# =============================================================================
# UI: Assunzioni
# =============================================================================


st.subheader("Assunzioni economiche")

left, right = st.columns([1, 1])
with left:
    if st.button("Reset assunzioni (default)", type="secondary"):
        # reinit for current run
        st.session_state.pop("econ_tables_key", None)
        _init_tables_if_needed(energy_run)
        st.rerun()

with right:
    st.caption(
        "Suggerimento: inserisci prima 'Policy CER' e 'Anagrafica', poi tariffe, asset e fiscalità. "
        "Imposta il PZO solo se ti serve per TIP=f(PZO) o per prezzi acquisto/vendita indicizzati."
    )


# --- Dashboard unica (modalità avanzata rimossa) ---

def _safe_first(df: pd.DataFrame, col: str, default=None):
    try:
        return df.loc[0, col]
    except Exception:
        return default

def _apply_global_buy_mode(mode: str, *, fixed=None, f1=None, f2=None, f3=None, spread=None, mult=None):
    buy = st.session_state.get("econ_tariffs_buy_df")
    if buy is None or buy.empty:
        return
    buy = buy.copy()
    # NOTE (UI granularità per membro):
    # - il *mode* è globale (replicato su tutti i membri)
    # - i parametri *member-level* (es. fixed, spread) NON vengono più
    #   forzati globalmente; sono editabili in tabella per-member.
    buy["buy_price_mode"] = str(mode)
    # fixed / spread / fasce sono per-membro: NON sovrascrivere in bulk.
    # Per UX: se l'utente inserisce dei "default" globali (es. F1/F2/F3),
    # li usiamo solo per riempire eventuali celle vuote (NaN).
    if f1 is not None and "f1_eur_kwh" in buy.columns:
        buy["f1_eur_kwh"] = buy["f1_eur_kwh"].where(~buy["f1_eur_kwh"].isna(), float(f1))
    if f2 is not None and "f2_eur_kwh" in buy.columns:
        buy["f2_eur_kwh"] = buy["f2_eur_kwh"].where(~buy["f2_eur_kwh"].isna(), float(f2))
    if f3 is not None and "f3_eur_kwh" in buy.columns:
        buy["f3_eur_kwh"] = buy["f3_eur_kwh"].where(~buy["f3_eur_kwh"].isna(), float(f3))
    # spread è per-membro: non sovrascrivere in bulk
    if mult is not None and "buy_multiplier" in buy.columns:
        buy["buy_multiplier"] = float(mult)
    st.session_state["econ_tariffs_buy_df"] = buy

def _apply_global_sell_mode(mode: str, *, fixed=None, fee=None, mult=None, enabled=True):
    sell = st.session_state.get("econ_tariffs_sell_df")
    if sell is None or sell.empty:
        return
    sell = sell.copy()
    if "sell_enabled" in sell.columns:
        sell["sell_enabled"] = bool(enabled)
    # NOTE (UI granularità per membro):
    # - il *mode* è globale (replicato su tutti i membri)
    # - i parametri *member-level* (es. fixed, fee) NON vengono più
    #   forzati globalmente; sono editabili in tabella per-member.
    sell["sell_price_mode"] = str(mode)
    # fixed e fee sono per-membro: non sovrascrivere in bulk
    if mult is not None and "sell_multiplier" in sell.columns:
        sell["sell_multiplier"] = float(mult)
    st.session_state["econ_tariffs_sell_df"] = sell


def _render_member_tariff_table(*, kind: str, mode: str) -> None:
    """Editor tabellare per le tariffe *per membro*.

    La logica è guidata dal *mode globale* selezionato nella UI.
    TIP/TIAD rimangono globali e NON sono gestiti qui.
    """
    if kind == "buy":
        df_key = "econ_tariffs_buy_df"
        df = st.session_state.get(df_key)
        if df is None or df.empty:
            return

        df = df.copy()
        if "member_id" not in df.columns:
            st.error("Tabella tariffe acquisto: colonna 'member_id' mancante.")
            return

        if mode == "fixed":
            cols = ["member_id", "buy_fixed_eur_kwh"]
            title = "Override per membro – acquisto fisso (€/kWh)"
        elif mode == "pzo_plus_spread":
            cols = ["member_id", "buy_spread_eur_kwh"]
            title = "Override per membro – spread acquisto su PZO (€/kWh)"
        elif mode == "f1f2f3":
            cols = ["member_id", "f1_eur_kwh", "f2_eur_kwh", "f3_eur_kwh"]
            title = "Override per membro – acquisto per fasce F1/F2/F3 (€/kWh)"
        else:
            return

        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        view = df[cols].copy()
        st.markdown(f"#### {title}")
        edited = st.data_editor(
            view,
            use_container_width=True,
            num_rows="fixed",
            key=f"dash_{kind}_{mode}_table",
            column_config={
                "member_id": st.column_config.TextColumn("member_id", disabled=True),
            },
        )

        out = df.set_index("member_id")
        upd = edited.set_index("member_id")
        for c in cols:
            if c == "member_id":
                continue
            if c in upd.columns:
                out[c] = upd[c]
        st.session_state[df_key] = out.reset_index()
        return

    if kind == "sell":
        df_key = "econ_tariffs_sell_df"
        df = st.session_state.get(df_key)
        if df is None or df.empty:
            return

        df = df.copy()
        if "member_id" not in df.columns:
            st.error("Tabella tariffe vendita: colonna 'member_id' mancante.")
            return

        if mode == "fixed":
            cols = ["member_id", "sell_fixed_eur_kwh"]
            title = "Override per membro – vendita fissa (€/kWh)"
        elif mode == "pzo_minus_fee":
            cols = ["member_id", "sell_fee_eur_kwh"]
            title = "Override per membro – fee vendita su PZO (€/kWh)"
        else:
            # pzo: nessun parametro per-membro obbligatorio
            return

        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        view = df[cols].copy()
        st.markdown(f"#### {title}")
        edited = st.data_editor(
            view,
            use_container_width=True,
            num_rows="fixed",
            key=f"dash_{kind}_{mode}_table",
            column_config={
                "member_id": st.column_config.TextColumn("member_id", disabled=True),
            },
        )

        out = df.set_index("member_id")
        upd = edited.set_index("member_id")
        for c in cols:
            if c == "member_id":
                continue
            if c in upd.columns:
                out[c] = upd[c]
        st.session_state[df_key] = out.reset_index()
        return

def _set_policy_modes(*, tip_mode=None, tiad_mode=None, tip_val=None, tiad_val=None, alpha_c=None, years=None, year0=None):
    pol = st.session_state.get("econ_policy_df")
    if pol is None or pol.empty:
        return
    pol = pol.copy()
    if tip_mode is not None:
        pol.loc[0, "tip_mode"] = str(tip_mode)
    if tiad_mode is not None:
        pol.loc[0, "tiad_mode"] = str(tiad_mode)
    if tip_val is not None:
        pol.loc[0, "tip_value_eur_kwh"] = float(tip_val)
    if tiad_val is not None:
        pol.loc[0, "tiad_value_eur_kwh"] = float(tiad_val)
    if alpha_c is not None:
        pol.loc[0, "alpha_consumers"] = float(alpha_c)
    if years is not None:
        pol.loc[0, "incentive_years"] = int(years)
    if year0 is not None:
        pol.loc[0, "year0"] = int(year0)
    st.session_state["econ_policy_df"] = pol

def _set_dcf_basic(*, horizon=None, discount=None, inflation=None, esc_buy=None, esc_sell=None, esc_opex=None,
                  det_enabled=None, det_rate=None, det_cap=None, det_years=None):
    dcf = st.session_state.get("econ_dcf_df")
    if dcf is None or dcf.empty:
        return
    dcf = dcf.copy()
    if horizon is not None and "horizon_years" in dcf.columns:
        dcf.loc[0, "horizon_years"] = int(horizon)
    if discount is not None and "discount_rate" in dcf.columns:
        dcf.loc[0, "discount_rate"] = float(discount)
    if inflation is not None and "inflation_rate" in dcf.columns:
        dcf.loc[0, "inflation_rate"] = float(inflation)
    if esc_buy is not None and "escalation_buy" in dcf.columns:
        dcf.loc[0, "escalation_buy"] = float(esc_buy)
    if esc_sell is not None and "escalation_sell" in dcf.columns:
        dcf.loc[0, "escalation_sell"] = float(esc_sell)
    if esc_opex is not None and "escalation_opex" in dcf.columns:
        dcf.loc[0, "escalation_opex"] = float(esc_opex)
    # detrazione fiscale (Bonus ristrutturazione) - campi opzionali
    if det_enabled is not None:
        if "detraction_enabled" not in dcf.columns:
            dcf["detraction_enabled"] = False
        dcf.loc[0, "detraction_enabled"] = bool(det_enabled)
    if det_rate is not None:
        if "detraction_rate" not in dcf.columns:
            dcf["detraction_rate"] = 0.0
        dcf.loc[0, "detraction_rate"] = float(det_rate)
    if det_cap is not None:
        if "detraction_cap_eur" not in dcf.columns:
            dcf["detraction_cap_eur"] = 96000.0
        dcf.loc[0, "detraction_cap_eur"] = float(det_cap)
    if det_years is not None:
        if "detraction_years" not in dcf.columns:
            dcf["detraction_years"] = 10
        dcf.loc[0, "detraction_years"] = int(det_years)
    st.session_state["econ_dcf_df"] = dcf

# --- Dashboard helpers: Asset (PV/BESS) + Fiscalità ---
def _eligible_members(users: pd.DataFrame, *, roles: Optional[List[str]] = None) -> pd.DataFrame:
    """Ritorna users filtrati per enabled e (opzionale) ruolo."""
    u = users.copy()
    u["member_id"] = u["member_id"].astype(str)
    if "enabled" in u.columns:
        u = u[u["enabled"].fillna(True).astype(bool)]
    if roles is not None and "role" in u.columns:
        rr = set([str(r).strip().lower() for r in roles])
        u = u[u["role"].astype(str).str.strip().str.lower().isin(rr)]
    return u


def _apply_simple_pv(*, enabled: bool, apply_roles: List[str], kw_per_user: float, capex_eur_per_kw: float,
                     opex_eur_per_kw_year: float, life_years: int, sunk: bool,
                     inverter_repl_year: int, inverter_repl_eur_per_kw: float) -> None:
    pv = st.session_state.get("econ_assets_pv_df")
    users = st.session_state.get("econ_users_df")
    if pv is None or pv.empty or users is None or users.empty:
        return
    pv = pv.copy()
    users2 = _eligible_members(users, roles=apply_roles if enabled else [])
    eligible_ids = set(users2["member_id"].astype(str).tolist()) if enabled else set()

    pv["member_id"] = pv["member_id"].astype(str)
    # default: off
    if "pv_exists" in pv.columns:
        pv["pv_exists"] = pv["member_id"].isin(eligible_ids)
    if "pv_is_sunk" in pv.columns:
        pv["pv_is_sunk"] = bool(sunk)
    if "pv_capex_eur_per_kw" in pv.columns:
        pv["pv_capex_eur_per_kw"] = float(capex_eur_per_kw)
    if "pv_opex_eur_per_kw_year" in pv.columns:
        pv["pv_opex_eur_per_kw_year"] = float(opex_eur_per_kw_year)
    if "pv_life_years" in pv.columns:
        pv["pv_life_years"] = int(life_years)
    if "pv_inverter_repl_year" in pv.columns:
        pv["pv_inverter_repl_year"] = int(inverter_repl_year)
    if "pv_inverter_repl_eur_per_kw" in pv.columns:
        pv["pv_inverter_repl_eur_per_kw"] = float(inverter_repl_eur_per_kw)

    # sizing per user (kW/user * num)
    # NOTE: per compatibilità con tabella esistente: pv_installed_kw
    u_map = users.copy()
    u_map["member_id"] = u_map["member_id"].astype(str)
    if "num" in u_map.columns:
        u_map["num"] = u_map["num"].fillna(1).astype(int)
    else:
        u_map["num"] = 1
    num_by_id = dict(zip(u_map["member_id"], u_map["num"]))
    if "pv_installed_kw" in pv.columns:
        pv["pv_installed_kw"] = pv["member_id"].map(lambda mid: float(kw_per_user) * float(num_by_id.get(mid, 1)) if mid in eligible_ids else 0.0)

    st.session_state["econ_assets_pv_df"] = pv


def _apply_simple_bess(*, enabled: bool, apply_roles: List[str], kwh_per_user: float, capex_eur_per_kwh: float,
                       opex_pct_capex: float, opex_eur_per_kwh_year: float, opex_mode: str,
                       life_years: int, replacement: bool) -> None:
    bess = st.session_state.get("econ_assets_bess_df")
    users = st.session_state.get("econ_users_df")
    if bess is None or bess.empty or users is None or users.empty:
        return
    bess = bess.copy()
    users2 = _eligible_members(users, roles=apply_roles if enabled else [])
    eligible_ids = set(users2["member_id"].astype(str).tolist()) if enabled else set()

    bess["member_id"] = bess["member_id"].astype(str)

    # sizing kWh per user * num
    u_map = users.copy()
    u_map["member_id"] = u_map["member_id"].astype(str)
    if "num" in u_map.columns:
        u_map["num"] = u_map["num"].fillna(1).astype(int)
    else:
        u_map["num"] = 1
    num_by_id = dict(zip(u_map["member_id"], u_map["num"]))

    if "bess_initial_kwh" in bess.columns:
        bess["bess_initial_kwh"] = bess["member_id"].map(lambda mid: float(kwh_per_user) * float(num_by_id.get(mid, 1)) if mid in eligible_ids else 0.0)

    if "bess_capex_eur_per_kwh" in bess.columns:
        bess["bess_capex_eur_per_kwh"] = float(capex_eur_per_kwh)

    # OPEX: scegli 1 sola modalità semplice
    if opex_mode == "percent":
        if "bess_opex_pct_capex" in bess.columns:
            bess["bess_opex_pct_capex"] = float(opex_pct_capex)
        if "bess_opex_eur_per_kwh_year" in bess.columns:
            bess["bess_opex_eur_per_kwh_year"] = 0.0
    else:
        if "bess_opex_eur_per_kwh_year" in bess.columns:
            bess["bess_opex_eur_per_kwh_year"] = float(opex_eur_per_kwh_year)
        if "bess_opex_pct_capex" in bess.columns:
            bess["bess_opex_pct_capex"] = 0.0

    if "bess_life_years" in bess.columns:
        bess["bess_life_years"] = int(life_years)
    if "bess_replacement" in bess.columns:
        bess["bess_replacement"] = bool(replacement)

    st.session_state["econ_assets_bess_df"] = bess


def _set_tax_simple(*, enabled: bool, family_rate: float, business_rate: float,
                    depreciation_enabled: bool, carryforward_enabled: bool, carry_years: int) -> None:
    tax = st.session_state.get("econ_tax_by_class_df")
    if tax is None or tax.empty:
        return
    tax = tax.copy()
    tax["user_class"] = tax["user_class"].astype(str)
    if "tax_enabled" in tax.columns:
        tax["tax_enabled"] = bool(enabled)
    if "tax_rate_effective" in tax.columns:
        def _rate(uc):
            uc2 = str(uc).strip().lower()
            if uc2 == "business":
                return float(business_rate)
            return float(family_rate)
        tax["tax_rate_effective"] = tax["user_class"].map(_rate)
    if "depreciation_enabled" in tax.columns:
        tax["depreciation_enabled"] = bool(depreciation_enabled)
    if "allow_tax_loss_carryforward" in tax.columns:
        tax["allow_tax_loss_carryforward"] = bool(carryforward_enabled)
    if "loss_carry_years" in tax.columns:
        tax["loss_carry_years"] = int(carry_years) if carryforward_enabled else 0

    st.session_state["econ_tax_by_class_df"] = tax


def _model_transparency_summary() -> Dict[str, Any]:
    pol = st.session_state.get("econ_policy_df")
    buy = st.session_state.get("econ_tariffs_buy_df")
    sell = st.session_state.get("econ_tariffs_sell_df")
    pzo = st.session_state.get("econ_pzo_profile")
    pv = st.session_state.get("econ_assets_pv_df")
    bess = st.session_state.get("econ_assets_bess_df")
    tax = st.session_state.get("econ_tax_by_class_df")
    tax_ovr = st.session_state.get("econ_tax_overrides_df")
    dcf = st.session_state.get("econ_dcf_df")

    tip_mode = str(_safe_first(pol, "tip_mode", "fixed")).strip().lower() if pol is not None else "fixed"
    tiad_mode = str(_safe_first(pol, "tiad_mode", "fixed")).strip().lower() if pol is not None else "fixed"

    buy_modes: List[str] = []
    if buy is not None and not buy.empty and "buy_price_mode" in buy.columns:
        buy_modes = sorted(set(buy["buy_price_mode"].astype(str).str.strip().str.lower().tolist()))

    sell_modes: List[str] = []
    if sell is not None and not sell.empty and "sell_price_mode" in sell.columns:
        if "sell_enabled" in sell.columns:
            x = sell[sell["sell_enabled"].fillna(True).astype(bool)]
        else:
            x = sell
        sell_modes = sorted(set(x["sell_price_mode"].astype(str).str.strip().str.lower().tolist()))

    uses_pzo = (tip_mode in ("pzo_function", "rse_decree")) or (tiad_mode in ("rse_arera",))
    if any(m == "pzo_plus_spread" for m in buy_modes):
        uses_pzo = True
    if any(m in ("pzo", "pzo_minus_fee") for m in sell_modes):
        uses_pzo = True

    pzo_ok = (pzo is not None) and (not (isinstance(pzo, pd.DataFrame) and pzo.empty))

    notes: List[str] = []
    if not uses_pzo and pzo_ok:
        notes.append("PZO è presente ma NON viene usato dalle modalità correnti.")
    if uses_pzo and not pzo_ok:
        notes.append("PZO richiesto ma mancante: caricalo o simula un profilo.")

    # Asset summary (PV/BESS)
    pv_count = 0
    pv_overrides = 0
    if pv is not None and not pv.empty:
        x = pv.copy()
        x["member_id"] = x["member_id"].astype(str)
        exists = False
        if "pv_exists" in x.columns:
            exists = x["pv_exists"].fillna(False).astype(bool)
        else:
            exists = pd.Series([True] * len(x), index=x.index)
        if "pv_installed_kw" in x.columns:
            size = x["pv_installed_kw"].fillna(0).astype(float)
            pv_count = int(((exists) & (size > 0)).sum())
        else:
            pv_count = int(exists.sum())
        if "pv_capex_override_eur" in x.columns:
            pv_overrides = int(x["pv_capex_override_eur"].notna().sum())

    bess_count = 0
    bess_overrides = 0
    if bess is not None and not bess.empty:
        x = bess.copy()
        x["member_id"] = x["member_id"].astype(str)
        if "bess_initial_kwh" in x.columns:
            bess_count = int((x["bess_initial_kwh"].fillna(0).astype(float) > 0).sum())
        if "bess_optimize" in x.columns:
            bess_overrides = int(x["bess_optimize"].fillna(False).astype(bool).sum())

    # Tax summary
    tax_on = False
    tax_rate_family = None
    tax_rate_business = None
    depreciation = None
    carry = None
    carry_years = None
    if tax is not None and not tax.empty:
        t = tax.copy()
        if "tax_enabled" in t.columns:
            tax_on = bool(t["tax_enabled"].fillna(False).astype(bool).any())
        if "tax_rate_effective" in t.columns and "user_class" in t.columns:
            def _get_rate(cls):
                y = t[t["user_class"].astype(str).str.strip().str.lower() == cls]
                if y.empty:
                    return None
                return float(y["tax_rate_effective"].iloc[0])
            tax_rate_family = _get_rate("family")
            tax_rate_business = _get_rate("business")
        if "depreciation_enabled" in t.columns:
            depreciation = bool(t["depreciation_enabled"].fillna(False).astype(bool).any())
        if "allow_tax_loss_carryforward" in t.columns:
            carry = bool(t["allow_tax_loss_carryforward"].fillna(False).astype(bool).any())
        if "loss_carry_years" in t.columns:
            carry_years = int(t["loss_carry_years"].fillna(0).astype(int).max())

    tax_override_count = 0
    if tax_ovr is not None and not tax_ovr.empty:
        o = tax_ovr.copy()
        cols = [c for c in o.columns if c != "member_id"]
        if cols:
            tax_override_count = int(o[cols].notna().any(axis=1).sum())

    # DCF summary
    dcf_h = _safe_first(dcf, "horizon_years", None)
    dcf_dr = _safe_first(dcf, "discount_rate", None)
    dcf_inf = _safe_first(dcf, "inflation_rate", None)

    return {
        "energy_inputs": ["E_prel_kWh (prelievo rete)", "E_imm_kWh (immissione rete)", "E_cond_kWh (energia condivisa)"],
        "buy_mode": ", ".join(buy_modes) if buy_modes else "n/a",
        "sell_mode": ", ".join(sell_modes) if sell_modes else "n/a",
        "tip_mode": tip_mode,
        "tiad_mode": tiad_mode,
        "uses_pzo": uses_pzo,
        "pzo_ok": pzo_ok,
        "pv_count": pv_count,
        "pv_overrides": pv_overrides,
        "bess_count": bess_count,
        "bess_overrides": bess_overrides,
        "tax_on": tax_on,
        "tax_rate_family": tax_rate_family,
        "tax_rate_business": tax_rate_business,
        "depreciation": depreciation,
        "carry": carry,
        "carry_years": carry_years,
        "tax_override_count": tax_override_count,
        "dcf_horizon": dcf_h,
        "dcf_discount": dcf_dr,
        "dcf_inflation": dcf_inf,
        "notes": notes,
    }

def _render_model_transparency():
    s = _model_transparency_summary()
    st.markdown("### Model transparency")
    st.caption("Cosa entra davvero nei flussi e quali modalità sono attive.")

    st.markdown("**1) Driver energetici**")
    for x in s["energy_inputs"]:
        st.write(f"✅ {x}")

    st.markdown("**2) Prezzi & incentivi**")
    st.write(f"- Acquisto: **{s['buy_mode']}**")
    st.write(f"- Vendita: **{s['sell_mode']}**")
    st.write(f"- TIP: **{s['tip_mode']}**")
    st.write(f"- TIAD: **{s['tiad_mode']}**")
    st.write(f"- PZO usato: **{'Sì' if s['uses_pzo'] else 'No'}**  |  Profilo PZO: **{'OK' if s['pzo_ok'] else 'Mancante'}**")

    st.markdown("**3) Asset**")
    st.write(f"- FV attivo su membri: **{s['pv_count']}** (override CAPEX: {s['pv_overrides']})")
    st.write(f"- BESS attiva su membri: **{s['bess_count']}** (override/opt: {s['bess_overrides']})")

    st.markdown("**4) Finanza & tasse**")
    st.write(f"- DCF: horizon={s['dcf_horizon']}y, discount={s['dcf_discount']}, infl={s['dcf_inflation']}")
    st.write(f"- Tasse: **{'ON' if s['tax_on'] else 'OFF'}** | Aliquote family={s['tax_rate_family']} business={s['tax_rate_business']}")
    st.write(f"- Ammortamenti: **{s['depreciation']}** | Carryforward: **{s['carry']}** ({s['carry_years']} anni)")
    st.write(f"- Override fiscali membri: **{s['tax_override_count']}**")

    if s["notes"]:
        st.warning("\n".join(s["notes"]))
main, side = st.columns([3, 1], gap="large")
with side:
    _render_model_transparency()

with main:
    st.markdown("## 1) Prezzi & Incentivi")
    cA, cB, cC, cD = st.columns([1, 1, 1, 1])
    with cA:
        buy_choice = st.selectbox(
            "Prezzo acquisto (bolletta)",
            options=["Fisso", "Fasce F1/F2/F3", "PZO + spread"],
            key="dash_buy_choice",
        )
    with cB:
        sell_choice = st.selectbox(
            "Prezzo vendita (immissioni)",
            options=["Fisso", "PZO", "PZO - fee"],
            key="dash_sell_choice",
        )
    with cC:
        tip_choice = st.selectbox(
            "TIP",
            options=["Fisso", "f(PZO)", "RSE (decreto)"],
            key="dash_tip_choice",
        )
    with cD:
        tiad_choice = st.selectbox(
            "TIAD",
            options=["Fisso", "RSE (ARERA)"],
            key="dash_tiad_choice",
        )

    # --- Apply selections + show relevant parameters ---
    # BUY
    if buy_choice == "Fisso":
        _apply_global_buy_mode("fixed")
        _render_member_tariff_table(kind="buy", mode="fixed")
    elif buy_choice == "Fasce F1/F2/F3":
        # UI: niente "default globali" per le fasce. Si mantengono solo gli override per-membro.
        _apply_global_buy_mode("f1f2f3")
        _render_member_tariff_table(kind="buy", mode="f1f2f3")
    else:
        mult = st.number_input("Moltiplicatore PZO (acquisto)", value=float(_safe_first(st.session_state.get("econ_tariffs_buy_df"), "buy_multiplier", 1.0) or 1.0), step=0.05)
        _apply_global_buy_mode("pzo_plus_spread", mult=mult)
        _render_member_tariff_table(kind="buy", mode="pzo_plus_spread")

    # SELL
    st.divider()
    if sell_choice == "Fisso":
        _apply_global_sell_mode("fixed", enabled=True)
        _render_member_tariff_table(kind="sell", mode="fixed")
    elif sell_choice == "PZO":
        mult = st.number_input("Moltiplicatore PZO (vendita)", value=float(_safe_first(st.session_state.get("econ_tariffs_sell_df"), "sell_multiplier", 1.0) or 1.0), step=0.05)
        _apply_global_sell_mode("pzo", mult=mult, enabled=True)
    else:
        mult = st.number_input("Moltiplicatore PZO (vendita)", value=float(_safe_first(st.session_state.get("econ_tariffs_sell_df"), "sell_multiplier", 1.0) or 1.0), step=0.05, key="dash_sell_mult")
        _apply_global_sell_mode("pzo_minus_fee", mult=mult, enabled=True)
        _render_member_tariff_table(kind="sell", mode="pzo_minus_fee")

    # POLICY (TIP/TIAD + alpha)
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha_c = st.slider("α consumatori", min_value=0.0, max_value=1.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "alpha_consumers", 0.5) or 0.5), step=0.01)
    with c2:
        years = st.number_input("Durata incentivi (anni)", min_value=1, max_value=40, value=int(_safe_first(st.session_state.get("econ_policy_df"), "incentive_years", 20) or 20), step=1)
    with c3:
        tip_val = st.number_input("TIP fisso (€/kWh)", min_value=0.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tip_value_eur_kwh", 0.11) or 0.11), step=0.001)
    with c4:
        tiad_val = st.number_input("TIAD fisso (€/kWh)", min_value=0.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tiad_value_eur_kwh", 0.01) or 0.01), step=0.001)

    
    # Parametri RSE (mostrati solo quando selezionati)
    if tip_choice == "RSE (decreto)":
        st.markdown("#### Parametri TIP (RSE)")
        c1, c2, c3 = st.columns(3)
        with c1:
            macro = st.selectbox("Macro-area", options=["NORD", "CENTRO", "SUD"], key="dash_tip_rse_macro")
        with c2:
            grant = st.number_input("Intensità grant/PNRR (0..1)", min_value=0.0, max_value=1.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tip_rse_grant_intensity", 0.0) or 0.0), step=0.05)
        with c3:
            power = st.number_input("Potenza incentivata (kW) [opzionale]", min_value=0.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tip_rse_power_kw", 0.0) or 0.0), step=10.0)

        # Persist in policy table (year0/years handled below)
        pol = st.session_state.get("econ_policy_df")
        if pol is not None and not pol.empty:
            pol = pol.copy()
            pol.loc[0, "tip_rse_macro_area"] = str(macro)
            pol.loc[0, "tip_rse_grant_intensity"] = float(grant)
            # usa NaN se power==0 per attivare fallback nel core
            pol.loc[0, "tip_rse_power_kw"] = (float("nan") if float(power) <= 0.0 else float(power))
            st.session_state["econ_policy_df"] = pol

    if tiad_choice == "RSE (ARERA)":
        st.markdown("#### Parametri TIAD (RSE)")
        c1, c2, c3 = st.columns(3)
        with c1:
            cacer_type = st.selectbox("Tipo configurazione", options=["CER", "AUC"], key="dash_cacer_type")
        with c2:
            trase = st.number_input("TRASe (€/MWh)", min_value=0.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tiad_rse_TRASe_eur_mwh", 0.0) or 0.0), step=0.5)
        with c3:
            btau = st.number_input("BTAU (€/MWh)", min_value=0.0, value=float(_safe_first(st.session_state.get("econ_policy_df"), "tiad_rse_BTAU_eur_mwh", 0.0) or 0.0), step=0.5)

        c4, c5, c6 = st.columns(3)
        with c4:
            cpr_bt = st.number_input("Coeff. PZO BT", value=float(_safe_first(st.session_state.get("econ_policy_df"), "tiad_rse_Cpr_bt", 0.0) or 0.0), step=0.05)
        with c5:
            cpr_mt = st.number_input("Coeff. PZO MT", value=float(_safe_first(st.session_state.get("econ_policy_df"), "tiad_rse_Cpr_mt", 0.0) or 0.0), step=0.05)
        with c6:
            _sb = _safe_first(st.session_state.get("econ_policy_df"), "tiad_rse_share_bt", float("nan"))
            _sb_val = 1.0 if pd.isna(_sb) else float(_sb)
            share_bt = st.number_input("Quota BT (solo AUC)", min_value=0.0, max_value=1.0, value=_sb_val, step=0.05)

        pol = st.session_state.get("econ_policy_df")
        if pol is not None and not pol.empty:
            pol = pol.copy()
            pol.loc[0, "cacer_type"] = str(cacer_type)
            pol.loc[0, "tiad_rse_TRASe_eur_mwh"] = float(trase)
            pol.loc[0, "tiad_rse_BTAU_eur_mwh"] = float(btau)
            pol.loc[0, "tiad_rse_Cpr_bt"] = float(cpr_bt)
            pol.loc[0, "tiad_rse_Cpr_mt"] = float(cpr_mt)
            pol.loc[0, "tiad_rse_share_bt"] = float(share_bt)
            st.session_state["econ_policy_df"] = pol

    _set_policy_modes(
            tip_mode=("rse_decree" if tip_choice == "RSE (decreto)" else ("pzo_function" if tip_choice == "f(PZO)" else "fixed")),
            tiad_mode=("rse_arera" if tiad_choice == "RSE (ARERA)" else "fixed"),
            tip_val=tip_val,
            tiad_val=tiad_val,
            alpha_c=alpha_c,
            years=years,
    )

    # PZO panel: sempre visibile (anche se non richiesto dalle modalità correnti)
    st.divider()
    st.markdown("#### PZO (profilo orario)")
    _mt = _model_transparency_summary()
    if _mt["uses_pzo"]:
        st.info("Il PZO è richiesto dalle modalità selezionate (TIP/TIAD o prezzi indicizzati).")
    else:
        st.caption("Il PZO è opzionale: puoi caricarlo ora e attivarlo più avanti scegliendo una modalità che lo usa.")

    # stato attuale
    _pzo_cur = st.session_state.get("econ_pzo_profile")
    if _pzo_cur is not None and isinstance(_pzo_cur, pd.DataFrame) and not _pzo_cur.empty:
        try:
            st.success(f"PZO presente: {len(_pzo_cur)} righe")
        except Exception:
            st.success("PZO presente")
        c_prev, c_clear = st.columns([1, 1])
        with c_prev:
            with st.expander("Anteprima PZO", expanded=False):
                st.dataframe(_pzo_cur.head(24), use_container_width=True)
        with c_clear:
            if st.button("Rimuovi PZO", type="secondary", key="dash_pzo_clear"):
                st.session_state["econ_pzo_profile"] = None
                st.rerun()

    mode = st.radio("Fonte PZO", options=["Carica CSV", "Simula"], horizontal=True, key="dash_pzo_mode")
    def _normalize_pzo_df_dash(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        for c in list(x.columns):
            if str(c).startswith("Unnamed"):
                x = x.drop(columns=[c])
        if "time" not in x.columns and "timestamp" in x.columns:
            x = x.rename(columns={"timestamp": "time"})
        if "pzo_eur_kwh" not in x.columns:
            for cand in ("price_eur_kwh", "eur_kwh", "value", "PZO", "pzo"):
                if cand in x.columns:
                    x = x.rename(columns={cand: "pzo_eur_kwh"})
                    break
        if "time" not in x.columns or "pzo_eur_kwh" not in x.columns:
            raise ValueError("CSV PZO non valido: attese colonne time/timestamp e pzo_eur_kwh")
        x = x[["time", "pzo_eur_kwh"]].copy()
        x["time"] = pd.to_datetime(x["time"], utc=True)
        x["pzo_eur_kwh"] = x["pzo_eur_kwh"].astype(float)
        return x

    if mode == "Carica CSV":
        up = st.file_uploader("Carica PZO (CSV)", type=["csv"], key="dash_up_pzo")
        if up is not None:
            st.session_state["econ_pzo_profile"] = _normalize_pzo_df_dash(_read_uploaded_csv(up))
            st.success("PZO caricato.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            mean_pzo = st.number_input("Media (€/kWh)", min_value=0.0, value=0.12, step=0.01, key="dash_pzo_mean")
        with c2:
            std_pzo = st.number_input("Std (€/kWh)", min_value=0.0, value=0.03, step=0.01, key="dash_pzo_std")
        with c3:
            seed = st.number_input("Seed", min_value=0, value=42, step=1, key="dash_pzo_seed")
        if st.button("Genera PZO", type="secondary", key="dash_pzo_gen"):
            st.session_state["econ_pzo_profile"] = pd.DataFrame({"time": energy_run.cer_hourly.index, "pzo_eur_kwh": np.clip(np.random.default_rng(int(seed)).normal(mean_pzo, std_pzo, len(energy_run.cer_hourly.index)), 0.0, None)})
            st.success("PZO generato.")

    st.markdown("## 2) Utenti & Asset")
    st.caption("Gestisci anagrafica e classi fiscali (family/business) per membro. I dettagli avanzati sono esposti solo via expander mirati.")
    users_df = st.session_state["econ_users_df"].copy()
    users_df["member_id"] = users_df["member_id"].astype(str)

    # Streamlit data_editor: evita mismatch dtype (es. NaN -> float) su colonne testuali
    if "name" in users_df.columns:
        users_df["name"] = users_df["name"].astype("string").fillna("")
    if "role" in users_df.columns:
        users_df["role"] = users_df["role"].astype("string").fillna("consumer")
    if "user_class" in users_df.columns:
        users_df["user_class"] = users_df["user_class"].astype("string").fillna("family")
    if "enabled" in users_df.columns:
        # pandas può portare a object/float con NaN: normalizza a boolean
        users_df["enabled"] = users_df["enabled"].fillna(True).astype(bool)
    if "num" in users_df.columns:
        users_df["num"] = users_df["num"].fillna(1).astype(int)


    if "commissioning_month" in users_df.columns:
        cm = users_df["commissioning_month"].astype("string").fillna("")
        cm = cm.str.replace(r"\.0$", "", regex=True).str.strip()
        mask_yyyymm = cm.str.fullmatch(r"\d{6}")
        cm.loc[mask_yyyymm] = cm.loc[mask_yyyymm].str.slice(0, 4) + "-" + cm.loc[mask_yyyymm].str.slice(4, 6)
        users_df["commissioning_month"] = cm

    users_df = st.data_editor(
        users_df[["member_id","name","role","user_class","enabled","num","commissioning_month"]],
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "member_id": st.column_config.TextColumn("member_id", disabled=True),
            "name": st.column_config.TextColumn("Nome"),
            "role": st.column_config.SelectboxColumn("Ruolo", options=["consumer","producer","prosumer"]),
            "user_class": st.column_config.SelectboxColumn("Classe", options=["family","business"]),
            "enabled": st.column_config.CheckboxColumn("Attivo"),
            "num": st.column_config.NumberColumn("Molteplicità", min_value=1, step=1),
            "commissioning_month": st.column_config.TextColumn("Mese entrata (YYYY-MM)"),
        },
        key="dash_users_editor",
    )
    # merge back into full users df
    full_users = st.session_state["econ_users_df"].copy()
    full_users["member_id"] = full_users["member_id"].astype(str)
    full_users = full_users.drop(columns=[c for c in users_df.columns if c in full_users.columns and c != "member_id"]).merge(users_df, on="member_id", how="left")
    st.session_state["econ_users_df"] = full_users

    

    st.markdown("## 3) Asset & Fiscalità")
    st.caption("Modifica per utente taglia e costi principali di FV/WIND/BESS e la fiscalità di classe.")

    tab_pv, tab_wind, tab_bess, tab_tax = st.tabs(["Fotovoltaico (FV) — per utente", "Eolico (WIND) — per utente", "Batteria (BESS) — per utente", "Fiscalità (semplice)"])

    # ----------------------------
    # PV table (per utente)
    # ----------------------------
    with tab_pv:
        pv_df = st.session_state.get("econ_assets_pv_df")
        if pv_df is None or pv_df.empty:
            st.warning("Tabella FV non disponibile.")
        else:
            # View: join con anagrafica per rendere la tabella più leggibile
            u = st.session_state["econ_users_df"].copy()
            u["member_id"] = u["member_id"].astype(str)
            pv = pv_df.copy()
            pv["member_id"] = pv["member_id"].astype(str)

            pv_view = (
                u[["member_id", "name", "role", "enabled", "num"]]
                .merge(pv, on="member_id", how="left")
            )

            # Normalizzazione dtype per Streamlit data_editor
            for c in ["name", "role"]:
                if c in pv_view.columns:
                    pv_view[c] = pv_view[c].astype("string").fillna("")
            if "enabled" in pv_view.columns:
                pv_view["enabled"] = pv_view["enabled"].fillna(True).astype(bool)
            if "num" in pv_view.columns:
                pv_view["num"] = pd.to_numeric(pv_view["num"], errors="coerce").fillna(1).astype(int)

            # colonne PV editabili (dashboard)
            edit_cols = [
                "pv_exists",
                "pv_installed_kw",
                "pv_capex_eur_per_kw",
                "pv_opex_eur_per_kw_year",
                "pv_is_sunk",
                "pv_life_years",
                "pv_inverter_repl_year",
                "pv_inverter_repl_eur_per_kw",
            ]
            for c in edit_cols:
                if c not in pv_view.columns:
                    pv_view[c] = np.nan

            # coercion numeriche
            num_cols = ["pv_installed_kw", "pv_capex_eur_per_kw", "pv_opex_eur_per_kw_year", "pv_life_years", "pv_inverter_repl_year", "pv_inverter_repl_eur_per_kw"]
            for c in num_cols:
                pv_view[c] = pd.to_numeric(pv_view[c], errors="coerce")

            # bool
            for c in ["pv_exists", "pv_is_sunk"]:
                pv_view[c] = pv_view[c].fillna(False).astype(bool)

            pv_view = st.data_editor(
                pv_view[["member_id", "name", "role", "enabled", "num"] + edit_cols],
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "member_id": st.column_config.TextColumn("member_id", disabled=True),
                    "name": st.column_config.TextColumn("Nome", disabled=True),
                    "role": st.column_config.TextColumn("Ruolo", disabled=True),
                    "enabled": st.column_config.CheckboxColumn("Utente attivo", disabled=True),
                    "num": st.column_config.NumberColumn("Molteplicità", disabled=True),
                    "pv_exists": st.column_config.CheckboxColumn("FV presente"),
                    "pv_installed_kw": st.column_config.NumberColumn("Taglia FV (kW)", min_value=0.0, step=0.5),
                    "pv_capex_eur_per_kw": st.column_config.NumberColumn("CAPEX (€/kW)", min_value=0.0, step=25.0),
                    "pv_opex_eur_per_kw_year": st.column_config.NumberColumn("OPEX (€/kW/anno)", min_value=0.0, step=1.0),
                    "pv_is_sunk": st.column_config.CheckboxColumn("Sunk cost (CAPEX=0)"),
                    "pv_life_years": st.column_config.NumberColumn("Vita utile (anni)", min_value=1, step=1),
                    "pv_inverter_repl_year": st.column_config.NumberColumn("Anno repl inverter", min_value=0, step=1),
                    "pv_inverter_repl_eur_per_kw": st.column_config.NumberColumn("Costo repl inverter (€/kW)", min_value=0.0, step=5.0),
                },
                key="dash_pv_table",
            )

            # write-back: aggiorna solo le colonne dashboard, preservando colonne avanzate (es. capex_override)
            pv_out = pv.copy()
            keep = [c for c in pv_out.columns if c not in edit_cols]
            pv_out = pv_out[keep].merge(pv_view[["member_id"] + edit_cols], on="member_id", how="left")

            # Normalizza: se FV non presente -> forza taglia=0
            pv_out.loc[~pv_out["pv_exists"].astype(bool), "pv_installed_kw"] = 0.0

            st.session_state["econ_assets_pv_df"] = pv_out

            st.caption("Suggerimento: per applicare rapidamente lo stesso CAPEX/OPEX a tutti, imposta un valore in una riga e usa copia/incolla nel data editor.")


    # ----------------------------
    # WIND table (per utente)
    # ----------------------------
    with tab_wind:
        wind_df = st.session_state.get("econ_assets_wind_df")
        if wind_df is None or wind_df.empty:
            st.warning("Tabella Eolico non disponibile.")
        else:
            u = st.session_state["econ_users_df"].copy()
            u["member_id"] = u["member_id"].astype(str)
            w = wind_df.copy()
            w["member_id"] = w["member_id"].astype(str)

            wind_view = (
                u[["member_id", "name", "role", "enabled", "num"]]
                .merge(w, on="member_id", how="left")
            )

            for c in ["name", "role"]:
                if c in wind_view.columns:
                    wind_view[c] = wind_view[c].astype("string").fillna("")
            if "enabled" in wind_view.columns:
                wind_view["enabled"] = wind_view["enabled"].fillna(True).astype(bool)
            if "num" in wind_view.columns:
                wind_view["num"] = pd.to_numeric(wind_view["num"], errors="coerce").fillna(1).astype(int)

            edit_cols = [
                "wind_exists",
                "wind_installed_kw",
                "wind_capex_eur_per_kw",
                "wind_opex_eur_per_kw_year",
                "wind_is_sunk",
                "wind_life_years",
                "wind_major_repl_year",
                "wind_major_repl_eur_per_kw",
            ]
            for c in edit_cols:
                if c not in wind_view.columns:
                    wind_view[c] = np.nan

            num_cols = [
                "wind_installed_kw",
                "wind_capex_eur_per_kw",
                "wind_opex_eur_per_kw_year",
                "wind_life_years",
                "wind_major_repl_year",
                "wind_major_repl_eur_per_kw",
            ]
            for c in num_cols:
                wind_view[c] = pd.to_numeric(wind_view[c], errors="coerce")

            for c in ["wind_exists", "wind_is_sunk"]:
                wind_view[c] = wind_view[c].fillna(False).astype(bool)

            wind_view = st.data_editor(
                wind_view[["member_id", "name", "role", "enabled", "num"] + edit_cols],
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "member_id": st.column_config.TextColumn("member_id", disabled=True),
                    "name": st.column_config.TextColumn("Nome", disabled=True),
                    "role": st.column_config.TextColumn("Ruolo", disabled=True),
                    "enabled": st.column_config.CheckboxColumn("Utente attivo", disabled=True),
                    "num": st.column_config.NumberColumn("Molteplicità", disabled=True),
                    "wind_exists": st.column_config.CheckboxColumn("Eolico presente"),
                    "wind_installed_kw": st.column_config.NumberColumn("Taglia Eolico (kW)", min_value=0.0, step=0.5),
                    "wind_capex_eur_per_kw": st.column_config.NumberColumn("CAPEX (€/kW)", min_value=0.0, step=50.0),
                    "wind_opex_eur_per_kw_year": st.column_config.NumberColumn("OPEX (€/kW/anno)", min_value=0.0, step=2.0),
                    "wind_is_sunk": st.column_config.CheckboxColumn("Sunk cost (CAPEX=0)"),
                    "wind_life_years": st.column_config.NumberColumn("Vita utile (anni)", min_value=1, step=1),
                    "wind_major_repl_year": st.column_config.NumberColumn("Anno manut. straord.", min_value=0, step=1),
                    "wind_major_repl_eur_per_kw": st.column_config.NumberColumn("Costo manut. straord. (€/kW)", min_value=0.0, step=10.0),
                },
                key="dash_wind_table",
            )

            wind_out = w.copy()
            keep = [c for c in wind_out.columns if c not in edit_cols]
            wind_out = wind_out[keep].merge(wind_view[["member_id"] + edit_cols], on="member_id", how="left")

            # Normalizza: se eolico non presente -> forza taglia=0
            wind_out.loc[~wind_out["wind_exists"].astype(bool), "wind_installed_kw"] = 0.0

            st.session_state["econ_assets_wind_df"] = wind_out

            st.caption("Suggerimento: per applicare rapidamente lo stesso CAPEX/OPEX a tutti, imposta un valore in una riga e usa copia/incolla nel data editor.")

    # ----------------------------
    # BESS table (per utente)
    # ----------------------------
    with tab_bess:
        bess_df = st.session_state.get("econ_assets_bess_df")
        if bess_df is None or bess_df.empty:
            st.warning("Tabella BESS non disponibile.")
        else:
            u = st.session_state["econ_users_df"].copy()
            u["member_id"] = u["member_id"].astype(str)
            b = bess_df.copy()
            b["member_id"] = b["member_id"].astype(str)

            bess_view = (
                u[["member_id", "name", "role", "enabled", "num"]]
                .merge(b, on="member_id", how="left")
            )

            # dtype normalize
            for c in ["name", "role"]:
                if c in bess_view.columns:
                    bess_view[c] = bess_view[c].astype("string").fillna("")
            if "enabled" in bess_view.columns:
                bess_view["enabled"] = bess_view["enabled"].fillna(True).astype(bool)
            if "num" in bess_view.columns:
                bess_view["num"] = pd.to_numeric(bess_view["num"], errors="coerce").fillna(1).astype(int)

            # colonne BESS editabili (dashboard)
            edit_cols = [
                "bess_exists",
                "bess_initial_kwh",
                "bess_capex_eur_per_kwh",
                "bess_opex_pct_capex",
                "bess_opex_eur_per_kwh_year",
                "bess_life_years",
                "bess_replacement",
                # Flag usato per selezionare i membri candidati al sizing greedy
                "bess_optimize",
            ]
            for c in edit_cols:
                if c not in bess_view.columns:
                    bess_view[c] = np.nan

            # default: abilita ottimizzazione se la colonna è vuota
            if "bess_optimize" in bess_view.columns:
                bess_view["bess_optimize"] = bess_view["bess_optimize"].fillna(True).astype(bool)

            # default: se bess_initial_kwh > 0 -> exists
            bess_view["bess_initial_kwh"] = pd.to_numeric(bess_view["bess_initial_kwh"], errors="coerce")
            bess_view["bess_exists"] = bess_view["bess_exists"].fillna(bess_view["bess_initial_kwh"].fillna(0.0) > 0.0).astype(bool)

            # numeric coercion
            for c in ["bess_initial_kwh", "bess_capex_eur_per_kwh", "bess_opex_pct_capex", "bess_opex_eur_per_kwh_year", "bess_life_years"]:
                bess_view[c] = pd.to_numeric(bess_view[c], errors="coerce")

            # bool
            bess_view["bess_replacement"] = bess_view["bess_replacement"].fillna(True).astype(bool)

            bess_view = st.data_editor(
                bess_view[["member_id", "name", "role", "enabled", "num"] + edit_cols],
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "member_id": st.column_config.TextColumn("member_id", disabled=True),
                    "name": st.column_config.TextColumn("Nome", disabled=True),
                    "role": st.column_config.TextColumn("Ruolo", disabled=True),
                    "enabled": st.column_config.CheckboxColumn("Utente attivo", disabled=True),
                    "num": st.column_config.NumberColumn("Molteplicità", disabled=True),
                    "bess_exists": st.column_config.CheckboxColumn("BESS presente"),
                    "bess_initial_kwh": st.column_config.NumberColumn("Taglia BESS (kWh)", min_value=0.0, step=1.0),
                    "bess_capex_eur_per_kwh": st.column_config.NumberColumn("CAPEX (€/kWh)", min_value=0.0, step=10.0),
                    "bess_opex_pct_capex": st.column_config.NumberColumn("OPEX (% CAPEX) [0.02=2%]", min_value=0.0, step=0.005),
                    "bess_opex_eur_per_kwh_year": st.column_config.NumberColumn("OPEX (€/kWh/anno)", min_value=0.0, step=0.5),
                    "bess_life_years": st.column_config.NumberColumn("Vita utile (anni)", min_value=1, step=1),
                    "bess_replacement": st.column_config.CheckboxColumn("Replacement"),
                    "bess_optimize": st.column_config.CheckboxColumn("Candidato sizing", help="Se True, il membro può ricevere capacità BESS nel greedy"),
                },
                key="dash_bess_table",
            )

            # write-back preserving advanced columns (candidate sweep, optimize, etc.)
            b_out = b.copy()
            keep = [c for c in b_out.columns if c not in edit_cols and c != "bess_exists"]
            b_out = b_out[keep].merge(bess_view[["member_id"] + [c for c in edit_cols if c != "bess_exists"]], on="member_id", how="left")

            # se BESS non presente -> taglia=0
            exists_mask = bess_view.set_index("member_id")["bess_exists"].astype(bool)
            b_out = b_out.set_index("member_id")
            b_out.loc[~exists_mask.reindex(b_out.index).fillna(False).values, "bess_initial_kwh"] = 0.0
            b_out = b_out.reset_index()

            st.session_state["econ_assets_bess_df"] = b_out

            st.caption("Nota: in dashboard puoi usare sia OPEX %CAPEX sia OPEX €/kWh/anno. Se vuoi un solo metodo, lo rendiamo più semplice.")

    # ----------------------------
    # Tax simple (class-based)
    # ----------------------------
    with tab_tax:
        st.markdown("### Fiscalità (semplice)")
        tax_df = st.session_state.get("econ_tax_by_class_df")
        taxes_enabled = st.toggle("Tasse attive", value=bool(_safe_first(tax_df, "tax_enabled", False) or False), key="dash_tax_enabled")

        def _cls_rate(tdf, cls, default):
            try:
                y = tdf[tdf["user_class"].astype(str).str.strip().str.lower() == cls]
                if y.empty:
                    return default
                return float(y["tax_rate_effective"].iloc[0])
            except Exception:
                return default

        fam_default = _cls_rate(tax_df, "family", 0.26)
        bus_default = _cls_rate(tax_df, "business", 0.28)
        fam_rate = st.number_input("Aliquota family (effettiva)", min_value=0.0, max_value=1.0, value=float(fam_default), step=0.01, key="dash_tax_family")
        bus_rate = st.number_input("Aliquota business (effettiva)", min_value=0.0, max_value=1.0, value=float(bus_default), step=0.01, key="dash_tax_business")
        dep = st.checkbox("Ammortamenti attivi", value=bool(_safe_first(tax_df, "depreciation_enabled", True) if tax_df is not None else True), key="dash_tax_dep")
        carry = st.checkbox("Carryforward perdite", value=bool(_safe_first(tax_df, "allow_tax_loss_carryforward", True) if tax_df is not None else True), key="dash_tax_carry")
        carry_years = st.number_input("Anni carryforward", min_value=0, max_value=20, value=int(_safe_first(tax_df, "loss_carry_years", 5) or 5), step=1, key="dash_tax_carry_years")

        _set_tax_simple(
            enabled=taxes_enabled,
            family_rate=fam_rate,
            business_rate=bus_rate,
            depreciation_enabled=dep,
            carryforward_enabled=carry,
            carry_years=carry_years,
        )

    st.divider()
    
st.divider()
st.markdown("## 3B) Override fiscali per utente (opzionale)")
with st.expander("Override fiscali per utente (opzionale)", expanded=False):
    st.caption("Lascia i campi vuoti per applicare la fiscalità di classe. Usa gli override solo per eccezioni.")
    enable_over = st.toggle("Abilita editing override fiscali", value=False, key="dash_enable_tax_overrides")
    ov = st.session_state.get("econ_tax_overrides_df")
    if not enable_over:
        st.info("Override disattivati. Attiva il toggle per modificare.")
    elif ov is None or ov.empty:
        st.warning("Tabella override fiscali non disponibile.")
    else:
        ov2 = ov.copy()
        ov2["member_id"] = ov2["member_id"].astype(str)
        if "tax_enabled_override" in ov2.columns:
            ov2["tax_enabled_override"] = ov2["tax_enabled_override"].astype("object")
        if "tax_rate_override" in ov2.columns:
            ov2["tax_rate_override"] = pd.to_numeric(ov2["tax_rate_override"], errors="coerce")

        ov_edit = st.data_editor(
            ov2[["member_id", "tax_enabled_override", "tax_rate_override"]],
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "member_id": st.column_config.TextColumn("member_id", disabled=True),
                "tax_enabled_override": st.column_config.SelectboxColumn(
                    "Tax enabled override",
                    options=[None, True, False],
                    help="None = usa valore di classe; True/False = forza abilitazione tasse per il membro",
                ),
                "tax_rate_override": st.column_config.NumberColumn(
                    "Aliquota override (0-1)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    help="Vuoto (NaN) = usa aliquota di classe",
                ),
            },
            key="dash_tax_overrides_editor",
        )

        # write-back sulle colonne (mantieni altre colonne se esistono)
        ov_out = ov.copy()
        ov_out["member_id"] = ov_out["member_id"].astype(str)
        ov_out = ov_out.drop(columns=[c for c in ["tax_enabled_override", "tax_rate_override"] if c in ov_out.columns])
        ov_out = ov_out.merge(ov_edit, on="member_id", how="left")
        st.session_state["econ_tax_overrides_df"] = ov_out

st.markdown("## 4) Finanza (DCF)")
dcf = st.session_state["econ_dcf_df"].copy()
c1, c2, c3 = st.columns(3)
with c1:
    horizon = st.number_input(
        "Orizzonte (anni)",
        min_value=1,
        max_value=50,
        value=int(_safe_first(dcf, "horizon_years", 20) or 20),
        step=1,
    )
with c2:
    dr = st.number_input(
        "Discount rate",
        min_value=0.0,
        max_value=1.0,
        value=float(_safe_first(dcf, "discount_rate", 0.06) or 0.06),
        step=0.005,
    )
with c3:
    infl = st.number_input(
        "Inflazione",
        min_value=0.0,
        max_value=1.0,
        value=float(_safe_first(dcf, "inflation_rate", 0.02) or 0.02),
        step=0.005,
    )
c4, c5, c6 = st.columns(3)
with c4:
    esc_buy = st.number_input(
        "Escalation buy",
        min_value=-0.5,
        max_value=1.0,
        value=float(_safe_first(dcf, "escalation_buy", 0.02) or 0.02),
        step=0.01,
    )
with c5:
    esc_sell = st.number_input(
        "Escalation sell",
        min_value=-0.5,
        max_value=1.0,
        value=float(_safe_first(dcf, "escalation_sell", 0.02) or 0.02),
        step=0.01,
    )
with c6:
    esc_opex = st.number_input(
        "Escalation opex",
        min_value=-0.5,
        max_value=1.0,
        value=float(_safe_first(dcf, "escalation_opex", 0.02) or 0.02),
        step=0.01,
    )
st.markdown("### Detrazione fiscale (Bonus ristrutturazione)")
d1, d2, d3, d4 = st.columns(4)
with d1:
    det_enabled = st.checkbox(
        "Applica detrazione",
        value=bool(_safe_first(dcf, "detraction_enabled", False) or False),
        help="Se attivo, il beneficio viene aggiunto al cashflow per i primi N anni (ottica DCF).",
    )
with d2:
    det_rate = st.number_input(
        "Aliquota detrazione (0-1)",
        min_value=0.0,
        max_value=1.0,
        value=float(_safe_first(dcf, "detraction_rate", 0.0) or 0.0),
        step=0.01,
    )
with d3:
    det_cap = st.number_input(
        "Massimale spesa per unità (€)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=float(_safe_first(dcf, "detraction_cap_eur", 96000.0) or 96000.0),
        step=1000.0,
    )
with d4:
    det_years = st.number_input(
        "Quote (anni)",
        min_value=1,
        max_value=30,
        value=int(_safe_first(dcf, "detraction_years", 10) or 10),
        step=1,
    )

_set_dcf_basic(
    horizon=horizon,
    discount=dr,
    inflation=infl,
    esc_buy=esc_buy,
    esc_sell=esc_sell,
    esc_opex=esc_opex,
    det_enabled=det_enabled,
    det_rate=det_rate,
    det_cap=det_cap,
    det_years=det_years,
)

st.caption("Suggerimento: usa gli expander mirati solo quando necessario.")

return_hourly = st.checkbox("Includi breakdown orario (debug)", value=False)

run_eval = st.button("Calcola valutazione economica", type="primary")


def _make_out_dir() -> Tuple[str, Path]:
    """Crea la directory di output per un run economico.

    Returns
    -------
    Tuple[str, Path]
        ``(run_id, out_dir)`` dove ``run_id`` è un timestamp UTC in formato
        ``YYYYMMDDTHHMMSSZ`` e ``out_dir`` è ``economics/outputs/run_<run_id>``.

    Side effects
    ------------
    Crea la directory ``out_dir`` se non esiste.
    """
    run_id = _now_tag()
    d = ECON_OUTPUTS_DIR / f"run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return run_id, d



def _show_exception(e: Exception) -> None:
    # Dashboard unica: messaggio sintetico (niente expander di debug)
    st.error(str(e))


def _result_download_buttons(out_dir: Path) -> None:
    # compressione zip "rapida": in Streamlit è più semplice offrire singoli CSV
    # (l'utente può scaricare la cartella da filesystem locale). Qui forniamo
    # comunque download per i principali output.
    files = {
        "pnl_by_member.csv": out_dir / "pnl_by_member.csv",
        "cashflow_by_member.csv": out_dir / "cashflow_by_member.csv",
        "kpis_by_member.csv": out_dir / "kpis_by_member.csv",
        "pnl_total.csv": out_dir / "pnl_total.csv",
        "cashflow_total.csv": out_dir / "cashflow_total.csv",
        "kpis_total.csv": out_dir / "kpis_total.csv",
    }
    cols = st.columns(3)
    i = 0
    for label, path in files.items():
        if not path.exists():
            continue
        data = path.read_bytes()
        cols[i % 3].download_button(
            f"Scarica {label}",
            data=data,
            file_name=label,
            mime="text/csv",
        )
        i += 1


if run_eval:
    try:
        # Snapshot delle assunzioni correnti (Dashboard unica)
        assumptions = _snapshot_current_assumptions()

        # ---- Validazione UI: PZO richiesto per alcune modalita' (TIP=f(PZO), buy/sell indicizzati)
        def _needs_pzo(a: EconomicsAssumptions) -> bool:
            try:
                tip_mode = str(a.policy_cer.loc[0, "tip_mode"]).strip().lower()
            except Exception:
                tip_mode = "fixed"
            # TIP modes requiring PZO
            if tip_mode in ("pzo_function", "rse_decree", "rse", "rse_tip"):
                return True

            # TIAD ARERA (RSE-like) requires PZO in AUC mode (because Cpr is multiplied by PZO)
            try:
                tiad_mode = str(a.policy_cer.loc[0, "tiad_mode"]).strip().lower()
            except Exception:
                tiad_mode = "fixed"
            try:
                cacer_type = str(a.policy_cer.loc[0, "cacer_type"]).strip().upper()
            except Exception:
                cacer_type = "CER"
            if tiad_mode in ("rse_arera", "rse", "rse_tiad") and cacer_type == "AUC":
                return True
            if a.tariffs_buy is not None and not a.tariffs_buy.empty:
                m = a.tariffs_buy["buy_price_mode"].astype(str).str.strip().str.lower() == "pzo_plus_spread"
                if bool(m.any()):
                    return True
            if a.tariffs_sell is not None and not a.tariffs_sell.empty:
                sell_mode = a.tariffs_sell["sell_price_mode"].astype(str).str.strip().str.lower()
                sell_enabled = a.tariffs_sell.get("sell_enabled")
                if sell_enabled is None:
                    m2 = sell_mode.isin(["pzo", "pzo_minus_fee"])
                else:
                    m2 = sell_mode.isin(["pzo", "pzo_minus_fee"]) & sell_enabled.astype(bool)
                if bool(m2.any()):
                    return True
            return False

        if _needs_pzo(assumptions) and (assumptions.pzo_profile is None or (isinstance(assumptions.pzo_profile, pd.DataFrame) and assumptions.pzo_profile.empty)):
            st.error("PZO mancante: richiesto da TIP=f(PZO) o da una tariffa acquisto/vendita basata su PZO. Vai su '2) PZO' e carica o simula il profilo.")
            st.stop()

        with st.spinner("Calcolo economico in corso..."):
            res = evaluate_economics(energy_run, assumptions, return_hourly_breakdown=return_hourly)
        econ_run_id, out_dir = _make_out_dir()
        save_economic_outputs(out_dir, assumptions, res)

        # --- Registry economico + active run (robusto e riproducibile) ---
        created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        econ_scenario_slug = get_active_econ_scenario(ECON_ACTIVE_SCENARIO_TXT)

        # --- Audit trail (fingerprints) ---
        # NB: con la UX a scenari, il file scenario corretto è quello nello
        #     scenario selezionato. Manteniamo fallback a PATHS.scenario_json
        #     per compatibilità con la struttura legacy.
        try:
            scenario_json_path = _sp["scenario_json"] if _sp["scenario_json"].exists() else PATHS.scenario_json
        except Exception:
            scenario_json_path = PATHS.scenario_json
        energy_scenario_sha = scenario_file_content_fingerprint_sha256(scenario_json_path)
        energy_run_cfg_sha = None
        try:
            p_cfg = run_dir / "run_config.json"
            if p_cfg.exists():
                energy_run_cfg_sha = sha256_file(p_cfg)
        except Exception:
            energy_run_cfg_sha = None

        econ_scenario_content_sha = None
        try:
            if econ_scenario_slug:
                econ_scen_dir = ECON_SCENARIOS_DIR / str(econ_scenario_slug)
                econ_scenario_content_sha = get_econ_scenario_content_fingerprint_sha256(econ_scen_dir)
        except Exception:
            econ_scenario_content_sha = None

        # KPI totali (solo chiavi numeriche principali)
        kpi_total = {}
        try:
            if res.kpis_total is not None and not res.kpis_total.empty:
                row = res.kpis_total.iloc[0]
                for k in ["npv", "irr", "payback_year", "discounted_payback_year"]:
                    if k in row.index:
                        v = row[k]
                        kpi_total[k] = None if (v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)) else float(v)
        except Exception:
            kpi_total = {}

        def _rel(pth: Path) -> str:
            try:
                return str(pth.relative_to(SESSION_DIR))
            except Exception:
                return str(pth)

        record = {
            "run_id": str(econ_run_id),
            "created_at_utc": created_at_utc,
            "out_dir": _rel(out_dir),
            "energy_run_id": str(run_dir.name),
            "energy_run_dir": _rel(run_dir),
            "econ_scenario_slug": econ_scenario_slug,
            "energy_scenario_sha256": energy_scenario_sha,
            "energy_run_config_sha256": energy_run_cfg_sha,
            "econ_scenario_content_sha256": econ_scenario_content_sha,
            "kpi_total": kpi_total,
        }
        append_economic_run_record(ECON_RUNS_INDEX_JSONL, record)
        set_active_economic_run(ECON_ACTIVE_RUN_TXT, str(econ_run_id))

        st.session_state["econ_result"] = res
        st.session_state["econ_last_out_dir"] = str(out_dir)
        st.session_state["econ_last_run_id"] = str(econ_run_id)
        st.success(f"Valutazione completata. Output salvati in: {out_dir}")
    except Exception as e:
        _show_exception(e)


# =============================================================================
# Optimization (sweep) helpers
# =============================================================================


@st.cache_data(show_spinner=False)
def _cached_member_raw_series(
    member_id_int: int,
    period_t0: str,
    period_t1: str,
    tz: str,
    production_mode: str,
    selected_areas_json: str,
) -> Tuple[pd.Series, pd.Series]:
    """Carica e valida consumi e produzione per un membro (cache)."""
    period = build_period_config(pd.Timestamp(period_t0), pd.Timestamp(period_t1), tz=tz)
    prod_spec = ProductionSpec(enabled=True, mode=production_mode, selected_areas=json.loads(selected_areas_json))
    m = MemberSpec(
        member_id=str(member_id_int),
        name=str(member_id_int),
        consumption_csv=consumption_path(member_id_int),
        production_csv=production_path(member_id_int) if production_path(member_id_int).exists() else None,
        production_spec=prod_spec,
        battery=None,
    )
    data = load_and_validate_member(m, period)
    return data["P_load_15min_kW"], data["P_prod_hourly_kW"]


def _compute_member_hourly_with_bess(
    *,
    member_id: str,
    bess_kwh_agg: float,
    run_cfg: dict,
    period_t0: pd.Timestamp,
    period_t1: pd.Timestamp,
    tz: str,
    num: int,
) -> pd.DataFrame:
    """Ricalcola i flussi orari del membro per una taglia BESS aggregata.

    Interpreta bess_kwh_agg come taglia totale per il gruppo (num utenti identici).
    Il dispatch viene calcolato su taglia per-utente = bess_kwh_agg/num e poi
    i flussi energetici vengono scalati per num (coerente col bilanciamento).
    """
    mid_int = int(member_id)
    members = _members_from_run_cfg(run_cfg)
    m = next((x for x in members if str(x["member_id"]) == str(member_id)), None)
    if m is None:
        raise ValueError(f"Member {member_id} non trovato in run_config")

    production_mode = str(m.get("production_mode", "totale"))
    selected_areas = list(m.get("selected_areas", []) or [])

    # carica raw series (cache)
    P_load_15, P_prod_h = _cached_member_raw_series(
        mid_int,
        str(period_t0),
        str(period_t1),
        tz,
        production_mode,
        json.dumps(selected_areas),
    )

    # battery parameters from run_config (defaults if missing)
    bpar = _battery_defaults_from_run_cfg(run_cfg, str(member_id))
    per_user_kwh = float(bess_kwh_agg) / float(max(1, int(num)))
    battery = None
    if per_user_kwh > 0:
        battery = BatterySpec(
            capacity_kwh=float(per_user_kwh),
            dod=float(bpar["dod"]),
            roundtrip_eff=float(bpar["roundtrip_eff"]),
            derating_factor=float(bpar["derating_factor"]),
            init_soc_perc=float(bpar["init_soc_perc"]),
        )

    df_member = compute_member_energy_hourly(P_load_15min_kW=P_load_15, P_prod_hourly_kW=P_prod_h, battery=battery)

    # scale by multiplicity (SOC_perc is a state percentage, do not scale)
    if int(num) != 1:
        df_member = df_member.copy()
        for c in df_member.columns:
            if c == "SOC_perc":
                continue
            df_member[c] = df_member[c] * float(num)

    return df_member


def _energy_run_with_member_override(
    base: EnergyRunData,
    *,
    member_id: str,
    df_member_hourly: pd.DataFrame,
    enabled_member_ids: List[str],
) -> EnergyRunData:
    """Crea un nuovo EnergyRunData sostituendo i flussi di un membro e ricalcolando E_cond."""

    members_long = base.members_hourly_long.copy()
    members_long["member_id"] = members_long["member_id"].astype(str)

    # remove old rows for member
    others = members_long[members_long["member_id"] != str(member_id)].copy()

    # align columns
    new_rows = df_member_hourly.copy()
    new_rows.insert(0, "member_id", str(member_id))
    # ensure all columns exist
    target_cols = list(members_long.columns)
    for c in target_cols:
        if c not in new_rows.columns:
            new_rows[c] = 0.0
    new_rows = new_rows[target_cols]

    members_long_new = pd.concat([others, new_rows], axis=0)
    members_long_new = members_long_new.sort_index()

    # recompute CER aggregates (only on enabled members)
    tmp = members_long_new[members_long_new["member_id"].isin([str(x) for x in enabled_member_ids])]
    E_imm_tot = tmp.groupby(tmp.index)["E_imm_kWh"].sum().reindex(base.cer_hourly.index).fillna(0.0)
    E_prel_tot = tmp.groupby(tmp.index)["E_prel_kWh"].sum().reindex(base.cer_hourly.index).fillna(0.0)

    E_cond = pd.concat([E_imm_tot, E_prel_tot], axis=1).min(axis=1)
    E_export = E_imm_tot - E_cond
    E_import = E_prel_tot - E_cond

    cer_new = base.cer_hourly.copy()
    for col in ["E_imm_CER_kWh", "E_prel_CER_kWh", "E_cond_kWh", "E_export_kWh", "E_import_kWh"]:
        if col not in cer_new.columns:
            cer_new[col] = 0.0
    cer_new["E_imm_CER_kWh"] = E_imm_tot
    cer_new["E_prel_CER_kWh"] = E_prel_tot
    cer_new["E_cond_kWh"] = E_cond
    cer_new["E_export_kWh"] = E_export
    cer_new["E_import_kWh"] = E_import

    return EnergyRunData(
        run_dir=base.run_dir,
        period_tz=base.period_tz,
        t0=base.t0,
        t1=base.t1,
        cer_hourly=cer_new,
        members_hourly_long=members_long_new,
        run_config=base.run_config,
    )





# =============================================================================
# Greedy BESS sizing orchestration (NPV community)
# =============================================================================

with st.expander("Ottimizzazione BESS (Greedy)", expanded=False):
    users_df = st.session_state.get("econ_users_df")
    bess_df = st.session_state.get("econ_assets_bess_df")

    if users_df is None or bess_df is None:
        st.info("Carica una run energetica e inizializza le assunzioni economiche per abilitare il sizing.")
    else:
        u = users_df.copy()
        if "member_id" not in u.columns and u.index.name == "member_id":
            u = u.reset_index()

        if "enabled" in u.columns:
            enabled_member_ids = (
                u[u["enabled"].fillna(True).astype(bool)]["member_id"].astype(str).tolist()
            )
        else:
            enabled_member_ids = u["member_id"].astype(str).tolist()

        b = bess_df.copy()
        if "member_id" not in b.columns and b.index.name == "member_id":
            b = b.reset_index()

        if "bess_optimize" in b.columns:
            candidate_member_ids = (
                b[b["bess_optimize"].fillna(False).astype(bool)]["member_id"].astype(str).tolist()
            )
        else:
            candidate_member_ids = b["member_id"].astype(str).tolist()

        candidate_member_ids = [m for m in candidate_member_ids if m in enabled_member_ids]

        st.caption(
            f"Membri abilitati: {len(enabled_member_ids)} | candidati (bess_optimize): {len(candidate_member_ids)}"
        )

        step_default = 1.0
        try:
            if "bess_step_kwh" in b.columns and not b["bess_step_kwh"].dropna().empty:
                step_default = float(b["bess_step_kwh"].dropna().iloc[0])
        except Exception:
            step_default = 1.0

        step_kwh = st.number_input(
            "Step capacità (kWh, aggregati per membro)",
            min_value=0.5,
            value=float(step_default),
            step=0.5,
            key="bess_greedy_step",
        )
        max_total_kwh_in = st.number_input(
            "Capacità totale massima (kWh) - 0 = nessun limite",
            min_value=0.0,
            value=0.0,
            step=5.0,
            key="bess_greedy_max_total",
        )
        stop_on_negative = st.checkbox(
            "Ferma quando ΔNPV <= 0",
            value=True,
            key="bess_greedy_stop_negative",
        )

        selected_candidates = st.multiselect(
            "Candidati (opzionale)",
            options=candidate_member_ids,
            default=candidate_member_ids,
            key="bess_greedy_candidates",
        )

        do_run = st.button(
            "Esegui greedy sizing",
            disabled=(energy_run is None or len(selected_candidates) == 0),
            key="bess_greedy_run_btn",
        )

        if do_run:
            try:
                assumptions = _snapshot_current_assumptions()

                ts = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
                out_dir = ECON_OUTPUTS_DIR / "optimization" / f"bess_greedy_{ts}"

                cfg = GreedyConfig(
                    step_kwh=float(step_kwh),
                    max_total_kwh=None if float(max_total_kwh_in) <= 0 else float(max_total_kwh_in),
                    stop_on_negative=bool(stop_on_negative),
                )

                with st.spinner("Ottimizzazione greedy in corso..."):
                    opt = run_bess_greedy(
                        base_energy_run=energy_run,
                        base_assumptions=assumptions,
                        enabled_member_ids=enabled_member_ids,
                        candidate_member_ids=list(selected_candidates),
                        inputs_dir=INPUTS_DIR,
                        cfg=cfg,
                        out_dir=out_dir,
                    )

                st.session_state["econ_opt_results"] = opt
                st.success(f"Ottimizzazione completata. Output: {out_dir}")
            except Exception as e:
                _show_exception(e)

# =============================================================================
st.subheader("Risultati")

# -----------------------------------------------------------------------------
# Economic runs: registry + active run (ponte robusto fra run energetico e risultati economici)
# -----------------------------------------------------------------------------

_econ_registry = read_economic_run_registry(ECON_RUNS_INDEX_JSONL)
_econ_for_run = []
if _econ_registry:
    # Filtra per run energetico selezionato (coerenza dei risultati)
    _econ_for_run = [r for r in _econ_registry if str(r.get("energy_run_id", "")) == str(run_dir.name)]
    _econ_for_run = sorted(_econ_for_run, key=lambda r: r.get("created_at_utc", ""), reverse=True)

def _econ_out_dir(rec: dict) -> Path:
    pth = Path(str(rec.get("out_dir", "")))
    if not pth.is_absolute():
        pth = SESSION_DIR / pth
    return pth

def _econ_label(rec: dict) -> str:
    rid = str(rec.get("run_id", ""))
    scen = rec.get("econ_scenario_slug") or "-"
    fp = rec.get("econ_scenario_content_sha256")
    fp_s = (str(fp).strip()[:8]) if isinstance(fp, str) and fp.strip() else ""
    kpi = rec.get("kpi_total", {}) or {}
    npv = kpi.get("npv")
    try:
        npv_s = f"{float(npv):,.0f}"
    except Exception:
        npv_s = "n/d"
    extra = f" | scen_fp={fp_s}" if fp_s else ""
    return f"{rid} | scenario={scen}{extra} | NPV={npv_s}"

if _econ_for_run:
    with st.expander("Storico run economici (per questo run energetico)", expanded=False):
        labels = [_econ_label(r) for r in _econ_for_run]

        # default: active econ run -> primo
        default_idx_e = 0
        _active_econ_id = get_active_economic_run(ECON_ACTIVE_RUN_TXT)
        if _active_econ_id:
            for i, r in enumerate(_econ_for_run):
                if str(r.get("run_id")) == str(_active_econ_id):
                    default_idx_e = i
                    break

        sel_econ = st.selectbox("Run economico", options=labels, index=default_idx_e, key="econ_sel_saved_run")
        rec = _econ_for_run[labels.index(sel_econ)]
        out_dir_saved = _econ_out_dir(rec)

        c1, c2, c3, c4 = st.columns(4)
        kpi = rec.get("kpi_total", {}) or {}
        def _fmt(v):
            try:
                return f"{float(v):,.3f}"
            except Exception:
                return "n/d"
        c1.metric("NPV (tot)", _fmt(kpi.get("npv")))
        c2.metric("IRR (tot)", _fmt(kpi.get("irr")))
        c3.metric("Payback", _fmt(kpi.get("payback_year")))
        c4.metric("Discounted PB", _fmt(kpi.get("discounted_payback_year")))

        st.caption(f"Output: {out_dir_saved}")

        def _short_sha(v: Optional[str]) -> str:
            if isinstance(v, str) and v.strip():
                return v.strip()[:12]
            return "-"

        st.caption(
            "Audit trail | "
            f"energy_scenario={_short_sha(rec.get('energy_scenario_sha256'))} | "
            f"energy_run_cfg={_short_sha(rec.get('energy_run_config_sha256'))} | "
            f"econ_scenario={_short_sha(rec.get('econ_scenario_content_sha256'))}"
        )

        # Coerenza: evidenzia se lo scenario attuale è diverso da quello con cui
        # è stato generato il run economico (audit trail, non blocca l'uso).
        try:
            cur_energy_sha = scenario_file_content_fingerprint_sha256(PATHS.scenario_json)
            if cur_energy_sha and rec.get("energy_scenario_sha256") and cur_energy_sha != rec.get("energy_scenario_sha256"):
                st.warning(
                    "Lo **scenario energetico corrente** (scenario.json) è diverso da quello "
                    "registrato in questo run economico. Se hai modificato i membri/periodo "
                    "dopo aver generato il run energetico, questo è atteso."
                )
        except Exception:
            pass

        try:
            slug = rec.get("econ_scenario_slug")
            if slug:
                cur_econ_fp = get_econ_scenario_content_fingerprint_sha256(ECON_SCENARIOS_DIR / str(slug))
                if cur_econ_fp and rec.get("econ_scenario_content_sha256") and cur_econ_fp != rec.get("econ_scenario_content_sha256"):
                    st.warning(
                        "Lo **scenario economico salvato** associato a questo run (assumption pack) "
                        "risulta modificato rispetto al fingerprint registrato. "
                        "È possibile che lo scenario sia stato sovrascritto."
                    )
        except Exception:
            pass

        b_load, b_set = st.columns([1, 1])
        with b_load:
            do_load = st.button("Carica risultati di questo run", key="econ_load_saved_run")
        with b_set:
            do_set_active = st.button("Imposta come run economico attivo", key="econ_set_active_run")

        if do_set_active:
            set_active_economic_run(ECON_ACTIVE_RUN_TXT, str(rec.get("run_id")))
            st.success("Run economico attivo aggiornato.")

        if do_load:
            try:
                st.session_state["econ_result"] = load_economic_result(out_dir_saved)
                st.session_state["econ_last_out_dir"] = str(out_dir_saved)
                st.session_state["econ_last_run_id"] = str(rec.get("run_id"))
                set_active_economic_run(ECON_ACTIVE_RUN_TXT, str(rec.get("run_id")))
                st.success("Risultati caricati da disco.")
            except Exception as e:
                st.error(str(e))

# Auto-load: se esiste un run economico attivo per questo run energetico e non ci sono risultati in memoria
if st.session_state.get("econ_result") is None:
    _active_econ_id = get_active_economic_run(ECON_ACTIVE_RUN_TXT)
    _autoload_key = f"{run_dir.name}::{_active_econ_id}"
    if _active_econ_id and st.session_state.get("econ_autoload_key") != _autoload_key:
        rec = next((r for r in _econ_for_run if str(r.get("run_id")) == str(_active_econ_id)), None)
        if rec is not None:
            try:
                out_dir_saved = _econ_out_dir(rec)
                st.session_state["econ_result"] = load_economic_result(out_dir_saved)
                st.session_state["econ_last_out_dir"] = str(out_dir_saved)
                st.session_state["econ_last_run_id"] = str(rec.get("run_id"))
                st.session_state["econ_autoload_key"] = _autoload_key
            except Exception:
                # Non bloccare la pagina: se l'output è incompleto, l'utente può comunque ricalcolare.
                st.session_state["econ_autoload_key"] = _autoload_key


res = st.session_state.get("econ_result")
out_dir_str = st.session_state.get("econ_last_out_dir")

if res is None:
    st.info("Esegui 'Calcola valutazione economica' per visualizzare i risultati.")
else:
    tabs = st.tabs(["KPI", "Conto economico", "Cash flow", "Debug", "Download"])

    with tabs[0]:
        st.markdown("**KPI per membro**")
        st.dataframe(res.kpis_by_member, use_container_width=True)
        st.markdown("**KPI totale CER**")
        st.dataframe(res.kpis_total, use_container_width=True)

    with tabs[1]:
        st.markdown("**Conto economico (PnL) per membro**")
        members = sorted({idx[0] for idx in res.pnl_by_member.index.tolist()})
        sel = st.selectbox("Seleziona membro", options=members, key="econ_pnl_member")
        df = res.pnl_by_member.xs(sel, level=0).copy()
        st.dataframe(df, use_container_width=True)
        st.markdown("**PnL totale**")
        st.dataframe(res.pnl_total, use_container_width=True)

    with tabs[2]:
        st.markdown("**Cash flow per membro**")
        members = sorted({idx[0] for idx in res.cashflow_by_member.index.tolist()})
        sel = st.selectbox("Seleziona membro", options=members, key="econ_cf_member")
        df = res.cashflow_by_member.xs(sel, level=0).copy()
        st.dataframe(df, use_container_width=True)
        st.markdown("**Cash flow totale**")
        st.dataframe(res.cashflow_total, use_container_width=True)

    with tabs[3]:
        if res.hourly_breakdown is None:
            st.info("Breakdown orario non calcolato. Abilita l'opzione 'debug' prima di eseguire il calcolo.")
        else:
            st.dataframe(res.hourly_breakdown.head(200), use_container_width=True)
            st.caption("Mostrati i primi 200 record. Il breakdown completo è salvabile via output su disco.")

    with tabs[4]:
        if out_dir_str:
            out_dir = Path(out_dir_str)
            _result_download_buttons(out_dir)
            st.caption(f"Directory output: {out_dir}")


# =============================================================================
# Optimization results rendering
# =============================================================================


opt = st.session_state.get("econ_opt_results")
if opt is not None:
    st.subheader("Ottimizzazione batteria")
    st.markdown("**Sintesi**")
    st.dataframe(opt["summary"], use_container_width=True)

    st.markdown("**Curve NPV(kWh)**")
    curve = opt["curve"].copy()
    if not curve.empty:
        fig = px.line(curve, x="bess_kwh", y="npv", color="member_id", markers=True)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Extra charts for greedy runs
    if "alloc_trace_df" in opt and isinstance(opt["alloc_trace_df"], pd.DataFrame) and not opt["alloc_trace_df"].empty:
        tr = opt["alloc_trace_df"].copy()
        st.markdown("**Marginale ΔNPV per iterazione**")
        fig2 = px.bar(tr, x="iter", y="dNPV_CER", color="member_id")
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("**NPV cumulato (VAN comune)**")
        fig3 = px.line(tr, x="iter", y="NPV_CER", markers=True)
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)

    st.caption(f"Directory output ottimizzazione: {opt['out_dir']}")
