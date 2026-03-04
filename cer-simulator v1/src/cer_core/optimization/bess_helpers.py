# -*- coding: utf-8 -*-
"""cer_core.optimization.bess_helpers

Helper utilities for battery sizing/what-if analysis.

These functions are extracted from the Streamlit page logic to make them reusable
from core optimization algorithms (no Streamlit dependency).

Key idea
--------
Given a fixed energy run (loads/production profiles and membership), we can
"ceteris paribus" vary only the BESS size for a single member, recompute that
member's hourly flows (dispatch at 15-min, aggregated to hourly), then rebuild
CER aggregates (E_cond, import/export) by overriding only that member.

All paths are made explicit via ``inputs_dir`` so that the caller can point to
any energy scenario inputs folder.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from cer_core.bilanciamento.bilanciamento_energetico import (
    BatterySpec,
    MemberSpec,
    ProductionSpec,
    build_period_config,
    compute_member_energy_hourly,
    load_and_validate_member,
)
from cer_core.economics.economic_model import EnergyRunData


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def consumption_path(inputs_dir: Path, member_id: int) -> Path:
    return Path(inputs_dir) / f"member_{member_id}" / "consumption_15min.csv"


def production_path(inputs_dir: Path, member_id: int) -> Path:
    return Path(inputs_dir) / f"member_{member_id}" / "production_hourly.csv"


# -----------------------------------------------------------------------------
# Run-config helpers
# -----------------------------------------------------------------------------

def members_from_run_cfg(run_cfg: dict) -> List[dict]:
    """Normalize members list from a run_config dict."""
    members = run_cfg.get("members", []) or []
    out: List[dict] = []
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


def battery_defaults_from_run_cfg(run_cfg: dict, member_id: str) -> Dict[str, float]:
    """Extract battery parameter defaults (DoD, efficiencies, etc.) for a member."""
    defaults: Dict[str, float] = {
        "dod": 0.8,
        "roundtrip_eff": 0.9,
        "derating_factor": 0.0,
        "init_soc_perc": 0.2,
    }
    for m in members_from_run_cfg(run_cfg):
        if str(m.get("member_id")) != str(member_id):
            continue
        batt = m.get("battery")
        if isinstance(batt, dict):
            for k in list(defaults.keys()):
                if k in batt and batt[k] not in (None, ""):
                    defaults[k] = float(batt[k])
        return defaults
    return defaults


# -----------------------------------------------------------------------------
# Series loading (cached)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cached_member_raw_series(
    inputs_dir_str: str,
    member_id_int: int,
    period_t0_iso: str,
    period_t1_iso: str,
    tz: str,
    production_mode: str,
    selected_areas_json: str,
) -> Tuple[pd.Series, pd.Series]:
    """Load and validate member consumption/production series (in-process cache)."""
    inputs_dir = Path(inputs_dir_str)
    period = build_period_config(pd.Timestamp(period_t0_iso), pd.Timestamp(period_t1_iso), tz=tz)

    try:
        selected_areas = json.loads(selected_areas_json)
    except Exception:
        selected_areas = []

    prod_spec = ProductionSpec(enabled=True, mode=str(production_mode), selected_areas=selected_areas)

    m = MemberSpec(
        member_id=str(member_id_int),
        name=str(member_id_int),
        consumption_csv=consumption_path(inputs_dir, member_id_int),
        production_csv=production_path(inputs_dir, member_id_int)
        if production_path(inputs_dir, member_id_int).exists()
        else None,
        production_spec=prod_spec,
        battery=None,
    )

    data = load_and_validate_member(m, period)
    return data["P_load_15min_kW"], data["P_prod_hourly_kW"]


# -----------------------------------------------------------------------------
# Member recomputation + run override
# -----------------------------------------------------------------------------

def compute_member_hourly_with_bess(
    *,
    member_id: str,
    bess_kwh_agg: float,
    run_cfg: dict,
    period_t0: pd.Timestamp,
    period_t1: pd.Timestamp,
    tz: str,
    inputs_dir: Path,
) -> pd.DataFrame:
    """Recompute hourly flows for a member, varying only BESS size.

    Parameters
    ----------
    member_id:
        Member id as string.
    bess_kwh_agg:
        Aggregated BESS size for the member (already scaled by multiplicity `num`).
    run_cfg:
        Energy run configuration dict (from run_config.json).
    period_t0, period_t1, tz:
        Simulation period and timezone (from EnergyRunData).
    inputs_dir:
        Energy scenario inputs directory containing member_{id}/consumption_15min.csv,
        member_{id}/production_hourly.csv.

    Notes
    -----
    The dispatch is computed on per-user size = bess_kwh_agg / num and then all
    energy flows are scaled by num (SOC_perc is not scaled).
    """

    mid_int = int(member_id)

    members = members_from_run_cfg(run_cfg)
    m = next((x for x in members if str(x.get("member_id")) == str(member_id)), None)
    if m is None:
        raise ValueError(f"Member {member_id} non trovato in run_config")

    num = int(m.get("num", 1) or 1)
    production_mode = str(m.get("production_mode", "totale"))
    selected_areas = list(m.get("selected_areas", []) or [])

    # load raw series (cached)
    P_load_15, P_prod_h = _cached_member_raw_series(
        str(Path(inputs_dir)),
        mid_int,
        str(period_t0),
        str(period_t1),
        tz,
        production_mode,
        json.dumps(selected_areas),
    )

    # battery parameters from run_config
    bpar = battery_defaults_from_run_cfg(run_cfg, str(member_id))

    per_user_kwh = float(bess_kwh_agg) / float(max(1, num))
    battery = None
    if per_user_kwh > 0:
        battery = BatterySpec(
            capacity_kwh=float(per_user_kwh),
            dod=float(bpar["dod"]),
            roundtrip_eff=float(bpar["roundtrip_eff"]),
            derating_factor=float(bpar["derating_factor"]),
            init_soc_perc=float(bpar["init_soc_perc"]),
        )

    df_member = compute_member_energy_hourly(
        P_load_15min_kW=P_load_15,
        P_prod_hourly_kW=P_prod_h,
        battery=battery,
    )

    # scale by multiplicity (SOC_perc is a state percentage)
    if num != 1:
        df_member = df_member.copy()
        for c in df_member.columns:
            if c == "SOC_perc":
                continue
            df_member[c] = df_member[c] * float(num)

    return df_member


def energy_run_with_member_override(
    base: EnergyRunData,
    *,
    member_id: str,
    df_member_hourly: pd.DataFrame,
    enabled_member_ids: List[str],
) -> EnergyRunData:
    """Return a new EnergyRunData overriding one member and recomputing CER aggregates."""

    members_long = base.members_hourly_long.copy()
    members_long["member_id"] = members_long["member_id"].astype(str)

    # remove old rows for member
    others = members_long[members_long["member_id"] != str(member_id)].copy()

    # align columns
    new_rows = df_member_hourly.copy()
    new_rows.insert(0, "member_id", str(member_id))

    target_cols = list(members_long.columns)
    for c in target_cols:
        if c not in new_rows.columns:
            new_rows[c] = 0.0
    new_rows = new_rows[target_cols]

    members_long_new = pd.concat([others, new_rows], axis=0)
    members_long_new = members_long_new.sort_index()

    # recompute CER aggregates using only enabled members
    enabled_set = {str(x) for x in enabled_member_ids}
    tmp = members_long_new[members_long_new["member_id"].isin(enabled_set)]

    # Sum by timestamp
    E_imm_tot = (
        tmp.groupby(tmp.index)["E_imm_kWh"].sum().reindex(base.cer_hourly.index).fillna(0.0)
    )
    E_prel_tot = (
        tmp.groupby(tmp.index)["E_prel_kWh"].sum().reindex(base.cer_hourly.index).fillna(0.0)
    )

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
