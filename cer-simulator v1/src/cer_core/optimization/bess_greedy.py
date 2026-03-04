# -*- coding: utf-8 -*-
"""cer_core.optimization.bess_greedy

Greedy incremental battery sizing (BESS) at community level.

Goal
----
Allocate battery energy capacity (kWh) to individual members by iteratively adding
fixed capacity steps to the member that yields the largest *marginal* increase in
community NPV (VAN comune), keeping all other inputs fixed.

Inputs kept fixed
-----------------
- Energy run inputs (consumption/production time-series, members) are taken from a
  selected energy run directory (EnergyRunData).
- Economic assumptions (tariffs, policy, CAPEX/OPEX for other assets, fiscal/DCF)
  are frozen from a selected economic scenario/run.

What varies
-----------
Only the BESS size for one member at a time:
- Member hourly flows are recomputed with the new BESS size.
- The energy run is overridden for that member and CER aggregates (E_cond, import,
  export) are recomputed.
- Economic evaluation is rerun with the same assumptions except
  ``assets_bess.bess_initial_kwh`` for that member.

Notes
-----
This is a *heuristic* allocator (greedy). It captures interaction/saturation
because each trial is evaluated on the *current* (already allocated) run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cer_core.economics.economic_model import (
    EnergyRunData,
    EconomicsAssumptions,
    EconomicsResult,
    evaluate_economics,
)

from cer_core.optimization.bess_helpers import (
    compute_member_hourly_with_bess,
    energy_run_with_member_override,
)


@dataclass(frozen=True)
class GreedyConfig:
    step_kwh: float
    max_total_kwh: Optional[float] = None
    stop_on_negative: bool = True


def _kpis_total(res: EconomicsResult) -> Dict[str, float]:
    """Extract a compact KPI dict from EconomicsResult.kpis_total."""
    out: Dict[str, float] = {}
    try:
        if res.kpis_total is None or res.kpis_total.empty:
            return out
        row = res.kpis_total.iloc[0]
        for k in ["npv", "irr", "payback_year", "discounted_payback_year"]:
            if k in row.index:
                v = row[k]
                if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                    continue
                out[k] = float(v)
    except Exception:
        return out
    return out


def _npv_total(res: EconomicsResult) -> float:
    k = _kpis_total(res)
    return float(k.get("npv", np.nan))


def _ensure_assets_bess_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure assets_bess is indexed by member_id (string) AND keeps member_id column."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # Case 1: already indexed by member_id
    if out.index.name == "member_id":
        out.index = out.index.astype(str)
        if "member_id" not in out.columns:
            out["member_id"] = out.index.astype(str)
        return out

    # Case 2: member_id column exists -> set index but keep the column
    if "member_id" in out.columns:
        out["member_id"] = out["member_id"].astype(str)
        out = out.set_index("member_id", drop=False)
        return out

    return out


def run_bess_greedy(
    *,
    base_energy_run: EnergyRunData,
    base_assumptions: EconomicsAssumptions,
    enabled_member_ids: List[str],
    candidate_member_ids: List[str],
    inputs_dir: Path,
    cfg: GreedyConfig,
    out_dir: Optional[Path] = None,
    return_final_objects: bool = False,
) -> Dict[str, Any]:
    """Run greedy BESS sizing.

    Parameters
    ----------
    base_energy_run:
        Energy run selected by the user.
    base_assumptions:
        Frozen economic assumptions selected/edited by the user.
    enabled_member_ids:
        Members enabled in the economics UI (used to compute CER aggregates).
    candidate_member_ids:
        Subset of enabled members that participate in sizing.
    inputs_dir:
        Energy scenario inputs folder for member consumption/production csv.
    cfg:
        Greedy algorithm configuration.
    out_dir:
        If provided, results are persisted (csv/json) in this folder.
    return_final_objects:
        If True, also return heavy objects (final EnergyRunData and EconomicsResult)
        inside the output dict for drill-down/debug.

    Returns
    -------
    dict
        econ_opt_results (UI-friendly) containing alloc_trace_df, member_summary_df,
        kpis_base/opt and stop reason.
    """

    enabled_member_ids = [str(m) for m in enabled_member_ids]
    enabled_set = {str(m) for m in enabled_member_ids}
    candidate_member_ids = [str(m) for m in candidate_member_ids if str(m) in enabled_set]

    if cfg.step_kwh <= 0:
        raise ValueError("step_kwh deve essere > 0")

    # --- baseline state (frozen inputs)
    assets_bess_current = _ensure_assets_bess_index(base_assumptions.assets_bess).copy()
    assumptions_current = replace(base_assumptions, assets_bess=assets_bess_current)

    run_current = base_energy_run
    econ_current = evaluate_economics(run_current, assumptions_current)
    npv_current = _npv_total(econ_current)

    kpis_base = _kpis_total(econ_current)
    try:
        econd_base = float(run_current.cer_hourly["E_cond_kWh"].sum())
    except Exception:
        econd_base = np.nan

    # capacities (aggregated kWh per member)
    C: Dict[str, float] = {m: 0.0 for m in candidate_member_ids}

    alloc_rows: List[Dict[str, Any]] = []
    stop_reason: Optional[str] = None

    def total_kwh() -> float:
        return float(sum(C.values()))

    iter_idx = 0
    while True:
        iter_idx += 1

        if cfg.max_total_kwh is not None and total_kwh() + cfg.step_kwh > cfg.max_total_kwh:
            stop_reason = "max_total_kwh"
            break

        best: Optional[Tuple[str, float, EnergyRunData, EconomicsResult, pd.DataFrame, pd.DataFrame, float, float]] = None
        # (member_id, dnpv, run_trial, econ_trial, assets_bess_trial, assumptions_trial, npv_trial, econd_trial)

        for mid in candidate_member_ids:
            C_trial = C[mid] + cfg.step_kwh

            df_mid_trial = compute_member_hourly_with_bess(
                member_id=mid,
                bess_kwh_agg=float(C_trial),
                run_cfg=base_energy_run.run_config,
                period_t0=base_energy_run.t0,
                period_t1=base_energy_run.t1,
                tz=base_energy_run.period_tz,
                inputs_dir=inputs_dir,
            )

            run_trial = energy_run_with_member_override(
                run_current,
                member_id=mid,
                df_member_hourly=df_mid_trial,
                enabled_member_ids=enabled_member_ids,
            )

            # update only BESS size for mid (keep everything else frozen)
            assets_bess_trial = assets_bess_current.copy()
            if mid not in assets_bess_trial.index:
                raise ValueError(f"assets_bess non contiene member_id={mid}. Verifica che assets_bess sia indicizzato per member_id.")
            assets_bess_trial.loc[mid, "bess_initial_kwh"] = float(C_trial)
            assumptions_trial = replace(base_assumptions, assets_bess=assets_bess_trial)

            econ_trial = evaluate_economics(run_trial, assumptions_trial)
            npv_trial = _npv_total(econ_trial)
            dnpv = float(npv_trial - npv_current)

            try:
                econd_trial = float(run_trial.cer_hourly["E_cond_kWh"].sum())
            except Exception:
                econd_trial = np.nan

            if best is None or dnpv > best[1]:
                best = (mid, dnpv, run_trial, econ_trial, assets_bess_trial, assumptions_trial, float(npv_trial), float(econd_trial))

        if best is None:
            stop_reason = "no_feasible_candidate"
            break

        mid_best, dnpv_best, run_best, econ_best, assets_bess_best, assumptions_best, npv_best, econd_best = best

        if cfg.stop_on_negative and dnpv_best <= 0:
            stop_reason = "negative_marginal"
            break

        # accept
        C_before = C[mid_best]
        C[mid_best] = C_before + cfg.step_kwh

        run_current = run_best
        econ_current = econ_best
        assets_bess_current = assets_bess_best
        assumptions_current = assumptions_best
        npv_current = float(npv_best)

        alloc_rows.append(
            {
                "iter": int(iter_idx),
                "member_id": str(mid_best),
                "delta_kwh": float(cfg.step_kwh),
                "C_before_kwh": float(C_before),
                "C_after_kwh": float(C[mid_best]),
                "dNPV_CER": float(dnpv_best),
                "NPV_CER": float(npv_current),
                "Econd_kwh": float(econd_best),
                "dEcond_kwh": float(econd_best - econd_base) if np.isfinite(econd_base) and np.isfinite(econd_best) else np.nan,
            }
        )

    alloc_trace_df = pd.DataFrame(alloc_rows)

    # summary per member
    member_summary_df = pd.DataFrame({
        "member_id": list(C.keys()),
        "C_star_kwh": [float(C[m]) for m in C.keys()],
    })
    if not alloc_trace_df.empty:
        contrib = (
            alloc_trace_df.groupby("member_id")["dNPV_CER"].sum().reset_index().rename(columns={"dNPV_CER": "dNPV_CER_contrib"})
        )
        member_summary_df = member_summary_df.merge(contrib, on="member_id", how="left")
    else:
        member_summary_df["dNPV_CER_contrib"] = 0.0

    kpis_opt = _kpis_total(econ_current)

    # curve compatible with existing renderer (stepwise points)
    if not alloc_trace_df.empty:
        curve_df = alloc_trace_df.rename(columns={"C_after_kwh": "bess_kwh", "NPV_CER": "npv"})[["iter", "member_id", "bess_kwh", "npv"]].copy()
    else:
        curve_df = pd.DataFrame(columns=["iter", "member_id", "bess_kwh", "npv"])

    econ_opt_results: Dict[str, Any] = {
        "summary": member_summary_df,
        "curve": curve_df,
        "out_dir": str(out_dir) if out_dir is not None else "",
        "method": "bess_greedy",
        "objective": "NPV_CER",
        "stop_reason": stop_reason,
        "alloc_trace_df": alloc_trace_df,
        "member_summary_df": member_summary_df,
        "capacity_map": C,
        "kpis_base": kpis_base,
        "kpis_opt": kpis_opt,
        "meta": {
            "step_kwh": float(cfg.step_kwh),
            "max_total_kwh": None if cfg.max_total_kwh is None else float(cfg.max_total_kwh),
            "stop_on_negative": bool(cfg.stop_on_negative),
            "enabled_member_ids": enabled_member_ids,
            "candidate_member_ids": candidate_member_ids,
            "energy_run_id": str(getattr(base_energy_run.run_dir, "name", "")),
            "energy_run_dir": str(getattr(base_energy_run, "run_dir", "")),
        },
    }

    if return_final_objects:
        econ_opt_results["final_energy_run"] = run_current
        econ_opt_results["final_econ"] = econ_current

    if out_dir is not None:
        _save_outputs(Path(out_dir), econ_opt_results)

    return econ_opt_results


def _save_outputs(out_dir: Path, econ_opt_results: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    alloc = econ_opt_results.get("alloc_trace_df")
    if isinstance(alloc, pd.DataFrame):
        alloc.to_csv(out_dir / "alloc_trace.csv", index=False)

    summ = econ_opt_results.get("member_summary_df")
    if isinstance(summ, pd.DataFrame):
        summ.to_csv(out_dir / "member_summary.csv", index=False)

    curve = econ_opt_results.get("curve")
    if isinstance(curve, pd.DataFrame):
        curve.to_csv(out_dir / "curve.csv", index=False)

    # lightweight json
    meta = {
        "method": econ_opt_results.get("method"),
        "objective": econ_opt_results.get("objective"),
        "stop_reason": econ_opt_results.get("stop_reason"),
        "meta": econ_opt_results.get("meta", {}),
        "kpis_base": econ_opt_results.get("kpis_base", {}),
        "kpis_opt": econ_opt_results.get("kpis_opt", {}),
        "capacity_map": econ_opt_results.get("capacity_map", {}),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
