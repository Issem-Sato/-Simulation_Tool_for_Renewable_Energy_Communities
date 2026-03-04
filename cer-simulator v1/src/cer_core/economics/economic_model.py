from __future__ import annotations

"""cer_core.economics.economic_model

Motore di valutazione economico-finanziaria della Comunità Energetica Rinnovabile (CER).

Questo modulo trasforma gli output del *bilanciamento energetico* in risultati
economici (conto economico, flussi di cassa e KPI) a livello di singolo membro e
aggregati a livello di comunità.

L'interfaccia Streamlit (pagina ``4_Valutazione_Economica.py``) costruisce e
valida le tabelle di input (tariffe, parametri DCF, fiscalità, asset) e invoca
``evaluate_economics`` passandole tramite la dataclass :class:`EconomicsAssumptions`.

Input principali (energia)
--------------------------
Gli input energetici sono letti da ``load_energy_run`` e derivano dalla pagina di
bilanciamento (non modificata in questo scope):

* ``cer_hourly.csv``: serie oraria di comunità (almeno ``E_cond_kWh``).
* ``members_hourly_long.csv``: serie oraria per membro (long) con
  ``member_id``, ``E_prel_kWh`` (prelievo) e ``E_imm_kWh`` (immissione).

Convenzioni e invarianti
-----------------------
* Unità energetiche: **kWh**.
* Unità economiche: **€/kWh** per prezzi/ricavi/costi variabili; **€** per fee annue
  e CAPEX.
* Segni:
  - ``E_prel_kWh`` e ``E_imm_kWh`` sono attesi **non negativi**.
  - i costi (es. acquisto energia) sono trattati come valori positivi nella colonna
    di costo e sottratti nel conto economico.
* Timezone:
  - I CSV energetici vengono letti con ``utc=True`` (timestamp interpretati/convertiti
    in UTC). La classificazione F1/F2/F3 usa conversione a ``Europe/Rome``.
  - Se i CSV contengono timestamp *naive*, questi vengono interpretati come UTC.
    Questo comportamento è intenzionale per compatibilità con il layer di bilanciamento;
    eventuali conversioni/normalizzazioni devono essere gestite a monte.

Side effects
------------
* ``save_economic_outputs`` scrive su filesystem (CSV + ``manifest.json``).
  Le scritture sono *best-effort atomiche* (scrittura su file temporaneo + replace).

Compatibilità
-------------
* Nomi file di output e colonne CSV sono considerati parte del contratto con la UI.
  Questo modulo deve mantenere retro-compatibilità salvo esplicita deprecazione.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from cer_core import get_config


# =============================================================================
# Dataclasses (I/O)
# =============================================================================


@dataclass(frozen=True)
class EnergyRunData:
    """Contenitore dei dati di un run energetico."""

    run_dir: Path
    period_tz: str
    t0: pd.Timestamp
    t1: pd.Timestamp
    cer_hourly: pd.DataFrame
    members_hourly_long: pd.DataFrame
    run_config: dict


@dataclass(frozen=True)
class EconomicsAssumptions:
    """Assunzioni consolidate (serializzabili) per un run economico."""

    policy_cer: pd.DataFrame
    users: pd.DataFrame
    tariffs_buy: pd.DataFrame
    tariffs_sell: pd.DataFrame
    assets_pv: pd.DataFrame
    assets_wind: Optional[pd.DataFrame]
    assets_bess: pd.DataFrame
    tax_by_class: pd.DataFrame
    tax_overrides: Optional[pd.DataFrame]
    dcf_params: pd.DataFrame
    # profili opzionali (wide: time index; colonne per membro o un'unica colonna)
    pzo_profile: Optional[pd.DataFrame] = None
    tip_profile: Optional[pd.DataFrame] = None
    tiad_profile: Optional[pd.DataFrame] = None
    buy_profiles: Optional[pd.DataFrame] = None
    sell_profiles: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class EconomicsResult:
    """Risultati economici."""

    pnl_by_member: pd.DataFrame
    cashflow_by_member: pd.DataFrame
    kpis_by_member: pd.DataFrame
    pnl_total: pd.DataFrame
    cashflow_total: pd.DataFrame
    kpis_total: pd.DataFrame
    # opzionale: breakdown orario (utile per debug, di default None)
    hourly_breakdown: Optional[pd.DataFrame] = None


# =============================================================================
# Run discovery & loading
# =============================================================================


def list_energy_runs(session_dir: Path) -> List[Path]:
    """Elenca i run energetici disponibili in una sessione.

    Parameters
    ----------
    session_dir:
        Directory della sessione (root) contenente ``bilanciamento/outputs``.

    Returns
    -------
    list[pathlib.Path]
        Lista di directory ``run_<timestamp>`` ordinate dalla più recente.
    """
    out_root = session_dir / "bilanciamento" / "outputs"
    if not out_root.exists():
        return []
    runs = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    # ordinamento: prima i più recenti (nome contiene timestamp)
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def load_energy_run(run_dir: Path) -> EnergyRunData:
    """Carica un run energetico (CSV + ``run_config.json``).

    La funzione implementa un *contratto minimo* sugli output del bilanciamento.
    I timestamp dei CSV sono letti e convertiti in UTC (``utc=True``).

    Parameters
    ----------
    run_dir:
        Directory del run energetico contenente i CSV e il file di configurazione.

    Returns
    -------
    EnergyRunData
        Oggetto immutabile con DataFrame e metadati del periodo.

    Raises
    ------
    FileNotFoundError
        Se mancano uno o più file obbligatori.
    ValueError
        Se i DataFrame non rispettano lo schema minimo (colonne richieste).
    """
    p_cer = run_dir / "cer_hourly.csv"
    p_m = run_dir / "members_hourly_long.csv"
    p_cfg = run_dir / "run_config.json"
    p_meta = run_dir / "period_meta.csv"

    if not p_cer.exists() or not p_m.exists() or not p_cfg.exists():
        raise FileNotFoundError(
            f"Run energetico incompleto in {run_dir}. Attesi: cer_hourly.csv, "
            "members_hourly_long.csv, run_config.json"
        )

    cer = _read_time_indexed_csv(p_cer)
    members_long = _read_time_indexed_csv(p_m)
    run_cfg = json.loads(p_cfg.read_text(encoding="utf-8"))

    # metadata periodo
    if p_meta.exists():
        meta = pd.read_csv(p_meta)
        period_tz = str(meta.loc[0, "tz"]) if "tz" in meta.columns else "UTC"
        t0 = pd.Timestamp(meta.loc[0, "t0"])
        t1 = pd.Timestamp(meta.loc[0, "t1"])
        if t0.tzinfo is None:
            t0 = t0.tz_localize(period_tz)
        if t1.tzinfo is None:
            t1 = t1.tz_localize(period_tz)
    else:
        # fallback: deriviamo da index
        period_tz = "UTC"
        t0 = cer.index.min()
        t1 = cer.index.max()

    _validate_energy_run_frames(cer, members_long)

    return EnergyRunData(
        run_dir=run_dir,
        period_tz=period_tz,
        t0=t0,
        t1=t1,
        cer_hourly=cer,
        members_hourly_long=members_long,
        run_config=run_cfg,
    )


# =============================================================================
# Core evaluation
# =============================================================================


def evaluate_economics(
    energy_run: EnergyRunData,
    assumptions: EconomicsAssumptions,
    *,
    # opzionale: includere breakdown orario nei risultati (pesante)
    return_hourly_breakdown: bool = False,

) -> EconomicsResult:
    """Valuta economicamente un run energetico e produce risultati annualizzati.

    Il flusso di calcolo è:

    1. Validazione e normalizzazione delle tabelle di input (policy, tariffe,
       asset, fiscalità, parametri DCF).
    2. Costruzione delle serie orarie per membro (prelievo/immissione).
    3. Costruzione dei profili orari di prezzo (acquisto/vendita) e incentivi
       (TIP/TIAD), con eventuale utilizzo di profili caricati da CSV.
    4. Allocazione oraria di TIP/TIAD tra consumatori e produttori tramite pesi
       proporzionali ai volumi orari (robusta a ore con denominatore nullo).
    5. Aggregazione su "anno 1" e (opzionale) normalizzazione ad anno intero
       quando il periodo simulato non copre 8760/8784 ore.
    6. Costruzione dei prospetti annuali (PnL e Cash Flow), ammortamenti,
       fiscalità per classe utente e override per membro.
    7. Calcolo KPI (NPV/IRR/Payback) per membro e totale comunità.

    Parameters
    ----------
    energy_run:
        Dati run energetico caricati da ``load_energy_run``.
    assumptions:
        DataFrame delle tabelle UI (policy, tariffe, asset, fiscalità, DCF).
        componenti di ricavo/costo.

    Returns
    -------
    EconomicsResult
        Risultati economici per membro e totali, con eventuale breakdown orario.
    """
    # ---- 0) validate inputs & normalize schemas
    policy = _validate_policy_df(assumptions.policy_cer)
    dcf = _validate_dcf_df(assumptions.dcf_params)
    users = _validate_users_df(assumptions.users)
    buy = _validate_tariffs_buy_df(assumptions.tariffs_buy, users)
    sell = _validate_tariffs_sell_df(assumptions.tariffs_sell, users)
    assets_pv = _validate_assets_pv_df(assumptions.assets_pv, users)
    assets_wind = _validate_assets_wind_df(getattr(assumptions, 'assets_wind', None), users)
    assets_bess = _validate_assets_bess_df(assumptions.assets_bess, users)
    tax_by_class = _validate_tax_by_class_df(assumptions.tax_by_class)
    tax_over = _validate_tax_overrides_df(assumptions.tax_overrides, users)

    # ---- 1) build member hourly energy frames (E_prel, E_imm)
    member_energy = _members_hourly_map(energy_run.members_hourly_long)
    cer = energy_run.cer_hourly
    idx = cer.index

    # ---- 2) load/build common profiles (PZO, TIP, TIAD)
    cfg = get_config()
    local_tz = cfg.timezone

    # PZO: accetta diversi nomi colonna per retro-compatibilita' con CSV utente
    pzo = _maybe_profile_single(
        idx,
        assumptions.pzo_profile,
        value_col_candidates=("pzo_eur_kwh", "price_eur_kwh", "eur_kwh", "value", "PZO", "pzo"),
        name="PZO",
    )
    # Incentivi (TIP) e valorizzazione (TIAD)
    #
    # Oltre alle modalita' "fixed" / "profile_upload", il simulatore supporta una
    # modalita' *RSE-like* che replica la struttura delle formule usate nel simulatore
    # CACER di RSE (Decreto CACER + TIAD ARERA):
    #   - TIP: funzione di potenza incentivata, PZO, fattore zonale e contributi pubblici
    #   - TIAD: termine TRASe (CER) e, per AUC, termini aggiuntivi con dipendenza da PZO
    tip_mode = str(policy.loc[0, "tip_mode"])
    tip_mode_l = (tip_mode or "").strip().lower()
    tip_is_rse = tip_mode_l in ("rse_decree", "rse", "rse_tip")
    # Nota: per le modalita' RSE-like la TIP e' calcolata *per impianto* e quindi
    # verrà costruita più avanti, dopo aver calcolato i pesi orari di immissione.
    tip = None if tip_is_rse else _build_incentive_profile(
        idx,
        mode=tip_mode,
        fixed_value=float(policy.loc[0, "tip_value_eur_kwh"]),
        profile_df=assumptions.tip_profile,
        pzo=pzo,
    )

    tiad_mode = str(policy.loc[0, "tiad_mode"])
    if (tiad_mode or "").strip().lower() in ("rse_arera", "rse", "rse_tiad"):
        tiad = _build_tiad_profile_rse(idx=idx, pzo=pzo, policy=policy, users=users)
    else:
        tiad = _build_incentive_profile(
            idx,
            mode=tiad_mode,
            fixed_value=float(policy.loc[0, "tiad_value_eur_kwh"]),
            profile_df=assumptions.tiad_profile,
            pzo=None,
        )

    # ---- 3) build per-member buy/sell price profiles
    buy_profiles = _maybe_profiles_wide(idx, assumptions.buy_profiles)
    sell_profiles = _maybe_profiles_wide(idx, assumptions.sell_profiles)

    p_buy: Dict[str, pd.Series] = {}
    p_sell: Dict[str, pd.Series] = {}
    for mid in users["member_id"].astype(str).tolist():
        p_buy[mid] = _build_member_buy_price(
            idx,
            member_id=mid,
            tariff_row=buy.loc[mid],
            pzo=pzo,
            wide_profiles=buy_profiles,
            local_tz=local_tz,
        )
        p_sell[mid] = _build_member_sell_price(
            idx,
            member_id=mid,
            tariff_row=sell.loc[mid],
            pzo=pzo,
            wide_profiles=sell_profiles,
        )

    # ---- 4) hourly allocation of TIP/TIAD per member
    alpha_c = float(policy.loc[0, "alpha_consumers"])
    alpha_p = 1.0 - alpha_c

    # weights per hour
    E_prel_tot = pd.Series(0.0, index=idx)
    E_imm_tot = pd.Series(0.0, index=idx)
    for mid, df in member_energy.items():
        if mid not in users.index:
            continue
        if not bool(users.loc[mid, "enabled"]):
            continue
        E_prel_tot = E_prel_tot.add(df["E_prel_kWh"], fill_value=0.0)
        E_imm_tot = E_imm_tot.add(df["E_imm_kWh"], fill_value=0.0)

    # avoid division-by-zero; weights=0 when denominator=0
    w_prel: Dict[str, pd.Series] = {}
    w_imm: Dict[str, pd.Series] = {}
    for mid, df in member_energy.items():
        if mid not in users.index:
            continue
        if not bool(users.loc[mid, "enabled"]):
            continue
        w_prel[mid] = _safe_div(df["E_prel_kWh"], E_prel_tot)
        w_imm[mid] = _safe_div(df["E_imm_kWh"], E_imm_tot)

    E_cond = cer["E_cond_kWh"].astype(float)

    # --- TIP: modalità RSE-like *per impianto* (approssimazione compatibile con output attuali)
    #
    # Non avendo Econd per impianto nel run energetico, la quota di energia condivisa
    # attribuibile all'impianto/membro i è stimata come:
    #   Econd_i(h) = Econd(h) * Eimm_i(h)/Eimm_tot(h)
    # (cioè quota di immissione). Questo è coerente con la logica RSE: la TIP è per
    # impianto e si applica alla sua quota di energia condivisa.
    tip_rates: Dict[str, pd.Series] = {}
    if tip_is_rse:
        # impianti incentivabili: membri abilitati con new_plant=True e capacità > 0
        for mid in users.index.astype(str).tolist():
            if not bool(users.loc[mid, "enabled"]):
                continue
            if not bool(users.loc[mid, "new_plant"]):
                continue
            cap_kw = pd.to_numeric(users.loc[mid, "installed_capacity_kw"], errors="coerce")
            cap_kw = float(cap_kw) if np.isfinite(cap_kw) else 0.0
            num = users.loc[mid, "num"] if "num" in users.columns else 1
            try:
                num = int(num)
            except Exception:
                num = 1
            cap_kw = max(0.0, cap_kw) * max(1, num)
            if cap_kw <= 0:
                continue
            tip_rates[mid] = _build_tip_profile_rse_plant(idx=idx, pzo=pzo, policy=policy, plant_power_kw=cap_kw)

        # quota Econd per impianto/membro (solo per incentivabili)
        econd_by_plant: Dict[str, pd.Series] = {}
        for mid in tip_rates.keys():
            if mid in w_imm:
                econd_by_plant[mid] = E_cond * w_imm[mid]
            else:
                econd_by_plant[mid] = pd.Series(0.0, index=idx)

        # ricavo TIP totale come somma sui plant
        R_tip_tot = pd.Series(0.0, index=idx)
        for mid, r in tip_rates.items():
            R_tip_tot = R_tip_tot.add(econd_by_plant[mid] * r, fill_value=0.0)

        # pesi per distribuire la quota "producer" della TIP: proporzionali a Econd_i
        econd_tip_tot = pd.Series(0.0, index=idx)
        for s in econd_by_plant.values():
            econd_tip_tot = econd_tip_tot.add(s, fill_value=0.0)

        w_tip_prod: Dict[str, pd.Series] = {}
        for mid in users.index.astype(str).tolist():
            if not bool(users.loc[mid, "enabled"]):
                continue
            s = econd_by_plant.get(mid)
            if s is None:
                s = pd.Series(0.0, index=idx)
            w_tip_prod[mid] = _safe_div(s, econd_tip_tot)
    else:
        # modalità legacy: tariffa comunità applicata su Econd comunitaria
        assert tip is not None
        R_tip_tot = E_cond * tip
        w_tip_prod = w_imm

    R_tiad_tot = E_cond * tiad

    R_tip_c = alpha_c * R_tip_tot
    R_tip_p = alpha_p * R_tip_tot
    R_tiad_c = alpha_c * R_tiad_tot
    R_tiad_p = alpha_p * R_tiad_tot

    # ---- 5) hourly operational margin per member (buy/sell + incentives)
    hourly_rows = []
    yearly_base_rows = []

    annual_hours = _annual_hours_for_year(int(policy.loc[0, "year0"]), tz=str(energy_run.period_tz))
    scale_to_annual = bool(dcf.loc[0, "normalize_to_annual"])
    scale_factor = float(annual_hours) / float(len(idx)) if scale_to_annual else 1.0

    for mid in users.index.astype(str).tolist():
        if not bool(users.loc[mid, "enabled"]):
            continue
        if mid not in member_energy:
            # utente senza profili energetici (es. disabilitato nel bilanciamento)
            continue

        df_e = member_energy[mid]
        E_prel = df_e["E_prel_kWh"].astype(float)
        E_imm = df_e["E_imm_kWh"].astype(float)
        # Autoconsumo (kWh) => risparmio bolletta: energia che non viene prelevata dalla rete
        E_aut = df_e["E_aut_kWh"].astype(float) if "E_aut_kWh" in df_e.columns else pd.Series(0.0, index=idx)

        # prezzi
        pb = p_buy[mid]
        ps = p_sell[mid]

        # costi/ricavi energia
        C_buy_h = E_prel * pb
        # Risparmio da autoconsumo valorizzato al prezzo di acquisto (€/kWh)
        S_aut_h = E_aut * pb
        R_sell_h = E_imm * ps if bool(sell.loc[mid, "sell_enabled"]) else pd.Series(0.0, index=idx)

        # TIP/TIAD allocati
        tip_i_h = R_tip_c * w_prel[mid] + R_tip_p * w_tip_prod.get(mid, pd.Series(0.0, index=idx))
        tiad_i_h = R_tiad_c * w_prel[mid] + R_tiad_p * w_imm[mid]

        # risultati orari (per debug)
        if return_hourly_breakdown:
            tmp = pd.DataFrame(
                {
                    "member_id": mid,
                    "C_buy": C_buy_h,
                    "S_aut": S_aut_h,
                    "R_sell": R_sell_h,
                    "R_tip": tip_i_h,
                    "R_tiad": tiad_i_h,
                },
                index=idx,
            )
            hourly_rows.append(tmp)

        # aggregazione anno tipo (e scaling se periodo non è annuale)
        base = {
            "member_id": mid,
            "rev_sell_year1": float(R_sell_h.sum() * scale_factor),
            "rev_tip_year1": float(tip_i_h.sum() * scale_factor),
            "rev_tiad_year1": float(tiad_i_h.sum() * scale_factor),
            "cost_buy_year1": float(C_buy_h.sum() * scale_factor),
            # Risparmio bolletta da energia autoconsumata (valorizzata al prezzo di acquisto)
            "savings_autocons_year1": float(S_aut_h.sum() * scale_factor),
        }
        yearly_base_rows.append(base)

    hourly_breakdown = None
    if return_hourly_breakdown and hourly_rows:
        hourly_breakdown = pd.concat(hourly_rows, axis=0)
        hourly_breakdown.index.name = "time"

    base_year = pd.DataFrame(yearly_base_rows).set_index("member_id")

    # ---- 6) add fixed fees, OPEX, CAPEX schedules, depreciation, taxes, CF and KPIs
    results_member = _build_yearly_statements(
        users=users,
        base_year=base_year,
        buy=buy,
        sell=sell,
        assets_pv=assets_pv,
        assets_wind=assets_wind,
        assets_bess=assets_bess,
        tax_by_class=tax_by_class,
        tax_overrides=tax_over,
        policy=policy,
        dcf=dcf,
    )

    pnl_by_member, cash_by_member, kpi_by_member = results_member

    # ---- 7) totals
    pnl_total = _aggregate_total_statement(pnl_by_member)
    cash_total = _aggregate_total_statement(cash_by_member)
    kpi_total = _compute_kpis_for_statement(cash_total, discount_rate=float(dcf.loc[0, "discount_rate"]))

    return EconomicsResult(
        pnl_by_member=pnl_by_member,
        cashflow_by_member=cash_by_member,
        kpis_by_member=kpi_by_member,
        pnl_total=pnl_total,
        cashflow_total=cash_total,
        kpis_total=kpi_total,
        hourly_breakdown=hourly_breakdown,
    )


# =============================================================================
# Output persistence
# =============================================================================


def save_economic_outputs(
    out_dir: Path,
    assumptions: EconomicsAssumptions,
    result: EconomicsResult,
) -> Dict[str, Path]:
    """Scrive su disco gli output economici (CSV + manifest).

    Questa funzione è pensata per il layer UI, in modo da rendere la valutazione
    riproducibile: salva sia gli output (PnL, Cash Flow, KPI) sia le assunzioni
    effettivamente utilizzate.

    Convenzioni
    -----------
    * I CSV vengono salvati con le stesse colonne attese dalla UI.
    * ``manifest.json`` elenca i file prodotti e la versione del formato.
    * Le scritture sono *best-effort atomiche* tramite file temporaneo + ``Path.replace``.

    Parameters
    ----------
    out_dir:
        Directory di output (creata se non esiste).
    assumptions:
        Tabelle di input consolidate usate nel calcolo.
    result:
        Risultati economici calcolati.

    Returns
    -------
    dict[str, pathlib.Path]
        Mappa ``nome_file`` -> percorso assoluto del file creato.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # assumptions (CSV) + json manifest
    def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        """Scrittura atomica best-effort (file temporaneo + replace)."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding=encoding)
        tmp.replace(path)

    def _atomic_write_csv(df: pd.DataFrame, path: Path, *, index: bool = True) -> None:
        """Scrittura CSV atomica best-effort."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=index)
        tmp.replace(path)

    def _save_df(df: Optional[pd.DataFrame], name: str) -> None:
        if df is None:
            return
        p = out_dir / name
        _atomic_write_csv(df, p, index=True)
        paths[name] = p

    _save_df(assumptions.policy_cer, "policy_cer.csv")
    _save_df(assumptions.users, "users.csv")
    _save_df(assumptions.tariffs_buy, "tariffs_buy.csv")
    _save_df(assumptions.tariffs_sell, "tariffs_sell.csv")
    _save_df(assumptions.assets_pv, "assets_pv.csv")
    _save_df(getattr(assumptions, "assets_wind", None), "assets_wind.csv")
    _save_df(assumptions.assets_bess, "assets_bess.csv")
    _save_df(assumptions.tax_by_class, "tax_by_class.csv")
    _save_df(assumptions.tax_overrides, "tax_overrides.csv")
    _save_df(assumptions.dcf_params, "dcf_params.csv")
    _save_df(assumptions.pzo_profile, "pzo_profile.csv")
    _save_df(assumptions.tip_profile, "tip_profile.csv")
    _save_df(assumptions.tiad_profile, "tiad_profile.csv")
    _save_df(assumptions.buy_profiles, "buy_profiles.csv")
    _save_df(assumptions.sell_profiles, "sell_profiles.csv")

    # results
    _atomic_write_csv(result.pnl_by_member, out_dir / "pnl_by_member.csv", index=True)
    _atomic_write_csv(result.cashflow_by_member, out_dir / "cashflow_by_member.csv", index=True)
    _atomic_write_csv(result.kpis_by_member, out_dir / "kpis_by_member.csv", index=False)
    _atomic_write_csv(result.pnl_total, out_dir / "pnl_total.csv", index=True)
    _atomic_write_csv(result.cashflow_total, out_dir / "cashflow_total.csv", index=True)
    _atomic_write_csv(result.kpis_total, out_dir / "kpis_total.csv", index=False)
    paths.update(
        {
            "pnl_by_member.csv": out_dir / "pnl_by_member.csv",
            "cashflow_by_member.csv": out_dir / "cashflow_by_member.csv",
            "kpis_by_member.csv": out_dir / "kpis_by_member.csv",
            "pnl_total.csv": out_dir / "pnl_total.csv",
            "cashflow_total.csv": out_dir / "cashflow_total.csv",
            "kpis_total.csv": out_dir / "kpis_total.csv",
        }
    )
    if result.hourly_breakdown is not None:
        p = out_dir / "hourly_breakdown.csv"
        _atomic_write_csv(result.hourly_breakdown, p, index=True)
        paths["hourly_breakdown.csv"] = p

    # manifest
    manifest = {
        "version": 1,
        "files": {k: str(v) for k, v in paths.items()},
    }
    p_manifest = out_dir / "manifest.json"
    _atomic_write_text(p_manifest, json.dumps(manifest, indent=2))
    paths["manifest.json"] = p_manifest
    return paths



def load_economic_result(out_dir: Path) -> EconomicsResult:
    """Carica un set di output economici precedentemente salvati.

    La funzione è pensata per la UI Streamlit per il ripristino dei risultati da
    disco (senza ricalcolo). Legge i CSV prodotti da :func:`save_economic_outputs`.

    Parameters
    ----------
    out_dir:
        Directory contenente i CSV e (opzionalmente) ``hourly_breakdown.csv``.

    Returns
    -------
    EconomicsResult
        Oggetto risultati con DataFrame pronti per il rendering.

    Raises
    ------
    FileNotFoundError
        Se la directory non esiste.
    """
    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError(f"Directory output economici non trovata: {out_dir}")

    def _read_multiindex(path: Path) -> pd.DataFrame:
        # prova MultiIndex (member_id, year)
        try:
            df = pd.read_csv(path, index_col=[0, 1])
            if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
                df.index = pd.MultiIndex.from_tuples(
                    [(str(a), int(b)) for a, b in df.index.to_list()],
                    names=["member_id", "year"],
                )
            return df
        except Exception:
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.astype(str)
            return df

    pnl_by_member = _read_multiindex(out_dir / "pnl_by_member.csv")
    cash_by_member = _read_multiindex(out_dir / "cashflow_by_member.csv")

    pnl_total = pd.read_csv(out_dir / "pnl_total.csv", index_col=0)
    pnl_total.index = pnl_total.index.astype(int)

    cash_total = pd.read_csv(out_dir / "cashflow_total.csv", index_col=0)
    cash_total.index = cash_total.index.astype(int)

    kpis_by_member = pd.read_csv(out_dir / "kpis_by_member.csv")
    kpis_total = pd.read_csv(out_dir / "kpis_total.csv")

    hourly_breakdown = None
    p_h = out_dir / "hourly_breakdown.csv"
    if p_h.exists():
        hourly_breakdown = pd.read_csv(p_h, index_col=0)
        hourly_breakdown.index = pd.DatetimeIndex(pd.to_datetime(hourly_breakdown.index, utc=True))

    return EconomicsResult(
        pnl_by_member=pnl_by_member,
        cashflow_by_member=cash_by_member,
        kpis_by_member=kpis_by_member,
        pnl_total=pnl_total,
        cashflow_total=cash_total,
        kpis_total=kpis_total,
        hourly_breakdown=hourly_breakdown,
    )


# =============================================================================
# Validation & helpers
# =============================================================================


def _read_time_indexed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=True, errors="raise")
    df.index = pd.DatetimeIndex(idx)
    df.index.name = "time"
    return df


def _validate_energy_run_frames(cer: pd.DataFrame, members_long: pd.DataFrame) -> None:
    for c in ["E_cond_kWh", "E_export_kWh", "E_import_kWh"]:
        if c not in cer.columns:
            raise ValueError(f"cer_hourly: colonna mancante: {c}")
    if "member_id" not in members_long.columns:
        raise ValueError("members_hourly_long: colonna 'member_id' mancante")
    for c in ["E_prel_kWh", "E_imm_kWh"]:
        if c not in members_long.columns:
            raise ValueError(f"members_hourly_long: colonna mancante: {c}")


def _members_hourly_map(members_long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for mid, df in members_long.groupby("member_id"):
        tmp = df.drop(columns=["member_id"]).copy()
        tmp.index = pd.DatetimeIndex(pd.to_datetime(tmp.index, utc=True))
        out[str(mid)] = tmp.sort_index()
    return out


def _validate_policy_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("policy_cer: DataFrame vuoto")
    x = df.copy().reset_index(drop=True)
    required = [
        "alpha_consumers",
        "tip_mode",
        "tiad_mode",
        "tip_value_eur_kwh",
        "tiad_value_eur_kwh",
        "incentive_years",
        "year0",
    ]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"policy_cer: colonna mancante: {c}")
    # Colonne opzionali (modalità RSE-like e controlli di escalation)
    if "cacer_type" not in x.columns:
        x["cacer_type"] = "CER"  # CER / AUC / NO_CACER
    if "tip_rse_power_kw" not in x.columns:
        x["tip_rse_power_kw"] = np.nan
    if "tip_rse_macro_area" not in x.columns:
        x["tip_rse_macro_area"] = "NORD"  # NORD / CENTRO / SUD
    if "tip_rse_grant_intensity" not in x.columns:
        x["tip_rse_grant_intensity"] = 0.0  # 0..1
    if "tiad_rse_TRASe_eur_mwh" not in x.columns:
        x["tiad_rse_TRASe_eur_mwh"] = 0.0
    if "tiad_rse_BTAU_eur_mwh" not in x.columns:
        x["tiad_rse_BTAU_eur_mwh"] = 0.0
    if "tiad_rse_Cpr_bt" not in x.columns:
        x["tiad_rse_Cpr_bt"] = 0.0
    if "tiad_rse_Cpr_mt" not in x.columns:
        x["tiad_rse_Cpr_mt"] = 0.0
    if "tiad_rse_share_bt" not in x.columns:
        x["tiad_rse_share_bt"] = np.nan
    if "tip_escalation_rate" not in x.columns:
        x["tip_escalation_rate"] = 0.0
    if "tiad_escalation_rate" not in x.columns:
        x["tiad_escalation_rate"] = 0.0
    alpha = float(x.loc[0, "alpha_consumers"])
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("policy_cer.alpha_consumers deve essere in [0,1]")
    return x


def _validate_dcf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("dcf_params: DataFrame vuoto")
    x = df.copy().reset_index(drop=True)
    required = [
        "horizon_years",
        "discount_rate",
        "inflation_rate",
        "escalation_buy",
        "escalation_sell",
        "escalation_opex",
    ]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"dcf_params: colonna mancante: {c}")

    # Colonne opzionali (UI "advanced"). Se assenti, imposta default robusti.
    if "normalize_to_annual" not in x.columns:
        x["normalize_to_annual"] = True
    if "apply_incentives_beyond_run_year" not in x.columns:
        x["apply_incentives_beyond_run_year"] = True
    if "working_capital_enabled" not in x.columns:
        x["working_capital_enabled"] = False
    if "dso_days" not in x.columns:
        x["dso_days"] = 0
    if "dpo_days" not in x.columns:
        x["dpo_days"] = 0

    # Detrazione fiscale (Bonus ristrutturazione) - opzionale
    if "detraction_enabled" not in x.columns:
        x["detraction_enabled"] = False
    if "detraction_rate" not in x.columns:
        x["detraction_rate"] = 0.0
    if "detraction_cap_eur" not in x.columns:
        x["detraction_cap_eur"] = 96000.0
    if "detraction_years" not in x.columns:
        x["detraction_years"] = 10

    if int(x.loc[0, "horizon_years"]) <= 0:
        raise ValueError("dcf_params.horizon_years deve essere > 0")
    if float(x.loc[0, "discount_rate"]) <= -0.99:
        raise ValueError("dcf_params.discount_rate non valido")
    return x


def _validate_users_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("users: DataFrame vuoto")
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("users: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    # required columns
    for c in ["role", "user_class", "enabled"]:
        if c not in x.columns:
            raise ValueError(f"users: colonna mancante: {c}")
    x["enabled"] = x["enabled"].astype(bool)
    x["user_class"] = x["user_class"].astype(str).str.lower()
    return x


def _validate_tariffs_buy_df(df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("tariffs_buy: DataFrame vuoto")
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("tariffs_buy: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    # required
    for c in ["buy_price_mode", "buy_fixed_eur_kwh", "f1_eur_kwh", "f2_eur_kwh", "f3_eur_kwh", "buy_spread_eur_kwh", "buy_multiplier", "annual_fixed_fee_eur", "power_fee_eur_per_kw_year", "contract_power_kw"]:
        if c not in x.columns:
            raise ValueError(f"tariffs_buy: colonna mancante: {c}")
    # ensure all members present
    missing = [m for m in users.index if m not in x.index]
    if missing:
        raise ValueError(f"tariffs_buy: manca la riga per member_id: {missing}")
    return x


def _validate_tariffs_sell_df(df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("tariffs_sell: DataFrame vuoto")
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("tariffs_sell: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    for c in ["sell_enabled", "sell_price_mode", "sell_fixed_eur_kwh", "sell_fee_eur_kwh", "sell_multiplier", "annual_rid_fee_eur"]:
        if c not in x.columns:
            raise ValueError(f"tariffs_sell: colonna mancante: {c}")
    missing = [m for m in users.index if m not in x.index]
    if missing:
        raise ValueError(f"tariffs_sell: manca la riga per member_id: {missing}")
    x["sell_enabled"] = x["sell_enabled"].astype(bool)
    return x


def _validate_assets_pv_df(df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("assets_pv: DataFrame vuoto")
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("assets_pv: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    required = [
        "pv_exists",
        "pv_is_sunk",
        "pv_capex_eur_per_kw",
        "pv_installed_kw",
        "pv_capex_override_eur",
        "pv_opex_eur_per_kw_year",
        "pv_life_years",
        "pv_inverter_repl_year",
        "pv_inverter_repl_eur_per_kw",
    ]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"assets_pv: colonna mancante: {c}")
    missing = [m for m in users.index if m not in x.index]
    if missing:
        raise ValueError(f"assets_pv: manca la riga per member_id: {missing}")
    x["pv_exists"] = x["pv_exists"].astype(bool)
    x["pv_is_sunk"] = x["pv_is_sunk"].astype(bool)
    return x


def _default_assets_wind_df(users: pd.DataFrame) -> pd.DataFrame:
    """Default wind assets table (all zeros) for backward compatibility."""
    rows = []
    for mid in users.index.astype(str).tolist():
        rows.append(
            {
                "member_id": str(mid),
                "wind_exists": False,
                "wind_is_sunk": True,
                "wind_capex_eur_per_kw": 0.0,
                "wind_installed_kw": 0.0,
                "wind_capex_override_eur": np.nan,
                "wind_opex_eur_per_kw_year": 0.0,
                "wind_life_years": 20,
                "wind_major_repl_year": np.nan,
                "wind_major_repl_eur_per_kw": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _validate_assets_wind_df(df: Optional[pd.DataFrame], users: pd.DataFrame) -> pd.DataFrame:
    """Validate wind assets.

    If df is None/empty, returns a default (all zeros) table to preserve
    compatibility with scenarios created before wind support.
    """
    if df is None or df.empty:
        x = _default_assets_wind_df(users)
    else:
        x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("assets_wind: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    required = [
        "wind_exists",
        "wind_is_sunk",
        "wind_capex_eur_per_kw",
        "wind_installed_kw",
        "wind_capex_override_eur",
        "wind_opex_eur_per_kw_year",
        "wind_life_years",
        "wind_major_repl_year",
        "wind_major_repl_eur_per_kw",
    ]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"assets_wind: colonna mancante: {c}")
    missing = [m for m in users.index if m not in x.index]
    if missing:
        raise ValueError(f"assets_wind: manca la riga per member_id: {missing}")
    x["wind_exists"] = x["wind_exists"].fillna(False).astype(bool)
    x["wind_is_sunk"] = x["wind_is_sunk"].fillna(False).astype(bool)
    return x



def _validate_assets_bess_df(df: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("assets_bess: DataFrame vuoto")
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("assets_bess: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    required = [
        "bess_initial_kwh",
        "bess_capex_eur_per_kwh",
        "bess_opex_pct_capex",
        "bess_opex_eur_per_kwh_year",
        "bess_life_years",
        "bess_replacement",
    ]
    for c in required:
        if c not in x.columns:
            raise ValueError(f"assets_bess: colonna mancante: {c}")
    missing = [m for m in users.index if m not in x.index]
    if missing:
        raise ValueError(f"assets_bess: manca la riga per member_id: {missing}")
    x["bess_replacement"] = x["bess_replacement"].astype(bool)
    return x


def _validate_tax_by_class_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("tax_by_class: DataFrame vuoto")
    x = df.copy()
    if "user_class" not in x.columns:
        raise ValueError("tax_by_class: colonna 'user_class' mancante")
    x["user_class"] = x["user_class"].astype(str).str.lower()
    x = x.set_index("user_class", drop=False)
    for c in ["tax_enabled", "tax_rate_effective", "allow_tax_loss_carryforward", "loss_carry_years", "depreciation_enabled"]:
        if c not in x.columns:
            raise ValueError(f"tax_by_class: colonna mancante: {c}")
    x["tax_enabled"] = x["tax_enabled"].astype(bool)
    x["allow_tax_loss_carryforward"] = x["allow_tax_loss_carryforward"].astype(bool)
    x["depreciation_enabled"] = x["depreciation_enabled"].astype(bool)
    return x


def _validate_tax_overrides_df(df: Optional[pd.DataFrame], users: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    x = df.copy()
    if "member_id" not in x.columns:
        raise ValueError("tax_overrides: colonna 'member_id' mancante")
    x["member_id"] = x["member_id"].astype(str)
    x = x.set_index("member_id", drop=False)
    # allowed columns
    for c in ["tax_enabled_override", "tax_rate_override"]:
        if c not in x.columns:
            raise ValueError(f"tax_overrides: colonna mancante: {c}")
    # only members in users
    x = x.loc[x.index.intersection(users.index)].copy()
    return x



def _maybe_profile_single(
    idx: pd.DatetimeIndex,
    profile_df: Optional[pd.DataFrame],
    *,
    value_col_candidates: Tuple[str, ...],
    name: str,
) -> Optional[pd.Series]:
    if profile_df is None or profile_df.empty:
        return None
    df = profile_df.copy()
    if df.index.name != "time":
        # tenta: prima colonna come time
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time")
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
            df.index.name = "time"
        else:
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "time"
    col = None
    for c in value_col_candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"Profilo {name}: nessuna colonna valida trovata tra {value_col_candidates}")
    s = df[col].astype(float)
    s = s.reindex(idx)
    if s.isna().any():
        raise ValueError(f"Profilo {name}: non copre interamente il periodo del run")
    s.name = name
    return s


def _maybe_profiles_wide(idx: pd.DatetimeIndex, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    x = df.copy()
    # index handling
    if x.index.name != "time":
        if "time" in x.columns:
            x["time"] = pd.to_datetime(x["time"], utc=True)
            x = x.set_index("time")
        elif "timestamp" in x.columns:
            x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
            x = x.set_index("timestamp")
            x.index.name = "time"
        else:
            x.index = pd.to_datetime(x.index, utc=True)
            x.index.name = "time"
    x = x.reindex(idx)
    if x.isna().any().any():
        raise ValueError("Profilo prezzi (wide): non copre interamente il periodo del run")
    # normalize column names to string member_id
    x.columns = [str(c) for c in x.columns]
    return x


def _build_incentive_profile(
    idx: pd.DatetimeIndex,
    *,
    mode: str,
    fixed_value: float,
    profile_df: Optional[pd.DataFrame],
    pzo: Optional[pd.Series],
) -> pd.Series:
    mode = (mode or "fixed").strip().lower()
    if mode == "fixed":
        return pd.Series(float(fixed_value), index=idx)
    if mode == "profile_upload":
        s = _maybe_profile_single(idx, profile_df, value_col_candidates=("eur_kwh", "value", "tip_eur_kwh", "tiad_eur_kwh"), name="INC")
        assert s is not None
        return s
    if mode == "pzo_function":
        if pzo is None:
            raise ValueError("tip_mode=pzo_function richiede pzo_profile")
        # Funzione volutamente semplice (placeholder):
        # TIP = max(0, base + max(0, cap_ref - PZO)) con cap superiore.
        base = float(fixed_value)
        cap_ref = 0.18  # 180 €/MWh -> 0.18 €/kWh
        cap_max = 0.12  # 120 €/MWh -> 0.12 €/kWh extra massimo
        extra = (cap_ref - pzo).clip(lower=0.0)
        extra = extra.clip(upper=cap_max)
        return (base + extra).reindex(idx)
    raise ValueError(f"Modalità incentivo non supportata: {mode}")


def _build_tip_profile_rse(
    *,
    idx: pd.DatetimeIndex,
    pzo: Optional[pd.Series],
    policy: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.Series:
    """Costruisce il profilo TIP (€/kWh) con struttura *RSE-like*.

    Replica (in forma parametrica) l'impostazione presente nel simulatore CACER
    di RSE (Decreto 7/12/2023 n. 414 + Regole Operative GSE), limitatamente al
    calcolo della *tariffa premio* oraria:

    - scaglioni di potenza incentivata: <200 kW, 200-600 kW, >=600 kW
    - parte variabile: max(0, 180 - PZO) con cap dipendente dallo scaglione
    - fattore zonale (€/MWh): Nord=10, Centro=4, Sud=0
    - riduzione per contributi pubblici (es. PNRR): funzione della grant_intensity

    Note
    ----
    * Nel simulatore RSE la TIP viene calcolata per impianto e poi applicata alla
      quota di energia condivisa attribuita a quell'impianto. Nel presente
      simulatore (versione attuale) la serie ``E_cond_kWh`` e' aggregata a livello
      di comunità; di conseguenza la TIP è costruita come tariffa unica di comunità
      e sara' applicata a ``E_cond_kWh``.
    """
    if pzo is None:
        raise ValueError("tip_mode=RSE richiede pzo_profile")

    # --- Parametri (con fallback coerenti) ---
    power_kw = float(policy.loc[0, "tip_rse_power_kw"]) if "tip_rse_power_kw" in policy.columns else np.nan
    if np.isnan(power_kw) or power_kw <= 0:
        # fallback: usa la massima capacità installata tra membri abilitati con impianto 'new_plant'
        u = users.copy()
        u["_installed"] = pd.to_numeric(u.get("installed_capacity_kw", 0.0), errors="coerce").fillna(0.0)
        u["_new"] = u.get("new_plant", True).astype(bool)
        u["_enabled"] = u.get("enabled", True).astype(bool)
        power_kw = float(u.loc[u["_enabled"] & u["_new"], "_installed"].max())
        if not np.isfinite(power_kw) or power_kw <= 0:
            power_kw = 0.0

    macro = str(policy.loc[0, "tip_rse_macro_area"]) if "tip_rse_macro_area" in policy.columns else "NORD"
    macro = macro.strip().upper()
    if macro in ("NORD", "NORTH"):
        fc_zonale_eur_mwh = 10.0
    elif macro in ("CENTRO", "CENTER", "CENTRAL"):
        fc_zonale_eur_mwh = 4.0
    elif macro in ("SUD", "SOUTH"):
        fc_zonale_eur_mwh = 0.0
    else:
        # fallback: nessun fattore zonale
        fc_zonale_eur_mwh = 0.0

    grant = float(policy.loc[0, "tip_rse_grant_intensity"]) if "tip_rse_grant_intensity" in policy.columns else 0.0
    grant = 0.0 if (not np.isfinite(grant)) else float(grant)
    if grant > 0.4:
        f_pnrr = 1.0
    else:
        f_pnrr = (grant / 0.4) * 0.5
    f_pnrr = 0.0 if (not np.isfinite(f_pnrr)) else float(f_pnrr)

    # --- Scaglioni potenza (€/MWh) ---
    if power_kw >= 600.0:
        tp_base = 60.0
        cap_var = 100.0
    elif power_kw >= 200.0:
        tp_base = 70.0
        cap_var = 110.0
    else:
        tp_base = 80.0
        cap_var = 120.0

    # PZO e' gestito internamente in €/kWh nel simulatore; la formula RSE usa €/MWh.
    pzo_eur_mwh = (pzo.astype(float) * 1000.0).reindex(idx)
    # parte variabile max(0, 180 - PZO)
    var_part = (180.0 - pzo_eur_mwh).clip(lower=0.0)
    tariff_eur_mwh = (np.minimum(cap_var, tp_base + var_part) + fc_zonale_eur_mwh) * (1.0 - f_pnrr)

    # No-CACER -> nessun incentivo (compatibile con RSE)
    if "cacer_type" in policy.columns and str(policy.loc[0, "cacer_type"]).strip().upper() in ("NO_CACER", "NONE"):
        tariff_eur_mwh = tariff_eur_mwh * 0.0

    # €/kWh
    out = (tariff_eur_mwh / 1000.0).astype(float)
    out.name = "TIP"
    return out


def _build_tip_profile_rse_plant(
    *,
    idx: pd.DatetimeIndex,
    pzo: Optional[pd.Series],
    policy: pd.DataFrame,
    plant_power_kw: float,
) -> pd.Series:
    """Costruisce la TIP (€/kWh) *per impianto* con struttura RSE-like.

    Questa funzione è pensata per allineare la monetizzazione TIP all'approccio
    del simulatore CACER di RSE: tariffa premio calcolata in funzione della
    potenza dell'impianto, PZO, fattore zonale e contributi pubblici.

    Nota: la durata incentivo (20 anni dal commissioning) è gestita nel DCF; la
    tariffa qui è calcolata sul periodo simulato.
    """
    if pzo is None:
        raise ValueError("tip_mode=RSE richiede pzo_profile")

    power_kw = float(plant_power_kw) if np.isfinite(plant_power_kw) else 0.0
    power_kw = 0.0 if power_kw < 0 else power_kw

    macro = str(policy.loc[0, "tip_rse_macro_area"]) if "tip_rse_macro_area" in policy.columns else "NORD"
    macro = macro.strip().upper()
    if macro in ("NORD", "NORTH"):
        fc_zonale_eur_mwh = 10.0
    elif macro in ("CENTRO", "CENTER", "CENTRAL"):
        fc_zonale_eur_mwh = 4.0
    elif macro in ("SUD", "SOUTH"):
        fc_zonale_eur_mwh = 0.0
    else:
        fc_zonale_eur_mwh = 0.0

    grant = float(policy.loc[0, "tip_rse_grant_intensity"]) if "tip_rse_grant_intensity" in policy.columns else 0.0
    grant = 0.0 if (not np.isfinite(grant)) else float(grant)
    if grant > 0.4:
        f_pnrr = 1.0
    else:
        f_pnrr = (grant / 0.4) * 0.5
    f_pnrr = 0.0 if (not np.isfinite(f_pnrr)) else float(f_pnrr)

    if power_kw >= 600.0:
        tp_base = 60.0
        cap_var = 100.0
    elif power_kw >= 200.0:
        tp_base = 70.0
        cap_var = 110.0
    else:
        tp_base = 80.0
        cap_var = 120.0

    pzo_eur_mwh = (pzo.astype(float) * 1000.0).reindex(idx)
    var_part = (180.0 - pzo_eur_mwh).clip(lower=0.0)
    tariff_eur_mwh = (np.minimum(cap_var, tp_base + var_part) + fc_zonale_eur_mwh) * (1.0 - f_pnrr)

    if "cacer_type" in policy.columns and str(policy.loc[0, "cacer_type"]).strip().upper() in ("NO_CACER", "NONE"):
        tariff_eur_mwh = tariff_eur_mwh * 0.0

    out = (tariff_eur_mwh / 1000.0).astype(float)
    out.name = "TIP"
    return out


def _build_tiad_profile_rse(
    *,
    idx: pd.DatetimeIndex,
    pzo: Optional[pd.Series],
    policy: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.Series:
    """Costruisce il profilo TIAD/valorizzazione (€/kWh) in stile RSE.

    Implementa la struttura usata da RSE per la valorizzazione dell'energia
    condivisa (delibera ARERA 727/2022/R/eel), senza componenti surplus/social fund.

    - CER: solo termine TRASe
    - AUC: TRASe + BTAU + (Cpr_bt * share_bt + Cpr_mt * share_mt) * PZO

    I parametri sono letti dalla tabella policy (scenario economico).
    """
    cacer_type = str(policy.loc[0, "cacer_type"]).strip().upper() if "cacer_type" in policy.columns else "CER"

    trase_eur_mwh = float(policy.loc[0, "tiad_rse_TRASe_eur_mwh"]) if "tiad_rse_TRASe_eur_mwh" in policy.columns else 0.0
    btau_eur_mwh = float(policy.loc[0, "tiad_rse_BTAU_eur_mwh"]) if "tiad_rse_BTAU_eur_mwh" in policy.columns else 0.0
    cpr_bt = float(policy.loc[0, "tiad_rse_Cpr_bt"]) if "tiad_rse_Cpr_bt" in policy.columns else 0.0
    cpr_mt = float(policy.loc[0, "tiad_rse_Cpr_mt"]) if "tiad_rse_Cpr_mt" in policy.columns else 0.0

    # conversion €/MWh -> €/kWh
    trase = trase_eur_mwh / 1000.0
    btau = btau_eur_mwh / 1000.0

    if cacer_type in ("NO_CACER", "NONE"):
        return pd.Series(0.0, index=idx, name="TIAD")

    if cacer_type == "CER":
        return pd.Series(float(trase), index=idx, name="TIAD")

    # AUC (o altri tipi compatibili): serve PZO
    if pzo is None:
        raise ValueError("tiad_mode=RSE (AUC) richiede pzo_profile")

    share_bt = float(policy.loc[0, "tiad_rse_share_bt"]) if "tiad_rse_share_bt" in policy.columns else np.nan
    if not np.isfinite(share_bt):
        # fallback: stima share BT in base al conteggio utenti, se presente voltage_level
        if "voltage_level" in users.columns:
            u = users.copy()
            u["_enabled"] = u.get("enabled", True).astype(bool)
            bt = (u.loc[u["_enabled"], "voltage_level"].astype(str).str.upper() == "BT").mean()
            share_bt = float(bt) if np.isfinite(bt) else 1.0
        else:
            share_bt = 1.0
    share_bt = float(np.clip(share_bt, 0.0, 1.0))
    share_mt = 1.0 - share_bt

    # pzo in €/kWh
    pzo_kwh = pzo.astype(float).reindex(idx)
    out = trase + btau + (share_bt * cpr_bt + share_mt * cpr_mt) * pzo_kwh
    out = out.astype(float)
    out.name = "TIAD"
    return out


def _build_member_buy_price(
    idx: pd.DatetimeIndex,
    *,
    member_id: str,
    tariff_row: pd.Series,
    pzo: Optional[pd.Series],
    wide_profiles: Optional[pd.DataFrame],
    local_tz: str,
) -> pd.Series:
    mode = str(tariff_row["buy_price_mode"]).strip().lower()
    if mode == "fixed":
        return pd.Series(float(tariff_row["buy_fixed_eur_kwh"]), index=idx, name=f"p_buy_{member_id}")
    if mode == "f1f2f3":
        f1 = float(tariff_row["f1_eur_kwh"])
        f2 = float(tariff_row["f2_eur_kwh"])
        f3 = float(tariff_row["f3_eur_kwh"])
        bands = italian_fasce_band(idx, local_tz=local_tz)
        arr = np.where(bands == "F1", f1, np.where(bands == "F2", f2, f3)).astype(float)
        return pd.Series(arr, index=idx, name=f"p_buy_{member_id}")
    if mode == "profile_upload":
        if wide_profiles is None or member_id not in wide_profiles.columns:
            raise ValueError(f"buy_price_mode=profile_upload: manca profilo per member_id={member_id}")
        return wide_profiles[member_id].astype(float).rename(f"p_buy_{member_id}")
    if mode == "pzo_plus_spread":
        if pzo is None:
            raise ValueError("buy_price_mode=pzo_plus_spread richiede pzo_profile")
        spread = float(tariff_row["buy_spread_eur_kwh"])
        mult = float(tariff_row["buy_multiplier"]) if not pd.isna(tariff_row["buy_multiplier"]) else 1.0
        return (pzo * mult + spread).rename(f"p_buy_{member_id}")
    raise ValueError(f"buy_price_mode non supportata: {mode}")


def _build_member_sell_price(
    idx: pd.DatetimeIndex,
    *,
    member_id: str,
    tariff_row: pd.Series,
    pzo: Optional[pd.Series],
    wide_profiles: Optional[pd.DataFrame],
) -> pd.Series:
    mode = str(tariff_row["sell_price_mode"]).strip().lower()
    if mode == "fixed":
        return pd.Series(float(tariff_row["sell_fixed_eur_kwh"]), index=idx, name=f"p_sell_{member_id}")
    if mode == "profile_upload":
        if wide_profiles is None or member_id not in wide_profiles.columns:
            raise ValueError(f"sell_price_mode=profile_upload: manca profilo per member_id={member_id}")
        return wide_profiles[member_id].astype(float).rename(f"p_sell_{member_id}")
    if mode in ("pzo", "pzo_minus_fee"):
        if pzo is None:
            raise ValueError("sell_price_mode basata su PZO richiede pzo_profile")
        mult = float(tariff_row["sell_multiplier"]) if not pd.isna(tariff_row["sell_multiplier"]) else 1.0
        s = pzo * mult
        if mode == "pzo_minus_fee":
            fee = float(tariff_row["sell_fee_eur_kwh"])
            s = s - fee
        return s.rename(f"p_sell_{member_id}")
    raise ValueError(f"sell_price_mode non supportata: {mode}")


def italian_fasce_band(idx_utc: pd.DatetimeIndex, *, local_tz: str = "Europe/Rome") -> np.ndarray:
    """Assegna fascia F1/F2/F3 su calendario standard (semplificato).

    Regole (approssimazione pratica):
      - Lun-Ven: F1 08-19; F2 07-08 e 19-23; F3 23-07
      - Sabato: F2 07-23; F3 23-07
      - Domenica: F3

    Nota: non gestisce festività nazionali (può essere aggiunto come estensione).
    """
    local = pd.DatetimeIndex(idx_utc).tz_convert(local_tz)
    dow = local.dayofweek.values  # 0=Mon
    hour = local.hour.values

    band = np.full(len(local), "F3", dtype=object)

    # Mon-Fri
    mf = dow <= 4
    band[mf & (hour >= 8) & (hour < 19)] = "F1"
    band[mf & (((hour >= 7) & (hour < 8)) | ((hour >= 19) & (hour < 23)))] = "F2"

    # Saturday
    sat = dow == 5
    band[sat & (hour >= 7) & (hour < 23)] = "F2"

    # Sunday remains F3
    return band


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=num.index)
    mask = den.values != 0
    out.iloc[mask] = (num.iloc[mask].values / den.iloc[mask].values)
    return out


def _annual_hours_for_year(year: int, *, tz: str = "UTC") -> int:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{year+1}-01-01 00:00:00", tz=tz)
    return len(pd.date_range(start=start, end=end, freq="h", inclusive="left"))


# =============================================================================
# Statements, depreciation, taxes, KPIs
# =============================================================================


def _build_yearly_statements(
    *,
    users: pd.DataFrame,
    base_year: pd.DataFrame,
    buy: pd.DataFrame,
    sell: pd.DataFrame,
    assets_pv: pd.DataFrame,
    assets_wind: pd.DataFrame,
    assets_bess: pd.DataFrame,
    tax_by_class: pd.DataFrame,
    tax_overrides: Optional[pd.DataFrame],
    policy: pd.DataFrame,
    dcf: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizon = int(dcf.loc[0, "horizon_years"])
    disc = float(dcf.loc[0, "discount_rate"])
    esc_buy = float(dcf.loc[0, "escalation_buy"])
    esc_sell = float(dcf.loc[0, "escalation_sell"])
    esc_opex = float(dcf.loc[0, "escalation_opex"])
    infl = float(dcf.loc[0, "inflation_rate"])
    incentive_years = int(policy.loc[0, "incentive_years"])
    # Detrazione fiscale (Bonus ristrutturazione) - parametri globali DCF
    det_enabled = bool(dcf.loc[0, "detraction_enabled"]) if "detraction_enabled" in dcf.columns else False
    det_rate = float(dcf.loc[0, "detraction_rate"]) if "detraction_rate" in dcf.columns else 0.0
    det_cap_eur = float(dcf.loc[0, "detraction_cap_eur"]) if "detraction_cap_eur" in dcf.columns else 0.0
    det_years = int(dcf.loc[0, "detraction_years"]) if "detraction_years" in dcf.columns else 0


    # Se inflazione > 0, e escalation sono 0, puoi decidere di applicare inflazione.
    # In questa prima versione: escalation già include l'eventuale inflazione.
    _ = infl

    years = list(range(0, horizon + 1))  # include year 0

    pnl_rows = []
    cash_rows = []
    kpi_rows = []

    for mid in users.index.astype(str).tolist():
        if not bool(users.loc[mid, "enabled"]):
            continue
        if mid not in base_year.index:
            continue

        # --- base values (year1) ---
        rev_sell_y1 = float(base_year.loc[mid, "rev_sell_year1"])
        rev_tip_y1 = float(base_year.loc[mid, "rev_tip_year1"])
        rev_tiad_y1 = float(base_year.loc[mid, "rev_tiad_year1"])
        cost_buy_y1 = float(base_year.loc[mid, "cost_buy_year1"])
        # Risparmio da autoconsumo (se presente; default 0 per retrocompatibilità)
        savings_aut_y1 = float(base_year.loc[mid, "savings_autocons_year1"]) if "savings_autocons_year1" in base_year.columns else 0.0

        # add fixed + power fees (treated as cost) - scaled by numerosity (RSE-like)
        fixed_fee = float(buy.loc[mid, "annual_fixed_fee_eur"])
        power_fee = float(buy.loc[mid, "power_fee_eur_per_kw_year"]) * float(buy.loc[mid, "contract_power_kw"])
        n = int(users.loc[mid, "num"]) if "num" in users.columns else 1
        n = max(n, 1)
        annual_buy_fixed = (fixed_fee + power_fee) * n

        # sell fee (annual) treated as opex - scaled by numerosity (RSE-like)
        annual_rid_fee = float(sell.loc[mid, "annual_rid_fee_eur"]) * n

        # OPEX PV
        opex_pv_y1 = 0.0
        if bool(assets_pv.loc[mid, "pv_exists"]):
            opex_pv_y1 = float(assets_pv.loc[mid, "pv_opex_eur_per_kw_year"]) * float(assets_pv.loc[mid, "pv_installed_kw"])

        # OPEX WIND
        opex_wind_y1 = 0.0
        if bool(assets_wind.loc[mid, "wind_exists"]):
            opex_wind_y1 = float(assets_wind.loc[mid, "wind_opex_eur_per_kw_year"]) * float(assets_wind.loc[mid, "wind_installed_kw"])


        # OPEX BESS
        bess_kwh = float(assets_bess.loc[mid, "bess_initial_kwh"])
        capex_bess = bess_kwh * float(assets_bess.loc[mid, "bess_capex_eur_per_kwh"])
        opex_bess_y1 = 0.0
        if float(assets_bess.loc[mid, "bess_opex_pct_capex"]) not in (0.0, np.nan) and not pd.isna(assets_bess.loc[mid, "bess_opex_pct_capex"]):
            # Robustezza input: accetta sia "2" (2%) sia "0.02" (2%).
            pct = float(assets_bess.loc[mid, "bess_opex_pct_capex"])
            rate = (pct / 100.0) if pct > 1.0 else pct
            opex_bess_y1 = capex_bess * rate
        else:
            opex_bess_y1 = bess_kwh * float(assets_bess.loc[mid, "bess_opex_eur_per_kwh_year"])

        # Total OPEX year1 (include annual RID fee)
        opex_y1 = opex_pv_y1 + opex_wind_y1 + opex_bess_y1 + annual_rid_fee

        # --- CAPEX year 0 ---
        capex0 = 0.0
        capex_pv0 = 0.0
        capex_wind0 = 0.0
        # PV capex only if not sunk
        if bool(assets_pv.loc[mid, "pv_exists"]) and (not bool(assets_pv.loc[mid, "pv_is_sunk"])):
            capex_override = assets_pv.loc[mid, "pv_capex_override_eur"]
            if not pd.isna(capex_override) and float(capex_override) > 0:
                capex_pv0 += float(capex_override)
            else:
                capex_pv0 += float(assets_pv.loc[mid, "pv_capex_eur_per_kw"]) * float(assets_pv.loc[mid, "pv_installed_kw"])
        # WIND capex only if not sunk
        if bool(assets_wind.loc[mid, "wind_exists"]) and (not bool(assets_wind.loc[mid, "wind_is_sunk"])):
            capex_override = assets_wind.loc[mid, "wind_capex_override_eur"]
            if not pd.isna(capex_override) and float(capex_override) > 0:
                capex_wind0 += float(capex_override)
            else:
                capex_wind0 += float(assets_wind.loc[mid, "wind_capex_eur_per_kw"]) * float(assets_wind.loc[mid, "wind_installed_kw"])
        # BESS capex always included (taglia fissata)
        capex0 = capex_pv0 + capex_wind0
        capex0 += capex_bess
        # --- Detrazione fiscale (Bonus ristrutturazione) ---
        # Beneficio di cassa (IRPEF) ripartito in quote costanti per i primi N anni.
        # Applicazione "ottica DCF": non altera il PnL, ma incrementa il cashflow.
        det_annual = 0.0
        if det_enabled and det_rate > 0.0 and det_years > 0 and det_cap_eur > 0.0:
            eligible_capex = capex_pv0 + capex_bess + capex_wind0
            # Massimale per unità immobiliare; se num>1, scala il cap di conseguenza.
            cap_eff = det_cap_eur * float(n)
            det_base = min(float(eligible_capex), float(cap_eff))
            det_total = det_base * det_rate
            det_annual = det_total / float(det_years)

        # --- Replacement CAPEX schedule ---
        repl = {y: 0.0 for y in years}
        # BESS replacement
        if bool(assets_bess.loc[mid, "bess_replacement"]):
            life = int(assets_bess.loc[mid, "bess_life_years"])
            if life > 0:
                k = life
                while k < horizon:
                    repl_year = k + 1
                    if repl_year <= horizon:
                        repl[repl_year] += capex_bess
                    k += life
        # PV inverter replacement
        inv_year = assets_pv.loc[mid, "pv_inverter_repl_year"]
        if not pd.isna(inv_year):
            inv_year_int = int(inv_year)
            if 1 <= inv_year_int <= horizon:
                repl[inv_year_int] += float(assets_pv.loc[mid, "pv_inverter_repl_eur_per_kw"]) * float(assets_pv.loc[mid, "pv_installed_kw"])

        # WIND major replacement (e.g., gearbox/major overhaul)
        wind_repl_year = assets_wind.loc[mid, "wind_major_repl_year"]
        if not pd.isna(wind_repl_year):
            y_int = int(wind_repl_year)
            if 1 <= y_int <= horizon:
                repl[y_int] += float(assets_wind.loc[mid, "wind_major_repl_eur_per_kw"]) * float(assets_wind.loc[mid, "wind_installed_kw"])

        # --- Depreciation schedules ---
        # PV depreciation if capex PV included
        da_pv = {y: 0.0 for y in years}
        if bool(assets_pv.loc[mid, "pv_exists"]) and (not bool(assets_pv.loc[mid, "pv_is_sunk"])):
            capex_pv = capex_pv0
            life_pv = int(assets_pv.loc[mid, "pv_life_years"])
            if life_pv > 0 and capex_pv > 0:
                for y in range(1, min(life_pv, horizon) + 1):
                    da_pv[y] += capex_pv / float(life_pv)

        # WIND depreciation if capex WIND included
        da_wind = {y: 0.0 for y in years}
        if bool(assets_wind.loc[mid, "wind_exists"]) and (not bool(assets_wind.loc[mid, "wind_is_sunk"])):
            capex_wind = capex_wind0
            life_w = int(assets_wind.loc[mid, "wind_life_years"])
            if life_w > 0 and capex_wind > 0:
                for y in range(1, min(life_w, horizon) + 1):
                    da_wind[y] += capex_wind / float(life_w)

                # depreciation of major replacement (if any): starts year after replacement
                wind_repl_year = assets_wind.loc[mid, "wind_major_repl_year"]
                if not pd.isna(wind_repl_year):
                    y_rep = int(wind_repl_year)
                    if 1 <= y_rep <= horizon:
                        rep_amount = float(assets_wind.loc[mid, "wind_major_repl_eur_per_kw"]) * float(assets_wind.loc[mid, "wind_installed_kw"])
                        start = y_rep + 1
                        end = min(start + life_w - 1, horizon)
                        if start <= horizon and rep_amount > 0:
                            for yy in range(start, end + 1):
                                da_wind[yy] += rep_amount / float(life_w)

        # BESS depreciation
        da_bess = {y: 0.0 for y in years}
        life_b = int(assets_bess.loc[mid, "bess_life_years"])
        if life_b > 0 and capex_bess > 0:
            for y in range(1, min(life_b, horizon) + 1):
                da_bess[y] += capex_bess / float(life_b)
            # depreciation of replacements
            if bool(assets_bess.loc[mid, "bess_replacement"]):
                k = life_b
                while k < horizon:
                    repl_year = k + 1
                    # depreciation starts the year after the replacement
                    start = repl_year + 1
                    end = min(start + life_b - 1, horizon)
                    if start <= horizon:
                        for y in range(start, end + 1):
                            da_bess[y] += capex_bess / float(life_b)
                    k += life_b

        # enable/disable depreciation by user_class
        uclass = str(users.loc[mid, "user_class"]).lower()
        depr_enabled = True
        if uclass in tax_by_class.index:
            depr_enabled = bool(tax_by_class.loc[uclass, "depreciation_enabled"])
        if not depr_enabled:
            da_pv = {y: 0.0 for y in years}
            da_pv = {y: 0.0 for y in years}
            da_wind = {y: 0.0 for y in years}

        # --- Tax config (class + override) ---
        tax_enabled = False
        tax_rate = 0.0
        allow_cf = False
        loss_years = 0
        if uclass in tax_by_class.index:
            tax_enabled = bool(tax_by_class.loc[uclass, "tax_enabled"])
            tax_rate = float(tax_by_class.loc[uclass, "tax_rate_effective"])
            allow_cf = bool(tax_by_class.loc[uclass, "allow_tax_loss_carryforward"])
            loss_years = int(tax_by_class.loc[uclass, "loss_carry_years"])
        if tax_overrides is not None and mid in tax_overrides.index:
            ov = tax_overrides.loc[mid]
            if not pd.isna(ov.get("tax_enabled_override")):
                tax_enabled = bool(ov.get("tax_enabled_override"))
            if not pd.isna(ov.get("tax_rate_override")):
                tax_rate = float(ov.get("tax_rate_override"))

        # --- build yearly PnL & CashFlow ---
        nol_balance: float = 0.0
        nol_age: List[Tuple[int, float]] = []  # list of (year_generated, amount)

        # year 0 cashflow
        cf = {y: 0.0 for y in years}
        cf[0] = -capex0

        tip_esc = float(policy.loc[0, "tip_escalation_rate"]) if "tip_escalation_rate" in policy.columns else 0.0
        tiad_esc = float(policy.loc[0, "tiad_escalation_rate"]) if "tiad_escalation_rate" in policy.columns else 0.0

        for y in range(1, horizon + 1):
            # escalations
            buy_mult = (1.0 + esc_buy) ** (y - 1)
            sell_mult = (1.0 + esc_sell) ** (y - 1)
            opex_mult = (1.0 + esc_opex) ** (y - 1)
            tip_mult = (1.0 + tip_esc) ** (y - 1)
            tiad_mult = (1.0 + tiad_esc) ** (y - 1)

            rev_sell = rev_sell_y1 * sell_mult
            # Incentivi/valorizzazione: per coerenza regolatoria NON sono agganciati al prezzo di vendita.
            rev_tip = (rev_tip_y1 * tip_mult) if y <= incentive_years else 0.0
            rev_tiad = (rev_tiad_y1 * tiad_mult) if y <= incentive_years else 0.0
            cost_buy = cost_buy_y1 * buy_mult
            # fixed fees (assumed escalated with buy)
            cost_buy += annual_buy_fixed * buy_mult

            # Risparmio bolletta: autoconsumo valorizzato al prezzo di acquisto (escalato come buy)
            savings_aut = savings_aut_y1 * buy_mult

            # Detrazione fiscale (quota annua utilizzata)
            det_used = det_annual if y <= det_years else 0.0

            opex = opex_y1 * opex_mult

            # EBITDA
            rev = rev_sell + rev_tip + rev_tiad
            costs = cost_buy + opex
            ebitda = rev - costs

            # depreciation
            da = float(da_pv.get(y, 0.0) + da_wind.get(y, 0.0) + da_bess.get(y, 0.0))
            ebit = ebitda - da

            # taxes
            tax = 0.0
            taxable = ebit
            if tax_enabled:
                if allow_cf:
                    # update NOL
                    if taxable < 0:
                        # add to carryforward pool
                        amt = -taxable
                        nol_age.append((y, amt))
                        taxable_adj = 0.0
                    else:
                        # consume oldest first, respecting max carry years
                        # purge expired
                        if loss_years > 0:
                            nol_age = [(yy, a) for (yy, a) in nol_age if (y - yy) <= loss_years]
                        remaining = taxable
                        new_nol_age = []
                        for yy, a in sorted(nol_age, key=lambda t: t[0]):
                            if remaining <= 0:
                                new_nol_age.append((yy, a))
                                continue
                            use = min(a, remaining)
                            remaining -= use
                            left = a - use
                            if left > 1e-9:
                                new_nol_age.append((yy, left))
                        nol_age = new_nol_age
                        taxable_adj = max(0.0, remaining)
                    tax = taxable_adj * tax_rate
                else:
                    tax = max(0.0, taxable) * tax_rate

            ni = ebit - tax

            pnl_rows.append(
                {
                    "member_id": mid,
                    "year": y,
                    "rev_sell": rev_sell,
                    "rev_tip": rev_tip,
                    "rev_tiad": rev_tiad,
                    "cost_buy": cost_buy,
                    "opex": opex,
                    "ebitda": ebitda,
                    "depreciation": da,
                    "ebit": ebit,
                    "tax": tax,
                    "net_income": ni,
                }
            )

            # cashflow
            capex_repl = float(repl.get(y, 0.0))
            # Ottica DCF: escludi l'esborso per acquisto energia dal cashflow (add-back cost_buy)
            # e includi il risparmio da autoconsumo (bolletta evitata).
            cf[y] = ni + da - capex_repl + cost_buy + savings_aut + det_used

            cash_rows.append(
                {
                    "member_id": mid,
                    "year": y,
                    "capex_replacement": capex_repl,
                    "savings_autocons": savings_aut,
                    "tax_detraction": det_used,
                    "cashflow": cf[y],
                }
            )

        # year 0 row
        cash_rows.append(
            {
                "member_id": mid,
                "year": 0,
                "capex_replacement": 0.0,
                "savings_autocons": 0.0,
                "tax_detraction": 0.0,
                "cashflow": cf[0],
            }
        )

        # KPIs
        cash_series = pd.Series([cf[y] for y in years], index=years, dtype=float)
        kpi = _compute_kpis(cash_series.values.tolist(), discount_rate=disc)
        kpi_rows.append({"member_id": mid, **kpi})

    pnl = pd.DataFrame(pnl_rows)
    pnl = pnl.set_index(["member_id", "year"]).sort_index()

    cash = pd.DataFrame(cash_rows)
    cash = cash.set_index(["member_id", "year"]).sort_index()
    # add discounting columns
    cash = cash.copy()
    cash["discount_factor"] = (1.0 / (1.0 + disc)) ** cash.index.get_level_values("year").astype(float)
    cash["dcf"] = cash["cashflow"] * cash["discount_factor"]
    cash["cum_cashflow"] = cash.groupby(level=0)["cashflow"].cumsum()
    cash["cum_dcf"] = cash.groupby(level=0)["dcf"].cumsum()

    kpis = pd.DataFrame(kpi_rows)
    return pnl, cash, kpis


def _aggregate_total_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Somma per anno (tutti i membri) mantenendo le colonne."""
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        out = df.groupby(level="year").sum(numeric_only=True)
        out.index.name = "year"
        return out
    # already aggregated
    return df


def _compute_kpis_for_statement(cash_total: pd.DataFrame, *, discount_rate: float) -> pd.DataFrame:
    if cash_total.empty:
        return pd.DataFrame([])
    years = cash_total.index.astype(int).tolist()
    cf = cash_total["cashflow"].reindex(years).values.tolist()
    kpi = _compute_kpis(cf, discount_rate=discount_rate)
    return pd.DataFrame([{"scope": "total", **kpi}])


def _compute_kpis(cashflows: List[float], *, discount_rate: float) -> Dict[str, float]:
    cf = np.array(cashflows, dtype=float)
    years = np.arange(len(cf), dtype=float)
    disc_f = (1.0 / (1.0 + discount_rate)) ** years
    dcf = cf * disc_f
    npv = float(dcf.sum())

    irr = _irr_bisect(cf)
    pb = _payback_year(cf)
    dpb = _payback_year(dcf)
    return {
        "npv": npv,
        "irr": float(irr) if irr is not None else float("nan"),
        "payback_year": float(pb) if pb is not None else float("nan"),
        "discounted_payback_year": float(dpb) if dpb is not None else float("nan"),
    }


def _payback_year(cashflows: np.ndarray) -> Optional[int]:
    cum = np.cumsum(cashflows)
    for i, v in enumerate(cum):
        if v >= 0:
            return i
    return None


def _irr_bisect(cashflows: np.ndarray) -> Optional[float]:
    """IRR via bisezione su NPV(r) = 0.

    Ritorna None se non è possibile braccare una radice (es. tutti CF >= 0 o tutti <= 0).
    """
    # must have sign change in cumulative sense
    if np.all(cashflows >= 0) or np.all(cashflows <= 0):
        return None

    def npv_rate(r: float) -> float:
        denom = (1.0 + r) ** np.arange(len(cashflows), dtype=float)
        return float(np.sum(cashflows / denom))

    # bracket
    lo, hi = -0.95, 5.0
    f_lo = npv_rate(lo)
    f_hi = npv_rate(hi)
    # try expand hi
    it = 0
    while f_lo * f_hi > 0 and it < 20:
        hi *= 2.0
        f_hi = npv_rate(hi)
        it += 1
    if f_lo * f_hi > 0:
        return None

    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = npv_rate(mid)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0