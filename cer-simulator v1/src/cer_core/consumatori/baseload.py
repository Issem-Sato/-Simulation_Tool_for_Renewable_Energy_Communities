from __future__ import annotations

"""baseload.py — Modello di carichi base (frigo, freezer, router, standby).

Il modello è pensato per essere:
- auto-contenuto (può essere richiamato sia da Streamlit che da script),
- coerente con l'approccio bottom-up usato per lavanderia e cucina,
- sufficientemente semplice da parametrizzare via questionario.

Tutte le potenze sono espresse in kW nelle serie risultanti.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses di configurazione ad alto livello (vicine al questionario)
# ---------------------------------------------------------------------------


@dataclass
class ContinuousBaseConfig:
    """Carichi continui / quasi continui.

    I parametri sono volutamente "ad alto livello"; i dettagli (potenze,
    durate ON/OFF) sono ricavati internamente da preset.
    """

    n_fridges: int = 1
    fridge_efficiency: str = "standard"  # "modern" | "standard" | "old"
    has_separate_freezer: bool = False
    freezer_efficiency: str = "standard"  # come sopra
    has_router: bool = True
    n_other_always_on: int = 0  # es. NAS, sistemi di sicurezza, hub smart


@dataclass
class StandbyConfig:
    """Carichi in standby / elettronica non attiva."""

    n_tvs: int = 1
    n_consoles: int = 0
    n_pcs: int = 0
    n_decoders: int = 0
    other_standby_w: float = 0.0  # eventuale extra in Watt


@dataclass
class BaseLoadConfig:
    """Configurazione complessiva per i carichi base."""

    continuous: ContinuousBaseConfig = field(default_factory=ContinuousBaseConfig)
    standby: StandbyConfig = field(default_factory=StandbyConfig)
    seed: int = 0  # seed per la parte stocastica (frigo/freezer)


# ---------------------------------------------------------------------------
# Preset parametrici (approx. dalla letteratura)
# ---------------------------------------------------------------------------


FRIDGE_PRESETS: Dict[str, Dict[str, float]] = {
    # valori indicativi: potenza ON (W) e durate medie ON/OFF (minuti)
    "modern": {"p_on_w": 80.0, "mean_on_min": 10.0, "mean_off_min": 35.0},
    "standard": {"p_on_w": 100.0, "mean_on_min": 15.0, "mean_off_min": 30.0},
    "old": {"p_on_w": 120.0, "mean_on_min": 20.0, "mean_off_min": 25.0},
}

FREEZER_PRESETS: Dict[str, Dict[str, float]] = {
    "modern": {"p_on_w": 90.0, "mean_on_min": 12.0, "mean_off_min": 40.0},
    "standard": {"p_on_w": 110.0, "mean_on_min": 18.0, "mean_off_min": 32.0},
    "old": {"p_on_w": 130.0, "mean_on_min": 24.0, "mean_off_min": 26.0},
}

ROUTER_W: float = 10.0
OTHER_ALWAYS_ON_W: float = 8.0  # per ciascun dispositivo "always on"

# standby tipici
TV_STBY_W: float = 3.0
CONSOLE_STBY_W: float = 4.0
PC_STBY_W: float = 5.0
DECODER_STBY_W: float = 7.0


# ---------------------------------------------------------------------------
# Utilità interne
# ---------------------------------------------------------------------------


def _two_state_markov_profile(
    idx: pd.DatetimeIndex,
    mean_on_min: float,
    mean_off_min: float,
    power_w: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Genera un profilo ON/OFF con medie delle durate ON/OFF fissate.

    Modello Markoviano a 2 stati:
    - stato 1: ON
    - stato 0: OFF

    Le probabilità di transizione per step sono scelte in modo che
    l'aspettativa delle durate in minuti sia ~mean_on_min / mean_off_min.
    """
    if len(idx) == 0:
        return pd.Series([], dtype=float)

    # passo temporale in minuti (assumiamo equispaziato)
    if len(idx) > 1:
        dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
    else:
        # fallback: 60 minuti se singolo punto
        dt_min = 60.0

    # probabilità di uscita da ON e OFF per timestep
    p_on_off = min(1.0, dt_min / max(mean_on_min, 1e-3))
    p_off_on = min(1.0, dt_min / max(mean_off_min, 1e-3))

    state = 1  # partiamo in ON
    values = []
    for _ in range(len(idx)):
        values.append(power_w / 1000.0 if state == 1 else 0.0)  # kW
        r = rng.random()
        if state == 1:
            if r < p_on_off:
                state = 0
        else:
            if r < p_off_on:
                state = 1

    return pd.Series(values, index=idx, dtype=float)


def _get_fridge_params(eff: str) -> Dict[str, float]:
    eff = eff or "standard"
    if eff not in FRIDGE_PRESETS:
        eff = "standard"
    return FRIDGE_PRESETS[eff]


def _get_freezer_params(eff: str) -> Dict[str, float]:
    eff = eff or "standard"
    if eff not in FREEZER_PRESETS:
        eff = "standard"
    return FREEZER_PRESETS[eff]


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------


def build_baseload_profiles(
    idx: pd.DatetimeIndex,
    cfg: BaseLoadConfig,
) -> Dict[str, pd.Series]:
    """Restituisce i profili di potenza dei carichi base.

    Parameters
    ----------
    idx:
        DatetimeIndex del periodo di simulazione.
    cfg:
        Configurazione dei carichi base.

    Returns
    -------
    dict con chiavi:
        - "fridge": tutti i frigoriferi
        - "freezer": freezer separati (se presenti)
        - "router": modem/router
        - "always_on": altri carichi sempre accesi
        - "standby": standby/elettronica
        - "aggregated": somma di tutte le componenti
    """
    if len(idx) == 0:
        empty = pd.Series([], dtype=float)
        return {
            "fridge": empty,
            "freezer": empty,
            "router": empty,
            "always_on": empty,
            "standby": empty,
            "aggregated": empty,
        }

    rng = np.random.default_rng(int(cfg.seed or 0))

    # --- frigoriferi ---
    cont = cfg.continuous
    fridge_params = _get_fridge_params(cont.fridge_efficiency)
    n_fridges = max(int(cont.n_fridges or 0), 0)
    fridge_profile = pd.Series(0.0, index=idx, dtype=float)
    for _ in range(n_fridges):
        p = _two_state_markov_profile(
            idx,
            mean_on_min=fridge_params["mean_on_min"],
            mean_off_min=fridge_params["mean_off_min"],
            power_w=fridge_params["p_on_w"],
            rng=rng,
        )
        fridge_profile = fridge_profile.add(p, fill_value=0.0)

    # --- freezer separato ---
    freezer_profile = pd.Series(0.0, index=idx, dtype=float)
    if cont.has_separate_freezer:
        freezer_params = _get_freezer_params(cont.freezer_efficiency)
        # un solo freezer separato (molto raro averne due)
        freezer_profile = _two_state_markov_profile(
            idx,
            mean_on_min=freezer_params["mean_on_min"],
            mean_off_min=freezer_params["mean_off_min"],
            power_w=freezer_params["p_on_w"],
            rng=rng,
        )

    # --- router / altri always-on ---
    router_profile = pd.Series(
        ROUTER_W / 1000.0 if cont.has_router else 0.0,
        index=idx,
        dtype=float,
    )

    n_other = max(int(cont.n_other_always_on or 0), 0)
    always_on_w = n_other * OTHER_ALWAYS_ON_W
    always_on_profile = pd.Series(always_on_w / 1000.0, index=idx, dtype=float)

    # --- standby / elettronica ---
    stb = cfg.standby
    standby_w = (
        max(int(stb.n_tvs or 0), 0) * TV_STBY_W
        + max(int(stb.n_consoles or 0), 0) * CONSOLE_STBY_W
        + max(int(stb.n_pcs or 0), 0) * PC_STBY_W
        + max(int(stb.n_decoders or 0), 0) * DECODER_STBY_W
        + max(float(stb.other_standby_w or 0.0), 0.0)
    )
    standby_profile = pd.Series(standby_w / 1000.0, index=idx, dtype=float)

    aggregated = (
        fridge_profile
        .add(freezer_profile, fill_value=0.0)
        .add(router_profile, fill_value=0.0)
        .add(always_on_profile, fill_value=0.0)
        .add(standby_profile, fill_value=0.0)
    )

    return {
        "fridge": fridge_profile,
        "freezer": freezer_profile,
        "router": router_profile,
        "always_on": always_on_profile,
        "standby": standby_profile,
        "aggregated": aggregated,
    }
