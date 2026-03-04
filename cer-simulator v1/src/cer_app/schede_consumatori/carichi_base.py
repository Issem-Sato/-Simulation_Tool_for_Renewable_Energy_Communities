"""cer_app.schede_consumatori.carichi_base

Scheda Streamlit: **Carichi base** (frigoriferi/freezer, router, dispositivi always-on e standby).

Questa scheda appartiene al flusso *Consumatori* (pagina ``cer_app/pages/2_Consumatori.py``) e si occupa di:
- raccogliere parametri di configurazione dal registry ``consumers.json`` (per-consumer);
- invocare il modello core ``cer_core.consumatori.baseload`` per generare una curva a 15 minuti;
- salvare la curva in cache come CSV per poter essere sommata da ``2_Consumatori.py``.

### Contratti e assunzioni (importanti per riproducibilità e coerenza CER)
- **Timebase**: l'indice ``INDEX`` deve essere un ``pd.DatetimeIndex`` *timezone-aware* in **UTC** con passo tipico
  di **15 minuti**. La scheda non costruisce l'indice: viene iniettato dall'orchestratore.
- **Unità**: le curve prodotte sono **potenze in kW** (valore medio sul time-step). L'energia sul periodo si ottiene come
  ``kWh = sum(P_kW) * dt_hours``.
- **Riproducibilità**: il seed del modello viene derivato da un seed di sessione persistente (``BASE_SEED``) e dal
  ``consumer_id`` tramite ``derive_seed``; a parità di sessione/configurazione, l'output è deterministico.
- **I/O su disco** (compatibilità):
  - lettura/scrittura parametri: ``data/sessions/<SESSION>/consumers.json`` (gestita altrove; qui si usa ``save_consumers_json``)
  - scrittura curva componente: ``data/sessions/<SESSION>/cache/consumer_<id>/baseload.csv``
    con colonna ``baseload`` e indice ``timestamp`` (UTC).

### Dipendenze (iniezione di globali)
Per mantenere moduli UI indipendenti dal contesto Streamlit globale, ``2_Consumatori.py`` esegue una *dependency injection*
assegnando a questo modulo alcune variabili/funzioni (vedi ``_bind_panel_globals``). In particolare, ``baseload_panel()``
si aspetta che siano presenti nel namespace:
``SESSION_DIR``, ``INDEX``, ``BASE_SEED``, ``consumer``, ``consumers``,
``ensure_device()``, ``derive_seed()``, ``sanitize_for_dataclass()``, ``save_consumers_json()``,
``curve_view_mode`` e ``curve_view_month``.

Nota: questo modulo contiene anche utility per l'editing di una matrice 7x24 di stati di occupazione; tali funzioni possono
essere riutilizzate da altre schede, ma non sono strettamente necessarie al calcolo del baseload.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from cer_core.consumatori import baseload as B

# -------------------------------------------------------------------------
# Default device config (persistito in consumers.json)
# -------------------------------------------------------------------------

BASELOAD_DEF = {
    "present": True,
    "continuous": {
        "n_fridges": 1,
        "fridge_efficiency": "standard",
        "has_separate_freezer": False,
        "freezer_efficiency": "standard",
        "has_router": True,
        "n_other_always_on": 0,
    },
    "standby": {
        "n_tvs": 1,
        "n_consoles": 0,
        "n_pcs": 0,
        "n_decoders": 1,
        "other_standby_w": 0.0,
    },
    # Seed legacy mostrato in UI: il seed effettivo passato al core viene derivato con derive_seed(BASE_SEED, ...).
    "seed": 12345,
}


def baseload_panel():
    """Renderizza la scheda Streamlit *Carichi base* e (opzionalmente) calcola la curva.

    Output
    ------
    pd.Series | None
        Se l'utente clicca "Calcola curva", ritorna una ``pd.Series`` indicizzata su ``INDEX`` (UTC, 15 min) con nome
        ``"baseload"`` e unità **kW**. In caso contrario, ritorna ``None``.

    Side effects (compatibilità del simulatore)
    -------------------------------------------
    - Aggiorna ``consumers.json`` tramite ``save_consumers_json`` quando si clicca "Salva parametri".
    - Scrive la curva in cache in: ``SESSION_DIR/cache/consumer_<id>/baseload.csv``.
    """
    consumer_id = consumer.get("id")
    dev_b = ensure_device(consumer, "baseload", BASELOAD_DEF)

    with st.expander("⚡ Carichi base (frigo, router, standby)", expanded=True):
        dev_b["present"] = st.checkbox(
            "Includi i carichi base per questo consumatore",
            value=bool(dev_b.get("present", True)),
            key=f"c{consumer_id}_baseload_present",
        )
        if not dev_b["present"]:
            if st.button("Abilita carichi base", key=f"c{consumer_id}_baseload_enable"):
                dev_b["present"] = True
                save_consumers_json(SESSION_DIR, consumers)
                st.success("Carichi base abilitati per questo consumatore.")
            return None

        cont = dev_b.setdefault("continuous", dict(BASELOAD_DEF["continuous"]))
        stb = dev_b.setdefault("standby", dict(BASELOAD_DEF["standby"]))

        st.markdown("### Carichi continui")
        c1, c2, c3 = st.columns(3)
        cont["n_fridges"] = c1.number_input(
            "Numero di frigoriferi",
            min_value=0,
            max_value=4,
            value=int(cont.get("n_fridges", BASELOAD_DEF["continuous"]["n_fridges"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_fridges",
        )
        FR_EFF_OPTS = [
            ("modern", "Moderno (classe alta, <10 anni)"),
            ("standard", "Standard (10–20 anni)"),
            ("old", "Vecchio (>20 anni)"),
        ]
        current_eff = cont.get("fridge_efficiency", BASELOAD_DEF["continuous"]["fridge_efficiency"])
        if current_eff not in [o[0] for o in FR_EFF_OPTS]:
            current_eff = "standard"
        cont["fridge_efficiency"] = c2.selectbox(
            "Efficienza frigorifero principale",
            options=[o[0] for o in FR_EFF_OPTS],
            index=[o[0] for o in FR_EFF_OPTS].index(current_eff),
            format_func=lambda k: dict(FR_EFF_OPTS)[k],
            key=f"c{consumer_id}_baseload_fridge_eff",
        )
        cont["has_separate_freezer"] = c3.checkbox(
            "Freezer separato presente",
            value=bool(cont.get("has_separate_freezer", BASELOAD_DEF["continuous"]["has_separate_freezer"])),
            key=f"c{consumer_id}_baseload_has_freezer",
        )

        cont["n_other_always_on"] = st.number_input(
            "Altri dispositivi sempre accesi (NAS, sicurezza, ecc.)",
            min_value=0,
            max_value=10,
            value=int(cont.get("n_other_always_on", BASELOAD_DEF["continuous"]["n_other_always_on"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_other",
        )

        cont["has_router"] = st.checkbox(
            "Modem/router WiFi acceso quasi sempre",
            value=bool(cont.get("has_router", BASELOAD_DEF["continuous"]["has_router"])),
            key=f"c{consumer_id}_baseload_router",
        )

        st.markdown("### Standby / elettronica")
        s1, s2, s3, s4 = st.columns(4)
        stb["n_tvs"] = s1.number_input(
            "Numero TV",
            min_value=0,
            max_value=5,
            value=int(stb.get("n_tvs", BASELOAD_DEF["standby"]["n_tvs"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_tvs",
        )
        stb["n_consoles"] = s2.number_input(
            "Console di gioco",
            min_value=0,
            max_value=3,
            value=int(stb.get("n_consoles", BASELOAD_DEF["standby"]["n_consoles"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_consoles",
        )
        stb["n_pcs"] = s3.number_input(
            "PC fissi / monitor",
            min_value=0,
            max_value=4,
            value=int(stb.get("n_pcs", BASELOAD_DEF["standby"]["n_pcs"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_pcs",
        )
        stb["n_decoders"] = s4.number_input(
            "Decoder / smart box",
            min_value=0,
            max_value=4,
            value=int(stb.get("n_decoders", BASELOAD_DEF["standby"]["n_decoders"])),
            step=1,
            key=f"c{consumer_id}_baseload_n_decoders",
        )

        stb["other_standby_w"] = st.number_input(
            "Altri standby stimati (W)",
            min_value=0.0,
            max_value=200.0,
            value=float(stb.get("other_standby_w", BASELOAD_DEF["standby"]["other_standby_w"])),
            step=5.0,
            key=f"c{consumer_id}_baseload_other_stby_w",
        )

        # Seed mostrato: il seed effettivo usato nel core è derivato (BASE_SEED + consumer_id + device).
        dev_b["seed"] = st.number_input(
            "Seed simulazione carichi base",
            min_value=0,
            max_value=1_000_000,
            value=int(dev_b.get("seed", BASELOAD_DEF["seed"])),
            step=1,
            key=f"c{consumer_id}_baseload_seed",
        )

        ac1, ac2 = st.columns(2)
        if ac1.button("💾 Salva parametri carichi base", key=f"c{consumer_id}_baseload_save"):
            save_consumers_json(SESSION_DIR, consumers)
            st.success("Parametri carichi base salvati.")

        curve = None
        if ac2.button("⚙️ Calcola curva carichi base", key=f"c{consumer_id}_baseload_calc"):
            try:
                cont_cfg = B.ContinuousBaseConfig(**sanitize_for_dataclass(B.ContinuousBaseConfig, cont))
                stb_cfg = B.StandbyConfig(**sanitize_for_dataclass(B.StandbyConfig, stb))
                bl_cfg = B.BaseLoadConfig(
                    continuous=cont_cfg,
                    standby=stb_cfg,
                    seed=derive_seed(BASE_SEED, "consumer", consumer_id, "baseload"),
                )
                profiles = B.build_baseload_profiles(INDEX, bl_cfg)

                # Curva completa a 15 minuti su tutto l'INDEX (kW).
                curve = profiles["aggregated"].rename("baseload").astype(float)

                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer.get('id')}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                curve.to_csv(cache_dir / "baseload.csv", index=True, index_label="timestamp")

                # Visualizzazione: annuale = aggregata a 1h, mensile = 15 min sul mese scelto
                if curve_view_mode == "annuale" or curve_view_month is None:
                    curve_plot = curve.resample("H").mean()
                else:
                    year, month = curve_view_month
                    mask_month = (curve.index.year == year) & (curve.index.month == month)
                    curve_plot = curve[mask_month]

                st.line_chart(curve_plot)

                # KPI sul periodo completo simulato (conversione kW -> kWh)
                if len(curve) > 1:
                    dt_hours = (curve.index[1] - curve.index[0]).total_seconds() / 3600.0
                else:
                    dt_hours = 0.0
                kwh_periodo = float(curve.sum() * dt_hours)

                giorni = (INDEX.max() - INDEX.min()).days + 1
                kwh_giorno = kwh_periodo / max(giorni, 1)
                kwh_annuo = kwh_giorno * 365.0

                cols = st.columns(4)
                cols[0].metric("Carichi base: kWh periodo", f"{kwh_periodo:.1f}")
                cols[1].metric("Carichi base: kWh/anno stimato", f"{kwh_annuo:.0f}")
                cols[2].metric("Carichi base: media giornaliera", f"{kwh_giorno:.2f} kWh/g")

                # download CSV a 15 minuti (periodo completo)
                csv_buf = curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
                cols[3].download_button(
                    "⬇️ CSV 15 min (carichi base – anno)",
                    data=csv_buf,
                    file_name=f"consumer_{consumer.get('id')}_baseload_15min.csv",
                    mime="text/csv",
                    key=f"dl_{consumer.get('id')}_baseload",
                )
            except Exception as e:
                st.error(f"Errore simulazione carichi base: {e}")

        return curve


# -------------------------------------------------------------------------
# Occupancy helpers (matrice 7x24 di stati)
# -------------------------------------------------------------------------

OCC_STATE_LABELS = {
    0: "Fuori casa",
    1: "A casa (sveglio)",
    2: "A casa (dorme)",
}
OCC_STATE_INV = {v: k for k, v in OCC_STATE_LABELS.items()}


def _default_occ_state_matrix_7x24() -> List[List[int]]:
    """Crea una matrice default 7x24 di stati di presenza.

    Convenzione stati (coerente con ``cer_core.consumatori.occupancy``):
    - 0: fuori casa
    - 1: a casa e sveglio
    - 2: a casa e dorme

    La matrice è indicizzata come ``[day_of_week][hour]`` con day_of_week in [0..6] (lun..dom).
    """
    out: List[List[int]] = []
    for dow in range(7):
        row = [2] * 24  # sleep
        if dow <= 4:  # Mon-Fri
            for h in range(7, 9):
                row[h] = 1
            for h in range(9, 18):
                row[h] = 0
            for h in range(18, 23):
                row[h] = 1
            row[23] = 2
        else:  # weekend
            for h in range(8, 23):
                row[h] = 1
            row[23] = 2
        out.append(row)
    return out


def _normalize_occ_state_matrix(mat: object) -> List[List[int]]:
    """Normalizza una matrice 7x24 garantendo shape e dominio {0,1,2}.

    Se l'input non rispetta il contratto (tipo errato, dimensioni errate, valori non interpretabili),
    ritorna il default prodotto da ``_default_occ_state_matrix_7x24``.
    """
    if not isinstance(mat, list) or len(mat) != 7:
        return _default_occ_state_matrix_7x24()
    out: List[List[int]] = []
    for r in mat:
        if not isinstance(r, list):
            return _default_occ_state_matrix_7x24()
        row = []
        for j in range(24):
            try:
                v = int(r[j])
            except Exception:
                v = 0
            v = 0 if v < 0 else 2 if v > 2 else v
            row.append(v)
        out.append(row)
    return out


def occ_state_matrix_editor(key: str, init_matrix: object) -> List[List[int]]:
    """Editor Streamlit per una matrice 7x24 di stati di occupazione.

    Parametri
    ---------
    key:
        Key Streamlit (serve per lo state del widget).
    init_matrix:
        Matrice iniziale (eventualmente malformata), normalizzata via ``_normalize_occ_state_matrix``.

    Ritorna
    -------
    List[List[int]]
        Matrice 7x24 normalizzata, con stati interi {0,1,2}.
    """
    mat = _normalize_occ_state_matrix(init_matrix)

    cols = [f"{h:02d}" for h in range(24)]
    df = pd.DataFrame(
        {c: [OCC_STATE_LABELS[mat[d][i]] for d in range(7)] for i, c in enumerate(cols)},
        index=DAY_NAMES,
    )

    col_cfg = {
        c: st.column_config.SelectboxColumn(
            c,
            options=list(OCC_STATE_LABELS.values()),
            required=True,
        )
        for c in cols
    }

    edited = st.data_editor(
        df,
        column_config=col_cfg,
        use_container_width=True,
        hide_index=False,
        key=key,
    )

    out: List[List[int]] = []
    for d in range(7):
        row: List[int] = []
        for c in cols:
            lbl = str(edited.iloc[d][c])
            row.append(int(OCC_STATE_INV.get(lbl, 0)))
        out.append(row)
    return out


def _ensure_occ_residents(dev_occ: dict, n_residents: int) -> None:
    """Helper legacy per inizializzare una lista di residenti in una config occupancy.

    Mantiene compatibilità con eventuali JSON già presenti: se ``dev_occ["residents"]`` non esiste,
    non è una lista, oppure non ha la cardinalità richiesta, crea/aggiusta la lista.

    Note
    ----
    La struttura dei dizionari ``resident`` è volutamente "UI-facing" (stringhe per intensità/preferenze)
    perché viene serializzata in ``consumers.json`` e reinterpretata dalla scheda Occupancy.
    """
    residents = dev_occ.get("residents")
    if not isinstance(residents, list):
        residents = []
    while len(residents) < n_residents:
        i = len(residents)
        residents.append(
            {
                "rid": i,
                "label": f"Residente {i+1}",
                "schedule": {"state_matrix_7x24": _default_occ_state_matrix_7x24()},
                "pc": {
                    "type": "laptop",
                    "intensity": "low",
                    "time_pref": "mixed",
                    "weekend_enabled": True,
                },
                "charging": {
                    "phone_intensity": "medium",
                    "preference": "night",
                    "charge_type": "standard",
                },
            }
        )
    if len(residents) > n_residents:
        residents = residents[:n_residents]
    dev_occ["residents"] = residents
