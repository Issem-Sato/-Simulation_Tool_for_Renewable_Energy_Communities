"""cer_app.schede_consumatori.lavanderia

Scheda Streamlit: **Lavanderia** (lavatrice + asciugatrice).

Questa scheda è una “tab” della pagina ``cer_app/pages/2_Consumatori.py`` e svolge tre compiti:

1) **Raccogliere parametri** per lavatrice/asciugatrice dal registry per-consumer
   (``data/sessions/<SESSION>/consumers.json``) e salvarli nello stesso file.
2) **Invocare il modello core** ``cer_core.consumatori.lavanderia`` (funzione ``simulate``)
   per generare una curva di potenza su indice a 15 minuti.
3) **Persistenza per orchestrazione**: salvare la curva totale (washer + dryer) nella cache
   di sessione per consumatore, come::

       data/sessions/<SESSION>/cache/consumer_<ID>/laundry_total.csv

   Tale file viene poi letto e sommato dall’orchestratore ``2_Consumatori.py`` assieme alle
   altre componenti (cucina, baseload, occupancy, clima).

Contratti e assunzioni
----------------------
- **Timebase**: ``INDEX`` è un ``pandas.DatetimeIndex`` tz-aware in **UTC**, con frequenza
  ``15min`` (iniettato dall’orchestratore). La curva restituita da ``L.simulate`` viene
  reindicizzata implicitamente su ``INDEX``.
- **Unità**: le curve sono **potenze** in **kW** (valori medi sul time-step). L’energia (kWh)
  è un post-processing: ``sum(P_kW) * dt_hours``.
- **Riproducibilità**: il seed della simulazione è derivato da ``BASE_SEED`` + consumer id +
  device name tramite ``derive_seed(...)`` (iniettato). A parità di seed, indice e parametri
  si ottiene lo stesso output.
- **Finestra di utilizzo** (start matrix): l’utente seleziona bande orarie settimanali su una
  griglia 7×N (giorni × bande). La rappresentazione persistita in JSON è un mapping
  ``day_idx -> list[(h_start, h_end)]`` (chiavi int o string).

Variabili/funzioni iniettate
----------------------------
Per compatibilità con la struttura esistente, questo modulo **non** importa direttamente
utility e stato di sessione: ``2_Consumatori.py`` imposta a runtime i seguenti simboli nel
namespace del modulo (dependency injection via assegnamento di attributi):

- ``SESSION_DIR`` (Path), ``INDEX`` (DatetimeIndex UTC), ``BASE_SEED`` (int)
- ``consumer`` (dict), ``consumers`` (list[dict])
- ``T_AIR_15`` (Series, °C) — passato al core per eventuale stagionalità/condizioni
- helpers: ``ensure_device``, ``save_consumers_json``, ``derive_seed``,
  ``start_matrix_editor``, ``dict_to_matrix``, ``matrix_to_dict``,
  ``count_weekly_slots_from_grid``, ``prepare_curve_for_plot``, ``compute_kpis``
- parametri UI: ``curve_view_mode`` / ``curve_view_month``

Failure modes (gestiti a UI)
----------------------------
- Se l’utente richiede ``cycles_per_week`` > slot disponibili nella griglia, viene mostrato
  un errore e l’utente deve correggere la configurazione.
- Eccezioni dal core (es. slot insufficienti, input incoerenti) sono catturate e mostrate con
  ``st.error`` senza interrompere l’app.
"""

from __future__ import annotations

import streamlit as st

# Core model (simulate + energy classes + parsing start_matrix)
from cer_core.consumatori import lavanderia as L


# ---------------------------------------------------------------------
# Default device dictionaries (persistiti in consumers.json tramite ensure_device)
# ---------------------------------------------------------------------
#
# NOTE:
# - i campi sono intenzionalmente ridondanti e “flat” perché vengono serializzati in JSON;
# - la UI salva anche ``start_matrix`` (dict day->bands) e parametri specifici device.
#
W_DEF = {
    "present": True,
    "n_devices": 1,
    "modes_selected": ["standard"],
    "cycles_per_week": 0,          # 0 = nessun utilizzo
    "start_matrix": None,
    "P_nominal_W": None,
    "energy_class": None,
    "seed": 12345,
}

D_DEF = {
    "present": True,
    "n_devices": 1,
    "cycles_per_week": 0,          # 0 = nessun utilizzo
    "start_matrix": None,
    "seasonality": "tutto_anno",   # solo per dryer
    "P_nominal_W": None,
    "energy_class": None,
    "seed": 54321,
}


def laundry_device_config(device_name: str, title: str, defaults: dict) -> tuple[bool, dict | None]:
    """Render UI per configurare un device di lavanderia e produrre gli input per il core.

    Parametri
    ---------
    device_name:
        ``"washer"`` o ``"dryer"`` (coerente con ``cer_core.consumatori.lavanderia.simulate``).
    title:
        Etichetta leggibile per la UI (es. "Lavatrice").
    defaults:
        Dizionario di default con chiavi attese da questa scheda.

    Returns
    -------
    (present, dev_inputs):
        - ``present``: True se il device è attivo per il consumer.
        - ``dev_inputs``: dict pronto da passare a ``L.simulate(..., inputs=...)``.
          Se ``present`` è False ritorna ``None``.

    Side effects
    ------------
    Aggiorna il registry ``consumers.json`` quando l’utente salva i parametri o abilita
    un device precedentemente disattivato.
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

    # ---- Parametri numerici (in UI manteniamo W per potenza nominale) ----
    cols = st.columns(3)
    dev["n_devices"] = cols[0].number_input(
        "Numero dispositivi",
        min_value=1,
        value=int(dev.get("n_devices", 1)),
        step=1,
        key=f"c{consumer_id}_{device_name}_n",
    )

    dev["cycles_per_week"] = cols[2].number_input(
        "Cicli a settimana (0 = nessun utilizzo)",
        min_value=0,
        value=int(dev.get("cycles_per_week") or 0),
        step=1,
        key=f"c{consumer_id}_{device_name}_cpw",
    )

    cols2 = st.columns(3)

    # Dryer only: seasonality (washer non deve esporre questa chiave)
    if device_name == "dryer":
        season_opts = ["tutto_anno", "inverno"]
        current_season = dev.get("seasonality", "tutto_anno")
        if current_season not in season_opts:
            current_season = "tutto_anno"
        dev["seasonality"] = cols2[0].selectbox(
            "Stagionalità",
            season_opts,
            index=season_opts.index(current_season),
            key=f"c{consumer_id}_{device_name}_season",
        )
    else:
        dev.pop("seasonality", None)

    # Energy class + nominal power
    options = list(L.ENERGY_CLASS_OPTIONS)
    current_class = dev.get("energy_class", None)
    if current_class not in options:
        current_class = options[0] if options else None
    dev["energy_class"] = cols2[1].selectbox(
        "Classe energetica",
        options,
        index=options.index(current_class),
        key=f"c{consumer_id}_{device_name}_class",
    )

    dev["P_nominal_W"] = cols2[2].number_input(
        "Potenza nominale (W)",
        min_value=0,
        value=int(dev.get("P_nominal_W") or 0),
        step=50,
        key=f"c{consumer_id}_{device_name}_pnom",
    )

    # Modes selection (opzioni definita nel core per compatibilità)
    if device_name in ("washer", "dryer"):
        dev["modes_selected"] = st.multiselect(
            "Modalità di utilizzo",
            L.WASHER_MODE_OPTIONS,
            default=dev.get("modes_selected") or ["standard"],
            key=f"c{consumer_id}_{device_name}_modes",
        )

    # -------------------- Weekly start windows (7×N grid) --------------------
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

    # ---- Input dict per il core ----
    dev_inputs = dict(dev)
    if device_name == "washer":
        # Pulizia per robustezza: alcune chiavi sono solo per altri device o legacy.
        dev_inputs.pop("cycle_duration_min", None)
        dev_inputs.pop("seasonality", None)

    dev_inputs["start_matrix"] = matrix_to_dict(grid)
    return True, dev_inputs


def laundry_panel():
    """Tab Streamlit: costruisce curva totale (lavatrice + asciugatrice) e salva cache.

    Returns
    -------
    pandas.Series | None
        Serie su ``INDEX`` (15min, UTC) con potenza totale **kW** e nome ``"laundry_total"``.
        Ritorna ``None`` se non ci sono dispositivi attivi o se la simulazione fallisce.

    Side effects
    ------------
    Quando l’utente clicca “Calcola curva…”, salva::

        cache/consumer_<ID>/laundry_total.csv

    con colonna ``timestamp`` + valori (kW).
    """
    consumer_id = consumer.get("id")

    with st.expander("🧺 Lavanderia (totale: lavatrice + asciugatrice)", expanded=True):
        tab_w, tab_d = st.tabs(["Lavatrice", "Asciugatrice"])

        with tab_w:
            w_present, w_inputs = laundry_device_config("washer", "Lavatrice", W_DEF)

        with tab_d:
            d_present, d_inputs = laundry_device_config("dryer", "Asciugatrice", D_DEF)

        st.markdown("---")

        laundry_curve = None
        if st.button("⚙️ Calcola curva LAVANDERIA (totale)", key=f"c{consumer_id}_laundry_total_calc"):
            try:
                curves = []

                if w_present and w_inputs is not None:
                    w_curve = (
                        L.simulate(
                            "washer",
                            index=INDEX,
                            inputs=w_inputs,
                            temp=T_AIR_15,
                            seed=derive_seed(BASE_SEED, "consumer", consumer_id, "washer"),
                        )
                        .rename("washer")
                        .astype(float)
                    )
                    curves.append(w_curve)

                if d_present and d_inputs is not None:
                    d_curve = (
                        L.simulate(
                            "dryer",
                            index=INDEX,
                            inputs=d_inputs,
                            temp=T_AIR_15,
                            seed=derive_seed(BASE_SEED, "consumer", consumer_id, "dryer"),
                        )
                        .rename("dryer")
                        .astype(float)
                    )
                    curves.append(d_curve)

                if not curves:
                    st.info("Nessun dispositivo lavanderia attivo per questo consumatore.")
                    return None

                # Somma totale
                laundry_curve = curves[0]
                for c in curves[1:]:
                    laundry_curve = laundry_curve.add(c, fill_value=0.0)
                laundry_curve = laundry_curve.rename("laundry_total").astype(float)

                # Cache CSV totale (usata da 2_Consumatori.py)
                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer_id}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                laundry_curve.to_csv(cache_dir / "laundry_total.csv", index=True, index_label="timestamp")

                # Plot singolo (rispetta view mode “anno”/“mensile” imposto in pagina)
                laundry_plot = prepare_curve_for_plot(laundry_curve, curve_view_mode, curve_view_month)
                st.line_chart(laundry_plot)

                # KPI energetici
                kwh_periodo, kwh_giorno, kwh_annuo = compute_kpis(laundry_curve, INDEX)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lavanderia: kWh periodo", f"{kwh_periodo:.1f}")
                c2.metric("Lavanderia: kWh/anno stimato", f"{kwh_annuo:.0f}")
                c3.metric("Lavanderia: media giornaliera", f"{kwh_giorno:.2f} kWh/g")

                # Download CSV totale 15 min
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
