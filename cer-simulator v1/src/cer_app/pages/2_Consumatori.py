from __future__ import annotations

MODULE_DOC = """
Pagina Streamlit: consumatori (orchestratore)

Questa pagina implementa l'orchestrazione della parte *domanda* del simulatore CER.
Nel progetto la generazione delle curve di carico è divisa in due livelli:

- **UI** (`cer_app/schede_consumatori/*`): pannelli Streamlit per configurare i dispositivi e
  richiamare il calcolo.
- **Core** (`cer_core/consumatori/*`): modelli deterministici/stocastici che, dato un indice
  temporale e una configurazione, producono profili di potenza.

La presente pagina fornisce tre servizi trasversali:

1. **Sessione e meteo**: legge la directory di sessione tramite `get_paths()` e costruisce
   l'indice temporale `INDEX` (tipicamente 15 minuti, timezone-aware UTC) usando
   `load_time_index_and_meteo()`. Il file di input canonico è:

   - `data/sessions/<SESSION>/meteo_hourly.csv` (prodotto dalla Home), con colonna `timestamp` (UTC)
     e temperatura esterna (°C). La temperatura è interpolata a 15 minuti per alimentare il modulo Clima.

   Se la temperatura a 15 minuti non è disponibile, viene usato un *fallback* costante a **15°C**
   (solo per mantenere la UI operativa; il warning viene mostrato nel grafico dell'aggregato).

2. **Registry consumatori**: gestisce CRUD e persistenza di `consumers.json`, file che contiene l'elenco
   dei consumatori e le configurazioni dei device per ciascun consumatore.

   - `data/sessions/<SESSION>/consumers.json`

3. **Cache e aggregazione**: le schede salvano CSV per-consumatore in cache, con convenzione:

   - `data/sessions/<SESSION>/cache/consumer_<ID>/<component>.csv`

   Le componenti attese (se presenti) sono:
   `baseload.csv`, `occupancy_15min.csv`, `kitchen.csv`, `laundry_total.csv`, `climate.csv`.

   Ogni CSV ha una colonna `timestamp` (UTC) e una colonna numerica che rappresenta **potenza** in **kW**.
   L'energia (kWh) viene derivata a valle come `sum(P_kW) * dt_hours`.

Nota architetturale (compatibilità):
-----------------------------------
Le schede UI sono state sviluppate come moduli che si aspettano alcune variabili globali
(`SESSION_DIR`, `INDEX`, `BASE_SEED`, ecc.). Per preservare struttura e compatibilità, questa pagina
usa una forma di *dependency injection* "per binding" (funzione `_bind_panel_globals`) che assegna
tali variabili ai moduli importati.

Vincoli temporali e unità:
--------------------------
- Tutte le serie devono essere allineate a `INDEX` ed essere timezone-aware **UTC**.
- La risoluzione prevista è 15 minuti; la pagina è robusta a indici equispaziati diversi, ma KPI e
  plotting sono tarati su 15 min / 1 h.
- Potenza in kW, temperatura in °C.

"""
__doc__ = MODULE_DOC

import streamlit as st
import pandas as pd
from pathlib import Path

from cer_app.session_paths import get_paths

# Panels (schede)
from cer_app.schede_consumatori import lavanderia, cucina, carichi_base, occupancy, clima
from cer_app.schede_consumatori.shared import (

    load_session_seed,
    load_time_index_and_meteo,
    load_consumers_json,
    save_consumers_json,
    next_consumer_id,
    consumer_cache_dir,
    derive_seed,
    ensure_device,
    dict_to_matrix,
    matrix_to_dict,
    count_weekly_slots_from_grid,
    start_matrix_editor,
    SLOT_LABELS,
    DAY_NAMES,
    prepare_curve_for_plot,
    sanitize_for_dataclass,
)


st.set_page_config(page_title="Consumatori", layout="wide")
st.title("Consumatori")

# -------------------- session paths --------------------
paths = get_paths()
SESSION_DIR: Path = paths.session_dir

# -------------------- time index + meteo --------------------
INDEX, T_AIR_DAILY, T_AIR_HOURLY, T_AIR_15 = load_time_index_and_meteo(SESSION_DIR)
BASE_SEED = load_session_seed(SESSION_DIR, default=2025)

# Clima richiede sempre una pd.Series per la temperatura esterna.
# Se mancante (es. sessione senza meteo), inizializziamo una serie costante.
if len(INDEX) > 0:
    _TEMP_FALLBACK_USED = False
    if T_AIR_15 is None:
        _TEMP_FALLBACK_USED = True
        T_AIR_15 = pd.Series(15.0, index=INDEX, name="t_air")
    elif not isinstance(T_AIR_15, pd.Series):
        # forza la temperatura in forma pd.Series allineata all'indice
        try:
            T_AIR_15 = pd.Series(list(T_AIR_15), index=INDEX, name="t_air").astype(float)
        except Exception:
            _TEMP_FALLBACK_USED = True
            T_AIR_15 = pd.Series(15.0, index=INDEX, name="t_air")

if len(INDEX) == 0:
    st.error("Indice temporale non disponibile. Verifica che i file meteo/sessione siano presenti.")
    st.stop()

# -------------------- load/save consumers --------------------
consumers = load_consumers_json(SESSION_DIR)

# -------------------- consumer list + CRUD (layout tipo "Produttori") --------------------
# Obiettivo: evitare che la gestione sia "nascosta" in sidebar e mostrare una lista centrale.

left, right = st.columns([3, 1], gap="large")

with right:
    st.markdown("### Gestione consumatori")

    if st.button("➕ Aggiungi nuovo", use_container_width=True):
        new_id = next_consumer_id(consumers)
        consumers.append({"id": new_id, "name": f"Consumatore {new_id}", "note": "", "people": 1})
        save_consumers_json(SESSION_DIR, consumers)
        st.session_state["_selected_consumer_id"] = int(new_id)
        st.rerun()

if not consumers:
    with left:
        st.info("Nessun consumatore presente. Usa **Aggiungi nuovo** per crearne uno.")
    st.stop()

# selezione consumatore attivo (persistente)
consumer_ids = [int(c.get("id")) for c in consumers]
default_id = int(st.session_state.get("_selected_consumer_id", consumer_ids[0]))
if default_id not in consumer_ids:
    default_id = consumer_ids[0]

with right:
    selected_id = st.selectbox(
        "Modifica ID",
        options=consumer_ids,
        index=consumer_ids.index(default_id),
        key="_selected_consumer_id",
    )

consumer = next(c for c in consumers if int(c.get("id")) == int(selected_id))
consumer_id = int(consumer.get("id"))

with left:
    st.markdown("### Elenco consumatori")
    df_consumers = pd.DataFrame(consumers)
    # pulizia colonne per la vista tabellare
    for col in ["id", "name", "note"]:
        if col not in df_consumers.columns:
            df_consumers[col] = ""
    df_view = df_consumers[["id", "name", "note"]].rename(
        columns={"id": "ID", "name": "Nome", "note": "Note"}
    )
    st.dataframe(df_view, hide_index=True, use_container_width=True)

with right:
    st.text_input("Nome", value=str(consumer.get("name", "")), key=f"c{consumer_id}_name")
    st.text_area("Note", value=str(consumer.get("note", "")), key=f"c{consumer_id}_note")
    st.number_input("Numero residenti",min_value=0,max_value=10,value=int(consumer.get("people") or 1),step=1,key=f"c{consumer_id}_people",)

    if st.button("✏️ Salva modifiche", use_container_width=True):
        consumer["name"] = st.session_state.get(f"c{consumer_id}_name", consumer.get("name", ""))
        consumer["note"] = st.session_state.get(f"c{consumer_id}_note", consumer.get("note", ""))
        consumer["people"] = int(st.session_state.get(f"c{consumer_id}_people", consumer.get("people", 1)))
        save_consumers_json(SESSION_DIR, consumers)
        st.success("Modifica salvata.")
        st.rerun()

    # eliminazione (lista separata come in Produttori)
    st.markdown("---")
    del_id = st.selectbox("Elimina ID", options=consumer_ids, index=consumer_ids.index(consumer_id), key="_del_id")
    if st.button("🗑️ Elimina", use_container_width=True):
        consumers = [c for c in consumers if int(c.get("id")) != int(del_id)]
        save_consumers_json(SESSION_DIR, consumers)
        # riallinea selezione
        remaining_ids = [int(c.get("id")) for c in consumers]
        if remaining_ids:
            st.session_state["_selected_consumer_id"] = remaining_ids[0]
        st.warning("Consumatore eliminato.")
        st.rerun()

# -------------------- view controls --------------------
st.subheader("Visualizzazione curve")
cA, cB = st.columns([1, 1])
_view_label = cA.radio(
    "Modalità grafico",
    options=["Annuale (1h)", "Mensile (15min)"],
    horizontal=True,
)
# Le schede usano i valori storici "annuale" / "mensile".
curve_view_mode = "annuale" if _view_label.startswith("Annuale") else "mensile"

_month_int = cB.selectbox(
    "Mese (solo se Mensile)",
    options=list(range(1, 13)),
    index=0,
)
# (anno, mese) come nella pagina originale
curve_view_month = (int(INDEX[0].year), int(_month_int)) if curve_view_mode == "mensile" else None

# -------------------- inject shared globals into panels --------------------
def _bind_panel_globals(mod):
    """Bind delle dipendenze runtime per le schede UI.

    Le schede in `cer_app.schede_consumatori` sono moduli Streamlit che, per vincoli storici,
    accedono a variabili globali (es. `SESSION_DIR`, `INDEX`, `consumer`, ecc.) invece di riceverle
    sempre come argomenti. Questa funzione "inietta" tali dipendenze nel namespace del modulo
    mantenendo invariata l'architettura e i path del progetto.

    Parameters
    ----------
    mod:
        Modulo Python di una scheda (es. `cer_app.schede_consumatori.cucina`).
    """
    mod.SESSION_DIR = SESSION_DIR
    mod.INDEX = INDEX
    mod.BASE_SEED = BASE_SEED
    mod.T_AIR_15 = T_AIR_15
    mod.save_consumers_json = save_consumers_json
    mod.consumers = consumers
    mod.consumer = consumer
    mod.curve_view_mode = curve_view_mode
    mod.curve_view_month = curve_view_month
    mod.ensure_device = ensure_device
    mod.derive_seed = derive_seed
    mod.start_matrix_editor = start_matrix_editor
    # grid helpers used by Lavanderia/Clima panels
    mod.dict_to_matrix = dict_to_matrix
    mod.matrix_to_dict = matrix_to_dict
    mod.count_weekly_slots_from_grid = count_weekly_slots_from_grid
    # labels used by Cucina (and potentially other panels)
    mod.SLOT_LABELS = SLOT_LABELS
    mod.DAY_NAMES = DAY_NAMES
    mod.prepare_curve_for_plot = prepare_curve_for_plot
    mod.sanitize_for_dataclass = sanitize_for_dataclass
    # KPI + rescaling helpers (needed by multiple panels)
    if hasattr(clima, "compute_kpis"):
        mod.compute_kpis = clima.compute_kpis
    if hasattr(clima, "rescale_curve_to_annual_target"):
        mod.rescale_curve_to_annual_target = clima.rescale_curve_to_annual_target
    if hasattr(clima, "_load_cached_curve"):
        mod._load_cached_curve = clima._load_cached_curve

for m in [lavanderia, cucina, carichi_base, occupancy, clima]:
    _bind_panel_globals(m)

# -------------------- panels --------------------
st.markdown("---")
st.subheader("Schede consumatore")

tabs = st.tabs(["Lavanderia", "Cucina", "Carichi base", "Occupancy", "Clima"])
with tabs[0]:
    lavanderia.laundry_panel()
with tabs[1]:
    cucina.kitchen_panel()
with tabs[2]:
    carichi_base.baseload_panel()
with tabs[3]:
    occupancy.occupancy_panel()
with tabs[4]:
    # clima.py espone anche get_climate_device + build_climate_config...
    # usa la funzione panel se presente; in caso contrario non fare nulla
    if hasattr(clima, "climate_panel"):
        clima.climate_panel(consumer, INDEX, T_AIR_15)
    else:
        st.info("Scheda clima non disponibile in questo build.")

# -------------------- aggregate from cache (Option 2) --------------------
st.markdown("---")
st.subheader("Curva aggregata (da cache)")

cache_dir = consumer_cache_dir(SESSION_DIR, consumer_id)

component_files = [
    ("Lavanderia", cache_dir / "laundry_total.csv"),
    ("Cucina", cache_dir / "kitchen.csv"),
    ("Carichi base", cache_dir / "baseload.csv"),
    ("Occupancy", cache_dir / "occupancy_15min.csv"),
    ("Clima", cache_dir / "climate.csv"),
]

present = [(name, fp) for name, fp in component_files if fp.exists()]

if not present:
    st.info(
        "Nessun CSV componente trovato in cache. "
        "Calcola almeno una scheda (pulsante di calcolo nella scheda), "
        "poi torna qui per costruire l'aggregato."
    )
else:
    st.write("Componenti disponibili in cache:")
    st.write(", ".join([n for n, _ in present]))

    build = st.button("🧮 Costruisci/aggiorna aggregato da cache", key=f"agg_from_cache_{consumer_id}")
    total_fp = cache_dir / "total_load_15min.csv"

    def _load_series(fp: Path) -> pd.Series:
        """Carica una componente di carico dalla cache e la riallinea a INDEX.

        Il CSV è atteso nel formato: colonna `timestamp` + una colonna numerica (potenza in kW).
        La serie viene forzata a indice timezone-aware UTC e reindicizzata su `INDEX` con riempimento a 0.
        """
        s = pd.read_csv(fp, parse_dates=["timestamp"]).set_index("timestamp").iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True)
        s = s.reindex(INDEX).fillna(0.0).astype(float)
        return s

    if build:
        curves = []
        used = []
        for name, fp in present:
            try:
                curves.append(_load_series(fp))
                used.append(name)
            except Exception:
                continue

        if not curves:
            st.error("Impossibile leggere i CSV in cache.")
        else:
            total = sum(curves)
            total.rename("total_load").to_csv(total_fp, index=True, index_label="timestamp")
            st.success(f"Aggregato salvato in cache: total_load_15min.csv (componenti: {', '.join(used)})")

    # show current aggregate if available
    if total_fp.exists():
        total = _load_series(total_fp).rename("total_load")
        st.markdown("#### Curva aggregata")
        st.line_chart(clima.prepare_curve_for_plot(total, curve_view_mode, curve_view_month))

        # temperatura esterna usata per il calcolo (serve a diagnosticare fallback a 15°C)
        try:
            t_air = T_AIR_15.copy()
        except Exception:
            t_air = pd.Series(15.0, index=INDEX, name="t_air")
            _TEMP_FALLBACK_USED = True

        t_air = pd.Series(t_air, index=INDEX, name="t_air").astype(float).rename("temp_outdoor_C")

        if bool(locals().get("_TEMP_FALLBACK_USED", False)):
            st.warning("Meteo non disponibile: temperatura esterna impostata a **15°C costanti** (fallback).")


        kwh_periodo, kwh_giorno, kwh_annuo = clima.compute_kpis(total, INDEX)
        m1, m2, m3 = st.columns(3)
        m1.metric("kWh periodo", f"{kwh_periodo:.1f}")
        m2.metric("kWh/anno stimato", f"{kwh_annuo:.0f}")
        m3.metric("Media giornaliera", f"{kwh_giorno:.2f} kWh/g")

        csv_buf = total.to_csv(index=True, index_label="timestamp").encode("utf-8")
        st.download_button(
            "⬇️ CSV 15 min (aggregato – cache)",
            data=csv_buf,
            file_name=f"consumer_{consumer_id}_total_load_15min.csv",
            mime="text/csv",
            key=f"dl_{consumer_id}_total_cache",
        )
    else:
        st.info("Aggregato non presente. Premi “🧮 Costruisci/aggiorna aggregato da cache”.")

# -------------------- rescale aggregate (uses cached total) --------------------
st.markdown("---")
st.subheader("Riscala curva aggregata su consumo annuo target")

# La riscalatura è un post-processing dell'aggregato: si applica un fattore moltiplicativo
# tale che l'energia del periodo simulato (kWh) sia coerente con un target annuo (kWh/anno).
# Se l'orizzonte simulato non copre un anno completo, il target viene riproporzionato sui
# giorni simulati (vedi `clima.rescale_curve_to_annual_target`).

raw_curve_cached = clima._load_cached_curve(SESSION_DIR, consumer_id, "total_load_15min.csv", INDEX)
if raw_curve_cached is None:
    st.info("Prima costruisci la curva aggregata da cache (sezione sopra).")
else:
    raw_curve_cached = raw_curve_cached.rename("total_load").astype(float)

    default_target = float(consumer.get("annual_consumption_kwh") or 0.0)
    cA, cB, cC = st.columns([3, 1, 1])

    target_kwh = cA.number_input(
        "Target consumo annuo (kWh/anno)",
        min_value=0.0,
        value=default_target,
        step=100.0,
        key=f"c{consumer_id}_rescale_target_kwh",
        help="Il target è annuale. Se il periodo simulato non è un anno intero, viene riproporzionato sul periodo simulato.",
    )

    save_target = cB.button("💾 Salva target", key=f"c{consumer_id}_rescale_save_target")
    do_rescale = cC.button("📏 Riscala curva", key=f"c{consumer_id}_rescale_btn")

    if save_target:
        if target_kwh > 0:
            consumer["annual_consumption_kwh"] = float(target_kwh)
        else:
            consumer.pop("annual_consumption_kwh", None)
        save_consumers_json(SESSION_DIR, consumers)
        st.success("Target annuo salvato nel consumatore.")

    scaled_curve = None
    meta = None
    if do_rescale:
        scaled_curve, meta = clima.rescale_curve_to_annual_target(raw_curve_cached, INDEX, target_kwh)
        if scaled_curve is None:
            st.warning("Impossibile riscalare: inserisci un target > 0 e verifica che la curva non sia nulla.")
        else:
            scaled_curve.rename("total_load").to_csv(
                cache_dir / "total_load_15min_scaled.csv",
                index=True,
                index_label="timestamp",
            )
            st.success(f"Curva riscalata. Fattore scala: {meta['scale_factor']:.4f}")

    if scaled_curve is None:
        scaled_curve = clima._load_cached_curve(SESSION_DIR, consumer_id, "total_load_15min_scaled.csv", INDEX)
        if scaled_curve is not None:
            scaled_curve = scaled_curve.rename("total_load").astype(float)

    if scaled_curve is None:
        st.info("Nessuna curva riscalata disponibile. Premi “📏 Riscala curva”.")
    else:
        st.markdown("#### Curva aggregata riscalata")
        st.line_chart(clima.prepare_curve_for_plot(scaled_curve, curve_view_mode, curve_view_month))

        kwh_periodo_s, kwh_giorno_s, kwh_annuo_s = clima.compute_kpis(scaled_curve, INDEX)
        m1, m2, m3 = st.columns(3)
        m1.metric("Riscalata: kWh periodo", f"{kwh_periodo_s:.1f}")
        m2.metric("Riscalata: kWh/anno stimato", f"{kwh_annuo_s:.0f}")
        m3.metric("Riscalata: media giornaliera", f"{kwh_giorno_s:.2f} kWh/g")

        csv_buf_scaled = scaled_curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
        st.download_button(
            "⬇️ CSV 15 min (aggregato riscalato)",
            data=csv_buf_scaled,
            file_name=f"consumer_{consumer_id}_total_load_15min_scaled.csv",
            mime="text/csv",
            key=f"dl_{consumer_id}_total_scaled",
        )
