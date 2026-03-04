from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
import json
import altair as alt
from typing import List, Dict, Tuple, Optional
from dataclasses import fields as dc_fields
from io import StringIO

# Core modules
from cer_core.consumatori import lavanderia as L
from cer_core.consumatori import cucina as K
from cer_core.consumatori import baseload as B
from cer_core.consumatori import clima as CL
from cer_core.consumatori import occupancy as O


# --- Cucina ---
# defaults per blocco cucina
KITCHEN_DEF = {
    "present": False,
    "habits": {},
    "oven": {},
    "induction": {},
    "microwave": {},
    "dishwasher": {},
    "hood": {},
}
def kitchen_panel():
    """Pannello Streamlit per la cucina: abitudini + elettrodomestici cucina."""
    consumer_id = consumer.get("id"); dev_k = ensure_device(consumer, "kitchen", KITCHEN_DEF)
    with st.expander("🍳 Cucina (forno, induzione, microonde, lavastoviglie, cappa)", expanded=False):
        # flag presenza cucina
        present = st.checkbox(
            "Attiva modellazione cucina per questo consumatore",
            value=bool(dev_k.get("present", False)),
            key=f"c{consumer_id}_kitchen_present",
        )
        dev_k["present"] = bool(present)
        if not dev_k["present"]:
            if st.button("Abilita cucina", key=f"c{consumer_id}_add_kitchen"):
                dev_k.update(KITCHEN_DEF)
                dev_k["present"] = True
                save_consumers_json(SESSION_DIR, consumers)
                st.success("Cucina abilitata per il consumatore.")
            return None

        # ----- abitudini di pasto -----
        st.markdown("### Abitudini di pasto a casa")
        hab = dev_k.setdefault("habits", {})
        col_freq1, col_freq2 = st.columns(2)
        hab["weekday_lunch_at_home"] = col_freq1.select_slider(
            "Pranzi feriali a casa",
            options=[0,1,2,3],
            value=int(hab.get("weekday_lunch_at_home", 2)),
            format_func=lambda v: {
                0: "Quasi mai",
                1: "1–2 giorni/sett",
                2: "3–4 giorni/sett",
                3: "Quasi sempre (5/5)",
            }[v],
            key=f"c{consumer_id}_kitchen_weekday_lunch_at_home",
        )
        hab["weekday_dinner_at_home"] = col_freq2.select_slider(
            "Cene feriali a casa",
            options=[0,1,2,3],
            value=int(hab.get("weekday_dinner_at_home", 3)),
            format_func=lambda v: {
                0: "Quasi mai",
                1: "1–2 giorni/sett",
                2: "3–4 giorni/sett",
                3: "Quasi sempre (5/5)",
            }[v],
            key=f"c{consumer_id}_kitchen_weekday_dinner_at_home",
        )
        col_freq3, col_freq4 = st.columns(2)
        hab["weekend_lunch_at_home"] = col_freq3.select_slider(
            "Pranzi weekend a casa",
            options=[0,1,2],
            value=int(hab.get("weekend_lunch_at_home", 2)),
            format_func=lambda v: {
                0: "Quasi mai",
                1: "1 giorno/2",
                2: "Quasi sempre (2/2)",
            }[v],
            key=f"c{consumer_id}_kitchen_weekend_lunch_at_home",
        )
        hab["weekend_dinner_at_home"] = col_freq4.select_slider(
            "Cene weekend a casa",
            options=[0,1,2],
            value=int(hab.get("weekend_dinner_at_home", 2)),
            format_func=lambda v: {
                0: "Quasi mai",
                1: "1 giorno/2",
                2: "Quasi sempre (2/2)",
            }[v],
            key=f"c{consumer_id}_kitchen_weekend_dinner_at_home",
        )

        # orari tipici dei pasti -> slot
        st.markdown("#### Orari tipici dei pasti (slot giornalieri)")
        col_orari1, col_orari2 = st.columns(2)
        n_slots = len(SLOT_LABELS)
        lunch_slot_weekday = int(hab.get("lunch_slot_weekday", 3))
        if not 0 <= lunch_slot_weekday < n_slots:
            lunch_slot_weekday = 3
        hab["lunch_slot_weekday"] = col_orari1.selectbox(
            "Orario pranzo feriale",
            options=list(range(n_slots)),
            index=lunch_slot_weekday,
            format_func=lambda i: SLOT_LABELS[i],
            key=f"c{consumer_id}_kitchen_lunch_slot_weekday",
        )
        dinner_slot_weekday = int(hab.get("dinner_slot_weekday", 6))
        if not 0 <= dinner_slot_weekday < n_slots:
            dinner_slot_weekday = 6
        hab["dinner_slot_weekday"] = col_orari2.selectbox(
            "Orario cena feriale",
            options=list(range(n_slots)),
            index=dinner_slot_weekday,
            format_func=lambda i: SLOT_LABELS[i],
            key=f"c{consumer_id}_kitchen_dinner_slot_weekday",
        )

        col_orari3, col_orari4 = st.columns(2)
        lunch_slot_weekend = int(hab.get("lunch_slot_weekend", 3))
        if not 0 <= lunch_slot_weekend < n_slots:
            lunch_slot_weekend = 3
        hab["lunch_slot_weekend"] = col_orari3.selectbox(
            "Orario pranzo weekend",
            options=list(range(n_slots)),
            index=lunch_slot_weekend,
            format_func=lambda i: SLOT_LABELS[i],
            key=f"c{consumer_id}_kitchen_lunch_slot_weekend",
        )
        dinner_slot_weekend = int(hab.get("dinner_slot_weekend", 6))
        if not 0 <= dinner_slot_weekend < n_slots:
            dinner_slot_weekend = 6
        hab["dinner_slot_weekend"] = col_orari4.selectbox(
            "Orario cena weekend",
            options=list(range(n_slots)),
            index=dinner_slot_weekend,
            format_func=lambda i: SLOT_LABELS[i],
            key=f"c{consumer_id}_kitchen_dinner_slot_weekend",
        )

        st.markdown("#### Colazione (opzionale)")
        hab["enable_breakfast"] = st.checkbox(
            "Considera anche la colazione",
            value=bool(hab.get("enable_breakfast", False)),
            key=f"c{consumer_id}_kitchen_enable_breakfast",
        )
        if hab["enable_breakfast"]:
            col_b1, col_b2 = st.columns(2)
            hab["breakfast_at_home"] = col_b1.select_slider(
                "Colazioni a casa",
                options=[0,1,2],
                value=int(hab.get("breakfast_at_home", 1)),
                format_func=lambda v: {
                    0: "Quasi mai",
                    1: "Spesso (4/5 feriali)",
                    2: "Quasi sempre",
                }[v],
                key=f"c{consumer_id}_kitchen_breakfast_at_home",
            )
            breakfast_slot = int(hab.get("breakfast_slot", 1))
            if not 0 <= breakfast_slot < n_slots:
                breakfast_slot = 1
            hab["breakfast_slot"] = col_b2.selectbox(
                "Orario colazione",
                options=list(range(n_slots)),
                index=breakfast_slot,
                format_func=lambda i: SLOT_LABELS[i],
                key=f"c{consumer_id}_kitchen_breakfast_slot",
            )

        # ----- possesso apparecchi cucina -----
        st.markdown("### Apparecchi cucina presenti")
        col_app1, col_app2, col_app3, col_app4, col_app5 = st.columns(5)
        oven = dev_k.setdefault("oven", {})
        induction = dev_k.setdefault("induction", {})
        microwave = dev_k.setdefault("microwave", {})
        dishwasher = dev_k.setdefault("dishwasher", {})
        hood = dev_k.setdefault("hood", {})

        oven["has_oven"] = col_app1.checkbox(
            "Forno",
            value=bool(oven.get("has_oven", False)),
            key=f"c{consumer_id}_kitchen_has_oven",
        )
        induction["has_induction"] = col_app2.checkbox(
            "Piano a induzione",
            value=bool(induction.get("has_induction", False)),
            key=f"c{consumer_id}_kitchen_has_induction",
        )
        microwave["has_microwave"] = col_app3.checkbox(
            "Microonde",
            value=bool(microwave.get("has_microwave", False)),
            key=f"c{consumer_id}_kitchen_has_microwave",
        )
        dishwasher["has_dishwasher"] = col_app4.checkbox(
            "Lavastoviglie",
            value=bool(dishwasher.get("has_dishwasher", False)),
            key=f"c{consumer_id}_kitchen_has_dishwasher",
        )
        hood["has_hood"] = col_app5.checkbox(
            "Cappa cucina",
            value=bool(hood.get("has_hood", False)),
            key=f"c{consumer_id}_kitchen_has_hood",
        )

        # ----- dettagli forno -----
        if oven.get("has_oven", False):
            st.markdown("#### Forno")
            col_o1, col_o2 = st.columns(2)
            oven["oven_weekly_intensity"] = col_o1.slider(
                "Quanto spesso usi il forno in una settimana tipica?",
                min_value=0, max_value=5,
                value=int(oven.get("oven_weekly_intensity", 2)),
                help="0=quasi mai, 5=quasi ogni giorno",
                key=f"c{consumer_id}_kitchen_oven_weekly_intensity",
            )
            OVEN_DAY_OPTIONS = [
                ("solo_weekend", "Solo weekend"),
                ("soprattutto_weekend", "Soprattutto weekend"),
                ("equilibrato", "Equilibrato feriali/weekend"),
                ("anche_feriali_spesso", "Anche feriali molto spesso"),
            ]
            current_od = oven.get("oven_days_preference", "soprattutto_weekend")
            if current_od not in [o[0] for o in OVEN_DAY_OPTIONS]:
                current_od = "soprattutto_weekend"
            oven["oven_days_preference"] = col_o2.selectbox(
                "Per quali giorni usi di più il forno?",
                options=[opt[0] for opt in OVEN_DAY_OPTIONS],
                index=[o[0] for o in OVEN_DAY_OPTIONS].index(current_od),
                format_func=lambda k: dict(OVEN_DAY_OPTIONS)[k],
                key=f"c{consumer_id}_kitchen_oven_days_pref",
            )

            OVEN_MEAL_OPTIONS = [
                ("solo_cena", "Quasi solo a cena"),
                ("prevalentemente_cena", "Prevalentemente a cena"),
                ("pranzo_e_cena", "Pranzo e cena"),
                ("soprattutto_pranzo_festivi", "Soprattutto pranzo nei festivi"),
            ]
            current_om = oven.get("oven_meal_preference", "prevalentemente_cena")
            if current_om not in [o[0] for o in OVEN_MEAL_OPTIONS]:
                current_om = "prevalentemente_cena"
            col_o3, col_o4 = st.columns(2)
            oven["oven_meal_preference"] = col_o3.selectbox(
                "In quali pasti usi il forno?",
                options=[opt[0] for opt in OVEN_MEAL_OPTIONS],
                index=[o[0] for o in OVEN_MEAL_OPTIONS].index(current_om),
                format_func=lambda k: dict(OVEN_MEAL_OPTIONS)[k],
                key=f"c{consumer_id}_kitchen_oven_meal_pref",
            )
            OVEN_COMPLEXITY = [
                ("veloce", "Per piatti veloci / riscaldare"),
                ("misto", "Uso misto"),
                ("elaborato", "Piatti elaborati (arrosti, lasagne, ecc.)"),
            ]
            current_oc = oven.get("oven_complexity", "misto")
            if current_oc not in [o[0] for o in OVEN_COMPLEXITY]:
                current_oc = "misto"
            oven["oven_complexity"] = col_o4.selectbox(
                "Tipologia di cottura tipica",
                options=[o[0] for o in OVEN_COMPLEXITY],
                index=[o[0] for o in OVEN_COMPLEXITY].index(current_oc),
                format_func=lambda k: dict(OVEN_COMPLEXITY)[k],
                key=f"c{consumer_id}_kitchen_oven_complexity",
            )

            # Potenza forno configurabile (default 2.0 kW)
            oven["oven_power_kw"] = st.number_input(
                "Potenza nominale forno (kW)",
                min_value=0.1,
                max_value=5.0,
                value=float(oven.get("oven_power_kw", 2.0)),
                step=0.1,
                key=f"c{consumer_id}_kitchen_oven_power_kw",
            )

        # ----- dettagli piano a induzione -----
        if induction.get("has_induction", False):
            st.markdown("#### Piano a induzione")
            col_i1, col_i2 = st.columns(2)
            induction["induction_is_primary"] = col_i1.checkbox(
                "Il piano a induzione è il piano principale",
                value=bool(induction.get("induction_is_primary", True)),
                key=f"c{consumer_id}_kitchen_induction_primary",
            )
            RATIO_OPTS = [
                ("quasi_sempre", "Quasi sempre"),
                ("spesso", "Spesso"),
                ("meta", "Circa metà delle volte"),
                ("raramente", "Raramente"),
            ]
            current_ir = induction.get("induction_use_ratio", "spesso")
            if current_ir not in [o[0] for o in RATIO_OPTS]:
                current_ir = "spesso"
            induction["induction_use_ratio"] = col_i2.selectbox(
                "Quanto spesso usi il piano a induzione quando cucini?",
                options=[o[0] for o in RATIO_OPTS],
                index=[o[0] for o in RATIO_OPTS].index(current_ir),
                format_func=lambda k: dict(RATIO_OPTS)[k],
                key=f"c{consumer_id}_kitchen_induction_use_ratio",
            )

            MEAL_PREF_OPTS = [
                ("colazione_pranzo", "Colazione e pranzo"),
                ("pranzo_cena", "Pranzo e cena"),
                ("solo_cena", "Quasi solo cena"),
                ("tutti_i_pasti_caldi", "Tutti i pasti caldi"),
            ]
            current_imp = induction.get("induction_meal_preference", "pranzo_cena")
            if current_imp not in [o[0] for o in MEAL_PREF_OPTS]:
                current_imp = "pranzo_cena"
            col_i3, col_i4 = st.columns(2)
            induction["induction_meal_preference"] = col_i3.selectbox(
                "Per quali pasti usi più spesso il piano a induzione?",
                options=[o[0] for o in MEAL_PREF_OPTS],
                index=[o[0] for o in MEAL_PREF_OPTS].index(current_imp),
                format_func=lambda k: dict(MEAL_PREF_OPTS)[k],
                key=f"c{consumer_id}_kitchen_induction_meal_pref",
            )
            INTENSITY_OPTS = [
                ("semplice", "Cucina semplice"),
                ("misto", "Uso misto"),
                ("elaborata", "Cucina elaborata"),
            ]
            current_ci = induction.get("cooking_intensity", "misto")
            if current_ci not in [o[0] for o in INTENSITY_OPTS]:
                current_ci = "misto"
            induction["cooking_intensity"] = col_i4.selectbox(
                "Quanto è intensa la cucina in casa?",
                options=[o[0] for o in INTENSITY_OPTS],
                index=[o[0] for o in INTENSITY_OPTS].index(current_ci),
                format_func=lambda k: dict(INTENSITY_OPTS)[k],
                key=f"c{consumer_id}_kitchen_cooking_intensity",
            )
            induction["typical_burners_in_use"] = st.slider(
                "Numero tipico di fuochi usati contemporaneamente",
                min_value=1, max_value=3,
                value=int(induction.get("typical_burners_in_use", 2)),
                key=f"c{consumer_id}_kitchen_induction_burners",
            )

        # ----- dettagli microonde -----
        if microwave.get("has_microwave", False):
            st.markdown("#### Microonde")
            MAIN_USE_OPTS = [
                ("riscaldare_pranzo_lavoro", "Riscaldare pranzo/lavoro"),
                ("scongelare", "Scongelare"),
                ("cucinare_piatti_pronti", "Cucinare piatti pronti"),
                ("vario", "Un po' di tutto"),
            ]
            current_mu = microwave.get("microwave_main_use", "vario")
            if current_mu not in [o[0] for o in MAIN_USE_OPTS]:
                current_mu = "vario"
            col_m1, col_m2 = st.columns(2)
            microwave["microwave_main_use"] = col_m1.selectbox(
                "Uso principale del microonde",
                options=[o[0] for o in MAIN_USE_OPTS],
                index=[o[0] for o in MAIN_USE_OPTS].index(current_mu),
                format_func=lambda k: dict(MAIN_USE_OPTS)[k],
                key=f"c{consumer_id}_kitchen_micro_main_use",
            )
            TIME_PREF_OPTS = [
                ("solo_pranzo_feriale", "Quasi solo pranzo feriale"),
                ("spesso_cena", "Spesso a cena"),
                ("anche_colazioni_snack", "Anche colazioni/snack"),
                ("distribuito", "Distribuito nella giornata"),
            ]
            current_tp = microwave.get("microwave_time_preference", "distribuito")
            if current_tp not in [o[0] for o in TIME_PREF_OPTS]:
                current_tp = "distribuito"
            microwave["microwave_time_preference"] = col_m2.selectbox(
                "In quali momenti della giornata lo usi di più?",
                options=[o[0] for o in TIME_PREF_OPTS],
                index=[o[0] for o in TIME_PREF_OPTS].index(current_tp),
                format_func=lambda k: dict(TIME_PREF_OPTS)[k],
                key=f"c{consumer_id}_kitchen_micro_time_pref",
            )

            col_m3, col_m4 = st.columns(2)
            microwave["microwave_weekly_intensity"] = col_m3.slider(
                "Quanto spesso usi il microonde in una settimana?",
                min_value=0, max_value=5,
                value=int(microwave.get("microwave_weekly_intensity", 2)),
                key=f"c{consumer_id}_kitchen_micro_intensity",
            )
            DUR_OPTS = [
                ("1-3", "Durata tipica 1–3 min"),
                ("3-5", "Durata tipica 3–5 min"),
                ("5-10", "Durata tipica 5–10 min"),
            ]
            current_md = microwave.get("microwave_session_duration", "3-5")
            if current_md not in [o[0] for o in DUR_OPTS]:
                current_md = "3-5"
            microwave["microwave_session_duration"] = col_m4.selectbox(
                "Durata tipica di un uso",
                options=[o[0] for o in DUR_OPTS],
                index=[o[0] for o in DUR_OPTS].index(current_md),
                format_func=lambda k: dict(DUR_OPTS)[k],
                key=f"c{consumer_id}_kitchen_micro_duration",
            )

            SUBSTITUTE_OPTS = [
                ("mai", "Mai"),
                ("a_volte", "A volte"),
                ("spesso", "Spesso"),
            ]
            current_ms = microwave.get("microwave_as_oven_substitute", "mai")
            if current_ms not in [o[0] for o in SUBSTITUTE_OPTS]:
                current_ms = "mai"
            microwave["microwave_as_oven_substitute"] = st.selectbox(
                "Lo usi come sostituto del forno?",
                options=[o[0] for o in SUBSTITUTE_OPTS],
                index=[o[0] for o in SUBSTITUTE_OPTS].index(current_ms),
                format_func=lambda k: dict(SUBSTITUTE_OPTS)[k],
                key=f"c{consumer_id}_kitchen_micro_substitute",
            )

            # Potenza microonde configurabile (default 0.9 kW)
            microwave["microwave_power_kw"] = st.number_input(
                "Potenza nominale microonde (kW)",
                min_value=0.3,
                max_value=2.5,
                value=float(microwave.get("microwave_power_kw", 0.9)),
                step=0.1,
                key=f"c{consumer_id}_kitchen_micro_power_kw",
            )



        # ----- dettagli lavastoviglie -----
        if dishwasher.get("has_dishwasher", False):
            st.markdown("#### Lavastoviglie")
            col_dw1, col_dw2, col_dw3 = st.columns(3)
            dishwasher["cycles_per_week"] = col_dw1.slider(
                "Cicli a settimana",
                min_value=0, max_value=14,
                value=int(dishwasher.get("cycles_per_week", 4)),
                help="Numero medio di cicli settimanali (0 = mai).",
                key=f"c{consumer_id}_dishwasher_cycles_per_week",
            )
            DW_PROG_OPTS = [
                ("eco", "Eco (lungo, più efficiente)"),
                ("standard", "Standard"),
                ("quick", "Rapido (corto)"),
            ]
            current_prog = dishwasher.get("program", "eco")
            prog_ids = [k for k, _ in DW_PROG_OPTS]
            if current_prog not in prog_ids:
                current_prog = "eco"
            dishwasher["program"] = col_dw2.selectbox(
                "Programma",
                options=prog_ids,
                index=prog_ids.index(current_prog),
                format_func=lambda k: dict(DW_PROG_OPTS).get(k, k),
                key=f"c{consumer_id}_dishwasher_program",
            )
            EC_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
            current_ec = str(dishwasher.get("energy_class", "C")).upper()
            if current_ec not in EC_OPTIONS:
                current_ec = "C"
            dishwasher["energy_class"] = col_dw3.selectbox(
                "Classe energetica",
                options=EC_OPTIONS,
                index=EC_OPTIONS.index(current_ec),
                key=f"c{consumer_id}_dishwasher_energy_class",
            )

            MODE_OPTS = [
                ("after_meal", "Dopo i pasti (pranzo/cena)"),
                ("scheduled_fixed_time", "Orario fisso"),
            ]
            mode_ids = [k for k, _ in MODE_OPTS]
            current_mode = dishwasher.get("mode", "after_meal")
            if current_mode not in mode_ids:
                current_mode = "after_meal"
            dishwasher["mode"] = st.radio(
                "Modalità di avvio",
                options=mode_ids,
                index=mode_ids.index(current_mode),
                format_func=lambda k: dict(MODE_OPTS).get(k, k),
                horizontal=True,
                key=f"c{consumer_id}_dishwasher_mode",
            )
            if dishwasher["mode"] == "scheduled_fixed_time":
                dishwasher["fixed_start_time"] = st.text_input(
                    "Orario fisso (HH:MM, Europe/Rome)",
                    value=str(dishwasher.get("fixed_start_time", "22:30")),
                    help="Esempio: 22:30",
                    key=f"c{consumer_id}_dishwasher_fixed_time",
                )
            else:
                dishwasher.setdefault("fixed_start_time", "22:30")


        # ----- dettagli cappa -----
        if hood.get("has_hood", False):
            st.markdown("#### Cappa cucina")
            col_h1, col_h2 = st.columns(2)
            HOOD_USE_OPTS = [
                ("quasi_mai", "Quasi mai"),
                ("piatti_importanti", "Solo per piatti importanti/fritti"),
                ("spesso", "Spesso"),
                ("quasi_sempre", "Quasi sempre quando cucino"),
            ]
            current_hu = hood.get("hood_use_habit", "spesso")
            if current_hu not in [o[0] for o in HOOD_USE_OPTS]:
                current_hu = "spesso"
            hood["hood_use_habit"] = col_h1.selectbox(
                "Quanto spesso usi la cappa?",
                options=[o[0] for o in HOOD_USE_OPTS],
                index=[o[0] for o in HOOD_USE_OPTS].index(current_hu),
                format_func=lambda k: dict(HOOD_USE_OPTS)[k],
                key=f"c{consumer_id}_kitchen_hood_use_habit",
            )
            HOOD_TYPE_OPTS = [
                ("solo_fritture_griglia", "Solo con fritture / griglia"),
                ("piatti_elaborati", "Con piatti più elaborati"),
                ("qualsiasi_piatto_caldo", "Quasi qualsiasi piatto caldo"),
            ]
            current_ht = hood.get("hood_cooking_type", "piatti_elaborati")
            if current_ht not in [o[0] for o in HOOD_TYPE_OPTS]:
                current_ht = "piatti_elaborati"
            hood["hood_cooking_type"] = col_h2.selectbox(
                "Per quali tipi di cottura la usi?",
                options=[o[0] for o in HOOD_TYPE_OPTS],
                index=[o[0] for o in HOOD_TYPE_OPTS].index(current_ht),
                format_func=lambda k: dict(HOOD_TYPE_OPTS)[k],
                key=f"c{consumer_id}_kitchen_hood_cooking_type",
            )

            col_h3, col_h4 = st.columns(2)
            HOOD_DUR_OPTS = [
                ("solo_inizio", "Solo all'inizio"),
                ("meta_tempo", "Per circa metà del tempo"),
                ("quasi_tutto", "Per quasi tutta la cottura"),
            ]
            current_hd = hood.get("hood_duration_relative", "meta_tempo")
            if current_hd not in [o[0] for o in HOOD_DUR_OPTS]:
                current_hd = "meta_tempo"
            hood["hood_duration_relative"] = col_h3.selectbox(
                "Durata tipica rispetto alla cottura",
                options=[o[0] for o in HOOD_DUR_OPTS],
                index=[o[0] for o in HOOD_DUR_OPTS].index(current_hd),
                format_func=lambda k: dict(HOOD_DUR_OPTS)[k],
                key=f"c{consumer_id}_kitchen_hood_duration_relative",
            )
            hood["hood_power_w"] = col_h4.number_input(
                "Potenza nominale cappa (W)",
                min_value=20.0, max_value=500.0,
                value=float(hood.get("hood_power_w", 100.0)),
                step=10.0,
                key=f"c{consumer_id}_kitchen_hood_power",
            )

        # ----- azioni -----
        acols = st.columns(3)
        if acols[0].button("💾 Salva parametri cucina", key=f"c{consumer_id}_kitchen_save"):
            save_consumers_json(SESSION_DIR, consumers)
            st.success("Parametri cucina salvati.")

        kitchen_curve = None
        if acols[1].button("⚙️ Calcola curva cucina", key=f"c{consumer_id}_kitchen_calc"):
            try:
                # costruisci KitchenConfig dalle dict
                hab_cfg = K.MealHabitConfig(**sanitize_for_dataclass(K.MealHabitConfig, dev_k.get("habits", {})))
                oven_cfg = K.OvenConfig(**sanitize_for_dataclass(K.OvenConfig, dev_k.get("oven", {})))
                ind_cfg = K.InductionConfig(**sanitize_for_dataclass(K.InductionConfig, dev_k.get("induction", {})))
                mw_cfg = K.MicrowaveConfig(**sanitize_for_dataclass(K.MicrowaveConfig, dev_k.get("microwave", {})))
                dw_cfg = K.DishwasherConfig(**sanitize_for_dataclass(K.DishwasherConfig, dev_k.get("dishwasher", {})))
                hood_cfg = K.HoodConfig(**sanitize_for_dataclass(K.HoodConfig, dev_k.get("hood", {})))

                kitchen_cfg = K.KitchenConfig(
                    habits=hab_cfg,
                    oven=oven_cfg,
                    induction=ind_cfg,
                    microwave=mw_cfg,
                    dishwasher=dw_cfg,
                    hood=hood_cfg,
                )

                seed_kitchen_base = derive_seed(BASE_SEED, "consumer", consumer_id, "kitchen")

                profiles = K.build_kitchen_profiles(
                    INDEX,
                    kitchen_cfg,
                    n_slots_per_day=len(SLOT_LABELS),
                    meal_matrix_seed=derive_seed(seed_kitchen_base, "meal_matrix"),
                    seed_oven=derive_seed(seed_kitchen_base, "oven"),
                    seed_induction=derive_seed(seed_kitchen_base, "induction"),
                    seed_microwave=derive_seed(seed_kitchen_base, "microwave"),
                    seed_dishwasher=derive_seed(seed_kitchen_base, "dishwasher"),
                )
                # curva completa a 15 minuti su tutto l'INDEX
                kitchen_curve = profiles["aggregated"].rename("kitchen").astype(float)

                # cache CSV sempre a 15 minuti (periodo completo)
                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer.get('id')}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                kitchen_curve.to_csv(cache_dir / "kitchen.csv", index=True, index_label="timestamp")

                # Visualizzazione: annuale = aggregata a 1h, mensile = 15 min sul mese scelto
                if curve_view_mode == "annuale" or curve_view_month is None:
                    kitchen_plot = kitchen_curve.resample("H").mean()
                else:
                    year, month = curve_view_month
                    mask_month = (kitchen_curve.index.year == year) & (kitchen_curve.index.month == month)
                    kitchen_plot = kitchen_curve[mask_month]

                st.line_chart(kitchen_plot)

                # KPI sul periodo completo simulato (conversione kW -> kWh)
                if len(kitchen_curve) > 1:
                    dt_hours = (kitchen_curve.index[1] - kitchen_curve.index[0]).total_seconds() / 3600.0
                else:
                    dt_hours = 0.0
                kwh_periodo = float(kitchen_curve.sum() * dt_hours)

                giorni = (INDEX.max() - INDEX.min()).days + 1
                kwh_giorno = kwh_periodo / max(giorni, 1)
                kwh_annuo = kwh_giorno * 365.0

                kc1, kc2, kc3, kc4 = st.columns(4)
                kc1.metric("Cucina: kWh periodo", f"{kwh_periodo:.1f}")
                kc2.metric("Cucina: kWh/anno stimato", f"{kwh_annuo:.0f}")
                kc3.metric("Cucina: media giornaliera", f"{kwh_giorno:.2f} kWh/g")


                # download CSV a 15 minuti (periodo completo)
                csv_buf = kitchen_curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
                kc4.download_button(
                    "⬇️ CSV 15 min (cucina – anno)",
                    data=csv_buf,
                    file_name=f"consumer_{consumer.get('id')}_kitchen_15min.csv",
                    mime="text/csv",
                    key=f"dl_{consumer.get('id')}_kitchen",
                )
            except Exception as e:
                st.error(f"Errore simulazione cucina: {e}")

        return kitchen_curve


# <<< NEW: pannello carichi base
