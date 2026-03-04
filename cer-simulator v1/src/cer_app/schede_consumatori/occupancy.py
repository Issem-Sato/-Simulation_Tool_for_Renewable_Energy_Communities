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


# --- Occupancy ---
# <<< NEW: defaults per occupancy (luci + TV + PC + charging)
OCCUPANCY_DEF = {
    "present": False,
    "version": 1,
    "timezone": "Europe/Rome",
    "matrix_resolution_minutes": 60,  # UI attuale: 60 min
    "variability": {"level": "medium"},  # "off" | "low" | "medium" | "high"
    "shared": {
        "tv": {
            "intensity": "medium",        # "low" | "medium" | "high"
            "time_pref": "both",          # "weekday_evening" | "weekend" | "both"
            "tv_count": 1,
            "power_class": "standard",    # "efficient" | "standard" | "large"
        },
        "lighting": {
            "tech": "led",                # "led" | "mixed" | "halogen"
            "style": "standard",          # "minimal" | "standard" | "intense"
            "switch_pref": "as_soon_as_dark",  # "as_soon_as_dark" | "evening_only" | "frugal"
            "twilight_minutes": 45,
        },
    },
    # lista residenti: viene allineata automaticamente a consumer['people']
    "residents": [],
}
OCC_STATE_LABELS = {
    0: "Fuori casa",
    1: "A casa (sveglio)",
    2: "A casa (dorme)",
}
OCC_STATE_INV = {v: k for k, v in OCC_STATE_LABELS.items()}

def _default_occ_state_matrix_7x24() -> List[List[int]]:
    """Default 7x24 coerente con occupancy.py: feriali fuori 9-18, weekend a casa."""
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
    """Normalizza matrice 7x24 con stati {0,1,2}."""
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
    """Editor matrice 7x24 a 3 stati con data_editor (scroll orizzontale)."""
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
    residents = dev_occ.get("residents")
    if not isinstance(residents, list):
        residents = []
    while len(residents) < n_residents:
        i = len(residents)
        residents.append({
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
        })
    if len(residents) > n_residents:
        residents = residents[:n_residents]
    dev_occ["residents"] = residents

def occupancy_panel():
    """Pannello Streamlit: raccolta parametri + simulazione occupancy."""
    consumer_id = consumer.get("id")
    dev_occ = ensure_device(consumer, "occupancy", OCCUPANCY_DEF)

    n_residents = int(consumer.get("people") or 1)
    _ensure_occ_residents(dev_occ, n_residents)

    with st.expander("🏠 Occupancy (luci + TV + PC + ricariche)", expanded=True):
        dev_occ["present"] = st.checkbox(
            "Abilita modulo occupancy",
            value=bool(dev_occ.get("present", False)),
            key=f"c{consumer_id}_occ_present",
        )

        # Parametri globali
        c1, c2, c3 = st.columns(3)
        var_level = c1.selectbox(
            "Variabilità (jitter settimana tipo)",
            options=["off", "low", "medium", "high"],
            index=["off", "low", "medium", "high"].index(
                ((dev_occ.get("variability") or {}).get("level") or "medium")
                if ((dev_occ.get("variability") or {}).get("level") or "medium") in ["off", "low", "medium", "high"]
                else "medium"
            ),
            key=f"c{consumer_id}_occ_var_level",
        )
        dev_occ.setdefault("variability", {})["level"] = var_level

        # Shared: TV
        st.markdown("### Carichi condivisi")
        tv = (dev_occ.setdefault("shared", {}).setdefault("tv", {}))
        t1, t2, t3, t4 = st.columns(4)
        tv["intensity"] = t1.selectbox(
            "TV: intensità",
            options=["low", "medium", "high"],
            index=["low", "medium", "high"].index(tv.get("intensity", "medium") if tv.get("intensity", "medium") in ["low","medium","high"] else "medium"),
            format_func=lambda v: {"low": "Bassa", "medium": "Media", "high": "Alta"}[v],
            key=f"c{consumer_id}_occ_tv_int",
        )
        tv["time_pref"] = t2.selectbox(
            "TV: quando",
            options=["weekday_evening", "weekend", "both"],
            index=["weekday_evening", "weekend", "both"].index(tv.get("time_pref", "both") if tv.get("time_pref", "both") in ["weekday_evening","weekend","both"] else "both"),
            format_func=lambda v: {"weekday_evening": "Sera feriali", "weekend": "Weekend", "both": "Entrambi"}[v],
            key=f"c{consumer_id}_occ_tv_time",
        )
        tv["tv_count"] = int(t3.number_input(
            "Numero TV",
            min_value=1, max_value=5, step=1,
            value=int(tv.get("tv_count", 1) or 1),
            key=f"c{consumer_id}_occ_tv_count",
        ))
        tv["power_class"] = t4.selectbox(
            "Classe TV",
            options=["efficient", "standard", "large"],
            index=["efficient", "standard", "large"].index(tv.get("power_class", "standard") if tv.get("power_class", "standard") in ["efficient","standard","large"] else "standard"),
            format_func=lambda v: {"efficient": "Efficiente", "standard": "Standard", "large": "Grande"}[v],
            key=f"c{consumer_id}_occ_tv_pclass",
        )

        # Shared: Lighting
        lighting = (dev_occ.setdefault("shared", {}).setdefault("lighting", {}))
        l1, l2, l3, l4 = st.columns(4)
        lighting["tech"] = l1.selectbox(
            "Illuminazione: tecnologia",
            options=["led", "mixed", "halogen"],
            index=["led", "mixed", "halogen"].index(lighting.get("tech", "led") if lighting.get("tech", "led") in ["led","mixed","halogen"] else "led"),
            format_func=lambda v: {"led": "LED", "mixed": "Mista", "halogen": "Alogena"}[v],
            key=f"c{consumer_id}_occ_l_tech",
        )
        lighting["style"] = l2.selectbox(
            "Illuminazione: stile",
            options=["minimal", "standard", "intense"],
            index=["minimal", "standard", "intense"].index(lighting.get("style", "standard") if lighting.get("style","standard") in ["minimal","standard","intense"] else "standard"),
            format_func=lambda v: {"minimal": "Minimale", "standard": "Standard", "intense": "Intenso"}[v],
            key=f"c{consumer_id}_occ_l_style",
        )
        lighting["switch_pref"] = l3.selectbox(
            "Accensione luci",
            options=["as_soon_as_dark", "evening_only", "frugal"],
            index=["as_soon_as_dark", "evening_only", "frugal"].index(lighting.get("switch_pref", "as_soon_as_dark") if lighting.get("switch_pref","as_soon_as_dark") in ["as_soon_as_dark","evening_only","frugal"] else "as_soon_as_dark"),
            format_func=lambda v: {"as_soon_as_dark": "Appena fa buio", "evening_only": "Solo la sera", "frugal": "Parsimonioso"}[v],
            key=f"c{consumer_id}_occ_l_switch",
        )
        lighting["twilight_minutes"] = int(l4.slider(
            "Crepuscolo (minuti)",
            min_value=0, max_value=120, step=5,
            value=int(lighting.get("twilight_minutes", 45) or 45),
            key=f"c{consumer_id}_occ_l_tw",
        ))

        # Residenti
        st.markdown("### Residenti (settimana tipo + dispositivi personali)")
        tabs = st.tabs([f"Residente {i+1}" for i in range(n_residents)])
        for i, tab in enumerate(tabs):
            with tab:
                r = dev_occ["residents"][i]

                r["label"] = st.text_input(
                    "Etichetta (opzionale)",
                    value=str(r.get("label") or f"Residente {i+1}"),
                    key=f"c{consumer_id}_occ_r{i}_label",
                )

                st.markdown("#### Presenza in casa (settimana tipo)")
                sched = r.setdefault("schedule", {})
                sched["state_matrix_7x24"] = occ_state_matrix_editor(
                    key=f"c{consumer_id}_occ_r{i}_sched",
                    init_matrix=sched.get("state_matrix_7x24"),
                )

                st.markdown("#### PC / lavoro / studio")
                pc = r.setdefault("pc", {})
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc["type"] = pc1.selectbox(
                    "Tipo PC",
                    options=["none", "laptop", "desktop"],
                    index=["none", "laptop", "desktop"].index(pc.get("type", "laptop") if pc.get("type","laptop") in ["none","laptop","desktop"] else "laptop"),
                    format_func=lambda v: {"none": "Nessuno", "laptop": "Laptop", "desktop": "Desktop"}[v],
                    key=f"c{consumer_id}_occ_r{i}_pc_type",
                )
                pc["intensity"] = pc2.selectbox(
                    "Uso PC",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(pc.get("intensity", "low") if pc.get("intensity","low") in ["low","medium","high"] else "low"),
                    format_func=lambda v: {"low": "Basso", "medium": "Medio", "high": "Alto"}[v],
                    key=f"c{consumer_id}_occ_r{i}_pc_int",
                )
                pc["time_pref"] = pc3.selectbox(
                    "Preferenza oraria",
                    options=["morning", "afternoon", "evening", "mixed"],
                    index=["morning","afternoon","evening","mixed"].index(pc.get("time_pref","mixed") if pc.get("time_pref","mixed") in ["morning","afternoon","evening","mixed"] else "mixed"),
                    format_func=lambda v: {"morning": "Mattina", "afternoon": "Pomeriggio", "evening": "Sera", "mixed": "Misto"}[v],
                    key=f"c{consumer_id}_occ_r{i}_pc_pref",
                )
                pc["weekend_enabled"] = pc4.checkbox(
                    "Usa PC anche nel weekend",
                    value=bool(pc.get("weekend_enabled", True)),
                    key=f"c{consumer_id}_occ_r{i}_pc_we",
                )

                st.markdown("#### Ricariche dispositivi (smartphone)")
                ch = r.setdefault("charging", {})
                ch1, ch2, ch3 = st.columns(3)
                ch["phone_intensity"] = ch1.selectbox(
                    "Intensità ricarica",
                    options=["low", "medium", "high"],
                    index=["low","medium","high"].index(ch.get("phone_intensity","medium") if ch.get("phone_intensity","medium") in ["low","medium","high"] else "medium"),
                    format_func=lambda v: {"low": "Bassa", "medium": "Media", "high": "Alta"}[v],
                    key=f"c{consumer_id}_occ_r{i}_ch_int",
                )
                ch["preference"] = ch2.selectbox(
                    "Quando ricarichi",
                    options=["night", "return_home", "distributed"],
                    index=["night","return_home","distributed"].index(ch.get("preference","night") if ch.get("preference","night") in ["night","return_home","distributed"] else "night"),
                    format_func=lambda v: {"night": "Notte", "return_home": "Al rientro", "distributed": "Distribuito"}[v],
                    key=f"c{consumer_id}_occ_r{i}_ch_pref",
                )
                ch["charge_type"] = ch3.selectbox(
                    "Tipo caricatore",
                    options=["standard", "fast"],
                    index=["standard","fast"].index(ch.get("charge_type","standard") if ch.get("charge_type","standard") in ["standard","fast"] else "standard"),
                    format_func=lambda v: {"standard": "Standard", "fast": "Fast"}[v],
                    key=f"c{consumer_id}_occ_r{i}_ch_type",
                )

        b1, b2 = st.columns([1, 2])
        if b1.button("💾 Salva parametri occupancy", key=f"c{consumer_id}_occ_save"):
            save_consumers_json(SESSION_DIR, consumers)
            st.success("Parametri occupancy salvati.")

        curve = None
        if dev_occ.get("present", False) and b2.button(
            "⚙️ Calcola curva occupancy (luci + TV + PC + ricariche)",
            key=f"c{consumer_id}_occ_calc",
        ):
            try:
                seed_occ = derive_seed(BASE_SEED, "consumer", consumer_id, "occupancy")
                occ_cfg = O.OccupancyConfig.from_dict(dev_occ, n_residents=n_residents, seed=seed_occ)
                profiles = O.build_occupancy_profiles(INDEX, occ_cfg)
                curve = profiles["aggregated"].rename("occupancy").astype(float)

                cache_dir = SESSION_DIR / "cache" / f"consumer_{consumer_id}"
                cache_dir.mkdir(parents=True, exist_ok=True)
                curve.to_csv(cache_dir / "occupancy_15min.csv", index=True, index_label="timestamp")

                if curve_view_mode == "annuale" or curve_view_month is None:
                    curve_plot = curve.resample("H").mean()
                else:
                    year, month = curve_view_month
                    mask_month = (curve.index.year == year) & (curve.index.month == month)
                    curve_plot = curve[mask_month]

                st.line_chart(curve_plot)

                if len(curve) > 1:
                    dt_hours = (curve.index[1] - curve.index[0]).total_seconds() / 3600.0
                else:
                    dt_hours = 0.0
                kwh_periodo = float(curve.sum() * dt_hours)

                giorni = (INDEX.max() - INDEX.min()).days + 1
                kwh_giorno = kwh_periodo / max(giorni, 1)
                kwh_annuo = kwh_giorno * 365.0

                cols = st.columns(4)
                cols[0].metric("Occupancy: kWh periodo", f"{kwh_periodo:.1f}")
                cols[1].metric("Occupancy: kWh/anno stimato", f"{kwh_annuo:.0f}")
                cols[2].metric("Occupancy: media giornaliera", f"{kwh_giorno:.2f} kWh/g")

                csv_buf = curve.to_csv(index=True, index_label="timestamp").encode("utf-8")
                cols[3].download_button(
                    "⬇️ CSV 15 min (occupancy – anno)",
                    data=csv_buf,
                    file_name=f"consumer_{consumer_id}_occupancy_15min.csv",
                    mime="text/csv",
                    key=f"dl_{consumer_id}_occ",
                )

                if st.checkbox("Mostra breakdown (luci/TV/PC/ricariche)", key=f"c{consumer_id}_occ_breakdown"):
                    br = pd.DataFrame({
                        "lighting": profiles.get("lighting", 0.0),
                        "tv": profiles.get("tv", 0.0),
                        "pc_total": profiles.get("pc_total", 0.0),
                        "charging_total": profiles.get("charging_total", 0.0),
                    }).astype(float)
                    if curve_view_mode == "annuale" or curve_view_month is None:
                        br_plot = br.resample("H").mean()
                    else:
                        year, month = curve_view_month
                        mask_month = (br.index.year == year) & (br.index.month == month)
                        br_plot = br[mask_month]
                    st.line_chart(br_plot)

            except Exception as e:
                st.error(f"Errore simulazione occupancy: {e}")

        return curve
