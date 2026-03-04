# -*- coding: utf-8 -*-
"""cer_app.pages.3_Bilanciamento_CER
================================

Pagina Streamlit dedicata al **bilanciamento energetico** di una Comunità Energetica
Rinnovabile (CER). Il modulo implementa l'orchestrazione UI e la persistenza su
filesystem di:

- gestione degli **scenari** (workspace di input: membri, file, metadati periodo);
- esecuzione e **versioning** delle run energetiche (outputs per run + indice run);
- rendering dei risultati (KPI, grafici, download CSV) e dettaglio per singolo membro.

Architettura e responsabilità
-----------------------------
- `cer_app/` (questo modulo): UI Streamlit, validazione input a livello pagina,
  gestione stato (`st.session_state`), I/O file upload e navigazione risultati.
- `cer_core/bilanciamento/`: logica di dominio (validazione dati, trasformazioni
  temporali, calcolo flussi orari per membro, aggregazione CER, salvataggi output).

Persistenza su filesystem (per scenario)
----------------------------------------
La pagina lavora su una directory scenario:

    <session_dir>/bilanciamento/scenarios/<scenario_name>/
        scenario.json                # input scenario (membri + periodo)
        inputs/
            member_<id>/
                consumption_15min.csv    # richiesto
                production_hourly.csv    # opzionale (PVGIS wide orario)
        outputs/
            run_<run_id>/
                cer_hourly.csv
                members_hourly_long.csv
                members_summary.csv
                cer_summary.csv
                period_meta.csv
                run_config.json          # snapshot input usato per la run
                run_meta.json            # metadata e label run
        runs_index.jsonl             # append-only registry delle run
        active_run.txt               # puntatore run attiva per scenario

Formati input attesi
--------------------
- Consumi (`consumption_15min.csv`):
    - frequenza: 15 minuti
    - potenza: kW
    - parsing e validazione sono demandati a `cer_core.bilanciamento` tramite
      `load_and_validate_member()`.
- Produzione (`production_hourly.csv`, opzionale):
    - CSV orario in formato "wide" (es. PVGIS), con colonna temporale e colonne
      di potenza in kW (es. "Totale", "Area_*"), selezionate via `prod_mode` e
      `selected_areas` nel membro.
    - parsing e selezione colonne sono demandati a `cer_core.bilanciamento`.

Assunzioni e invarianti temporali
---------------------------------
- Gli indici temporali dei DataFrame risultato sono `DatetimeIndex` timezone-aware.
- La UI normalizza lettura e visualizzazione in **UTC** (`pd.to_datetime(..., utc=True)`).
- Il periodo di simulazione (`PeriodConfig`) è determinato automaticamente al
  primo caricamento consumi e persistito nello scenario.

Side effects rilevanti
----------------------
- Creazione directory scenario (`inputs/`, `outputs/`) e migrazione da legacy.
- Salvataggio file caricati dall'utente in `inputs/member_<id>/`.
- Creazione nuova run in `outputs/run_<run_id>/` quando si preme "Salva modifiche".
- Cancellazione directory input del membro quando si elimina un membro.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.express as px
import pandas as pd
import streamlit as st

from cer_app.session_paths import get_paths

from cer_core.bilanciamento.bilanciamento_energetico import (
    BatterySpec,
    MemberSpec,
    PeriodConfig,
    ProductionSpec,
    build_period_config,
    compute_cer_hourly,
    compute_member_energy_hourly,
    infer_period_from_consumption,
    load_and_validate_member,
    save_outputs,
    summarize_cer,
    summarize_member,
)

from cer_core.bilanciamento.run_registry import (
    append_energy_run_record,
    get_active_energy_run,
    normalize_run_label,
    set_active_energy_run,
    write_energy_run_meta,
)
from cer_core.bilanciamento.scenario import (
    migrate_legacy_to_scenario,
    load_scenario,
    get_members,
    get_period_meta,
    update_members,
    update_period,
)



# =============================================================================
# Page config
# =============================================================================

PAGE_TITLE = "Bilanciamento CER"
st.set_page_config(page_title=f"CER - {PAGE_TITLE}", page_icon="⚖️", layout="wide")
st.title("Bilanciamento energetico CER")

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,  # zoom con rotellina mouse
}

# =============================================================================
# Session paths (unificati)
# =============================================================================

PATHS = get_paths()
SESSION_DIR = PATHS.session_dir

BIL_DIR = PATHS.bil_dir

# -----------------------------------------------------------------------------
# Scenari (nuova UX): uno Scenario contiene input (membri + file) e una sequenza
# di run/versioni (outputs + index jsonl).
# -----------------------------------------------------------------------------

SCENARIOS_DIR = BIL_DIR / "scenarios"
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_MEMBERS_JSON = PATHS.bil_members_json
LEGACY_PERIOD_JSON = PATHS.bil_period_json


def _normalize_scenario_name(name: str) -> str:
    """Normalizza un nome scenario per l'uso come nome cartella.
    
    La normalizzazione:
    - trim e compressione spazi in underscore;
    - rimozione di caratteri non alfanumerici (ammessi: A-Z a-z 0-9, '_' e '-');
    - sostituzione di separatori di path e sequenze potenzialmente pericolose.
    
    Parameters
    ----------
    name : str
        Nome scenario inserito dall'utente.
    
    Returns
    -------
    str
        Nome scenario safe per filesystem. Se vuoto, ritorna "default".
    """
    import re

    n = (name or "").strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^A-Za-z0-9_\-]", "", n)
    n = n.replace("..", "_").replace("/", "_").replace("\\", "_")
    return n or "default"


def _scenario_dir(scenario_name: str) -> Path:
    """Restituisce la directory radice dello scenario.
    
    Parameters
    ----------
    scenario_name : str
        Nome scenario (non necessariamente normalizzato).
    
    Returns
    -------
    pathlib.Path
        Path della directory: <BIL_DIR>/scenarios/<normalized_name>.
    """
    return SCENARIOS_DIR / _normalize_scenario_name(scenario_name)


def _scenario_paths(scenario_name: str) -> dict:
    """Costruisce e garantisce la struttura directory/file per uno scenario.
    
    Crea (se mancanti) le directory:
    - inputs/
    - outputs/
    
    e restituisce i path dei file di scenario e registry run.
    
    Parameters
    ----------
    scenario_name : str
        Nome scenario (normalizzato internamente).
    
    Returns
    -------
    dict
        Dizionario con chiavi:
        - dir, inputs_dir, outputs_dir
        - scenario_json
        - runs_index_jsonl
        - active_run_txt
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


def _list_scenarios() -> List[str]:
    """Elenca gli scenari disponibili.
    
    Uno scenario è considerato valido se:
    - esiste una directory in SCENARIOS_DIR
    - contiene il file 'scenario.json'
    
    Returns
    -------
    List[str]
        Elenco nomi scenario (cartelle), includendo "default" come fallback.
    """
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[str] = []
    for p in sorted(SCENARIOS_DIR.glob("*")):
        if p.is_dir() and (p / "scenario.json").exists():
            out.append(p.name)
    if "default" not in out:
        out.insert(0, "default")
    return out


def _ensure_default_scenario() -> None:
    """Bootstrap: se non esiste alcuno scenario, crea 'default' e migra legacy."""
    has_any = any(p.is_dir() for p in SCENARIOS_DIR.glob("*"))
    sp = _scenario_paths("default")
    if sp["scenario_json"].exists():
        return

    # Migra scenario legacy (session_dir/scenario.json) se esiste, altrimenti legacy members/period.
    legacy_scenario_path = PATHS.scenario_json
    if legacy_scenario_path.exists():
        try:
            sc = json.loads(legacy_scenario_path.read_text(encoding="utf-8"))
        except Exception:
            sc = None
        if isinstance(sc, dict):
            # Normalizza chiavi minime
            sc.setdefault("members", [])
            sc.setdefault("period", {})
            (sp["scenario_json"]).write_text(json.dumps(sc, ensure_ascii=False, indent=2), encoding="utf-8")
            return

    # Fallback: migra dai legacy json del bilanciamento.
    _ = migrate_legacy_to_scenario(sp["scenario_json"], LEGACY_MEMBERS_JSON, LEGACY_PERIOD_JSON)


_ensure_default_scenario()

# -----------------------------------------------------------------------------
# Active scenario pointers (recomputed on each rerun)
# -----------------------------------------------------------------------------

if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = "default"

ACTIVE_SCENARIO = _normalize_scenario_name(st.session_state["active_scenario"])
ACTIVE_SP = _scenario_paths(ACTIVE_SCENARIO)

INPUTS_DIR = ACTIVE_SP["inputs_dir"]
OUTPUTS_DIR = ACTIVE_SP["outputs_dir"]
SCENARIO_JSON = ACTIVE_SP["scenario_json"]
RUNS_INDEX_JSONL = ACTIVE_SP["runs_index_jsonl"]
ACTIVE_RUN_TXT = ACTIVE_SP["active_run_txt"]


def member_dir(member_id: int) -> Path:
    """Restituisce (e crea se necessario) la directory input del membro.
    
    Parameters
    ----------
    member_id : int
        Identificativo membro.
    
    Returns
    -------
    pathlib.Path
        Directory: <scenario>/inputs/member_<id>/
    """
    d = INPUTS_DIR / f"member_{member_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def consumption_path(member_id: int) -> Path:
    """Path del file consumi a 15 minuti per un membro.
    
    Returns
    -------
    pathlib.Path
        <scenario>/inputs/member_<id>/consumption_15min.csv
    """
    return member_dir(member_id) / "consumption_15min.csv"


def production_path(member_id: int) -> Path:
    """Path del file produzione oraria per un membro.
    
    Returns
    -------
    pathlib.Path
        <scenario>/inputs/member_<id>/production_hourly.csv
    """
    return member_dir(member_id) / "production_hourly.csv"


def has_consumption(member_id: int) -> bool:
    """Verifica l'esistenza del file consumi per il membro.
    
    Returns
    -------
    bool
        True se esiste consumption_15min.csv.
    """
    return consumption_path(member_id).exists()


def has_production(member_id: int) -> bool:
    """Verifica l'esistenza del file produzione per il membro.
    
    Returns
    -------
    bool
        True se esiste production_hourly.csv.
    """
    return production_path(member_id).exists()


def save_uploaded_file(uploaded, dst: Path) -> None:
    """Salva su filesystem un file caricato via Streamlit.
    
    Parameters
    ----------
    uploaded
        Oggetto restituito da `st.file_uploader`.
    dst : pathlib.Path
        Destinazione del file.
    
    Side Effects
    ------------
    - Scrive bytes su `dst`, creando la directory padre se necessaria.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(uploaded.getbuffer())


# =============================================================================
# Persistence: members
# =============================================================================

def migrate_members_schema(members: List[dict]) -> List[dict]:
    """Migra/normalizza lo schema membri verso il formato canonico usato dalla UI.
    
    Migrazione principale:
    - legacy: 'battery_kwh'
    - schema corrente: 'has_battery' + campi 'battery_*'
    
    La funzione applica inoltre valori di default per campi attesi dalla UI:
    - metadati membro (role, num, voltage_level, new_plant, commissioning_month, installed_capacity_kw)
    - configurazione produzione (prod_mode, selected_areas)
    - configurazione batteria (capacity_kwh, dod, roundtrip_eff, derating_factor, init_soc_perc)
    
    Parameters
    ----------
    members : List[dict]
        Lista membri letta dallo scenario (potenzialmente legacy).
    
    Returns
    -------
    List[dict]
        Lista membri normalizzata con tutte le chiavi attese dalla pagina.
    
    Notes
    -----
    La migrazione è progettata per essere idempotente rispetto allo schema corrente.
    """
    out = []
    for m in members or []:
        mm = dict(m)

        # 1) Migrazione legacy battery_kwh -> new fields
        if "has_battery" not in mm:
            cap_legacy = mm.pop("battery_kwh", None)
            try:
                cap_val = float(cap_legacy) if cap_legacy not in (None, "", "None") else 0.0
            except Exception:
                cap_val = 0.0

            mm["has_battery"] = cap_val > 0
            mm["battery_capacity_kwh"] = cap_val

        # 2) Default parametri batteria (se mancanti)
        mm.setdefault("battery_capacity_kwh", 0.0)
        mm.setdefault("battery_dod", 0.8)
        mm.setdefault("battery_roundtrip_eff", 0.9)
        mm.setdefault("battery_derating_factor", 0.0)
        mm.setdefault("battery_init_soc_perc", 0.2)

        # 3) Default metadati (utile per RSE-like registry)
        mm.setdefault("role", "prosumer")
        mm.setdefault("num", 1)
        mm.setdefault("voltage_level", "BT")
        mm.setdefault("new_plant", True)
        mm.setdefault("commissioning_month", "")
        mm.setdefault("installed_capacity_kw", 0.0)

        # 4) Campi già esistenti nel tuo flusso produzione
        mm.setdefault("prod_mode", "totale")
        mm.setdefault("selected_areas", [])

        out.append(mm)

    return out


def load_members() -> List[dict]:
    """Carica i membri dallo scenario attivo.
    
    Se `scenario.json` non è disponibile o non è valido, tenta una migrazione
    da legacy tramite `migrate_legacy_to_scenario()` e poi rilegge.
    
    Returns
    -------
    List[dict]
        Lista membri come dict (non necessariamente migrati: la migrazione schema
        UI avviene in `migrate_members_schema()`).
    """
    scen = load_scenario(SCENARIO_JSON)
    if scen is None:
        scen = migrate_legacy_to_scenario(SCENARIO_JSON, LEGACY_MEMBERS_JSON, LEGACY_PERIOD_JSON)
    return get_members(scen)


def save_members(members: List[dict]) -> None:
    """Persiste i membri nello scenario attivo.
    
    Side Effects
    ------------
    - Scrive su `scenario.json` dello scenario.
    - Può aggiornare anche i path legacy se `legacy_members_path` è valorizzato
      (delegato a `update_members()`).
    """
    update_members(SCENARIO_JSON, members, legacy_members_path=LEGACY_MEMBERS_JSON)


def next_member_id(members: List[dict]) -> int:
    """Calcola il prossimo identificativo membro disponibile (incrementale).
    
    Parameters
    ----------
    members : List[dict]
        Lista membri attuale.
    
    Returns
    -------
    int
        max(id)+1 se presenti membri, altrimenti 1.
    """
    if not members:
        return 1
    ids = []
    for m in members:
        try:
            ids.append(int(m.get("id", 0)))
        except Exception:
            pass
    return (max(ids) if ids else 0) + 1


# =========================
# Registry helpers (stile RSE)
# =========================

def default_member(member_id: int) -> dict:
    """Costruisce un dizionario membro con schema canonico e valori di default.
    
    Il risultato include:
    - campi anagrafici (id, name)
    - metadati tecnici (role, num, voltage_level, new_plant, commissioning_month, installed_capacity_kw)
    - configurazione produzione (prod_mode, selected_areas)
    - configurazione batteria (has_battery e parametri battery_*)
    
    Parameters
    ----------
    member_id : int
        Identificativo del membro.
    
    Returns
    -------
    dict
        Dizionario membro pronto per essere fuso con eventuali dati persistiti.
    """
    return {
        "id": int(member_id),
        "name": f"Membro {member_id}",

        # metadati RSE-like
        "role": "prosumer",              # consumer | producer | prosumer
        "num": 1,                        # numerosità
        "voltage_level": "BT",           # BT | MT

        # TIP/seniority (non usati ora, ma pronti)
        "new_plant": True,
        "commissioning_month": "",       # "YYYY-MM"
        "installed_capacity_kw": 0.0,

        # produzione
        "prod_mode": "totale",           # totale | aree
        "selected_areas": [],

        # batteria (RSE-like)
        "has_battery": False,
        "battery_capacity_kwh": 0.0,
        "battery_dod": 0.8,
        "battery_roundtrip_eff": 0.9,
        "battery_derating_factor": 0.0,
        "battery_init_soc_perc": 0.2,
    }


REGISTRY_COLUMNS = [
    "id",
    "name",
    "role",
    "num",
    "voltage_level",
    "new_plant",
    "commissioning_month",
    "installed_capacity_kw",
    "has_battery",
    "battery_capacity_kwh",
    "battery_dod",
    "battery_roundtrip_eff",
    "battery_derating_factor",
    "battery_init_soc_perc",
    "prod_mode",
]


def members_to_registry_df(members: List[dict]) -> pd.DataFrame:
    """Converte la lista membri (dict) in un DataFrame editabile via `st.data_editor`.
    
    Per ogni membro:
    - costruisce una base con `default_member(id)`;
    - applica override con i valori persistiti;
    - esporta solo le colonne definite in `REGISTRY_COLUMNS`.
    
    Parameters
    ----------
    members : List[dict]
        Lista membri con chiavi eterogenee.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame con colonne e tipi coerenti con la UI.
    """
    rows = []
    for m in members or []:
        mid = int(m.get("id", 0) or 0)
        mm = default_member(mid)
        mm.update(m)  # merge: preserva chiavi già presenti
        rows.append({c: mm.get(c) for c in REGISTRY_COLUMNS})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["id"] = df["id"].astype(int)
        df["num"] = df["num"].astype(int)
        df["installed_capacity_kw"] = df["installed_capacity_kw"].astype(float)
        df["battery_capacity_kwh"] = df["battery_capacity_kwh"].astype(float)
        df["battery_dod"] = df["battery_dod"].astype(float)
        df["battery_roundtrip_eff"] = df["battery_roundtrip_eff"].astype(float)
        df["battery_derating_factor"] = df["battery_derating_factor"].astype(float)
        df["battery_init_soc_perc"] = df["battery_init_soc_perc"].astype(float)
    return df


def registry_df_to_members(df: pd.DataFrame, existing: List[dict]) -> List[dict]:
    """Converte un DataFrame del registry (editato in UI) in lista membri (dict).
    
    Aggiorna i soli campi presenti nel DataFrame e preserva campi extra
    non esposti (es. `selected_areas`) prendendoli da `existing`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame proveniente da `st.data_editor`.
    existing : List[dict]
        Lista membri corrente, usata come base per preservare campi non presenti nel df.
    
    Returns
    -------
    List[dict]
        Lista membri normalizzata e ordinata per id.
    """
    by_id = {int(m["id"]): dict(m) for m in (existing or []) if "id" in m}
    out: List[dict] = []

    for _, r in df.iterrows():
        mid = int(r["id"])
        base = default_member(mid)
        base.update(by_id.get(mid, {}))  # preserva campi extra (es. selected_areas)

        # aggiorna solo i campi del registry
        for c in REGISTRY_COLUMNS:
            base[c] = r[c]

        # normalizza
        base["id"] = int(base["id"])
        base["num"] = int(base.get("num", 1))
        out.append(base)

    out.sort(key=lambda x: int(x["id"]))
    return out

# =========================
# Validazione pre-run (controlli coerenza)
# =========================

def validate_registry(members: List[dict], period: PeriodConfig) -> Tuple[List[str], List[str]]:
    """Esegue controlli di coerenza pre-run sui membri e sugli input caricati.
    
    Controlli implementati:
    1) presenza file consumi per ogni membro (bloccante);
    2) validità metadati minimi (num>0, role in {consumer,producer,prosumer}, voltage_level in {BT,MT});
    3) coerenza produzione:
       - se file produzione presente: prod_mode in {totale,aree} e selected_areas richiesto se prod_mode=aree;
       - se file produzione assente e role in {producer,prosumer}: warning;
    4) parametri batteria se has_battery=True (range check, bloccante);
    5) campi TIP/seniority (commissioning_month, installed_capacity_kw) come warning.
    
    Parameters
    ----------
    members : List[dict]
        Lista membri scenario.
    period : PeriodConfig
        Periodo simulazione. **Nota:** nel codice attuale non è usato per validazioni
        di copertura temporale; è mantenuto per estensioni future.
    
    Returns
    -------
    (List[str], List[str])
        (errors, warnings), dove errors sono bloccanti per l'esecuzione.
    
    Notes
    -----
    - La funzione non valida la struttura interna dei CSV: tale validazione è
      demandata a `cer_core.bilanciamento.load_and_validate_member()`.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1) file consumi presenti
    for m in members:
        mid = int(m["id"])
        if not has_consumption(mid):
            errors.append(f"Membro {mid}: manca file consumi (15min).")

    # 2) numerosità e metadati minimi
    for m in members:
        mid = int(m["id"])

        try:
            num = int(m.get("num", 1))
        except Exception:
            num = 0

        if num <= 0:
            errors.append(f"Membro {mid}: num deve essere > 0.")

        role = str(m.get("role", "prosumer")).lower()
        if role not in ("consumer", "producer", "prosumer"):
            errors.append(f"Membro {mid}: role non valido ({role}).")

        vl = str(m.get("voltage_level", "BT")).upper()
        if vl not in ("BT", "MT"):
            errors.append(f"Membro {mid}: voltage_level non valido ({vl}).")

    # 3) produzione e aree
    for m in members:
        mid = int(m["id"])
        if has_production(mid):
            mode = str(m.get("prod_mode", "totale")).lower()
            if mode not in ("totale", "aree"):
                errors.append(f"Membro {mid}: prod_mode non valido ({mode}).")
            if mode == "aree":
                if not (m.get("selected_areas") or []):
                    errors.append(f"Membro {mid}: prod_mode=aree ma nessuna area selezionata.")
        else:
            # Se è producer/prosumer e non ha produzione, warning (non sempre bloccante)
            role = str(m.get("role", "prosumer")).lower()
            if role in ("producer", "prosumer"):
                warnings.append(f"Membro {mid}: ruolo {role} ma manca file produzione.")

    # 4) batteria: parametri
    for m in members:
        mid = int(m["id"])
        if bool(m.get("has_battery", False)):
            cap = float(m.get("battery_capacity_kwh", 0.0) or 0.0)
            dod = float(m.get("battery_dod", 0.8))
            rte = float(m.get("battery_roundtrip_eff", 0.9))
            der = float(m.get("battery_derating_factor", 0.0))
            init_soc = float(m.get("battery_init_soc_perc", 0.2))

            if cap <= 0:
                errors.append(f"Membro {mid}: batteria attiva ma battery_capacity_kwh <= 0.")
            if not (0 < dod <= 1):
                errors.append(f"Membro {mid}: battery_dod fuori range (0,1].")
            if not (0 < rte <= 1):
                errors.append(f"Membro {mid}: battery_roundtrip_eff fuori range (0,1].")
            if not (0 <= der < 1):
                errors.append(f"Membro {mid}: battery_derating_factor fuori range [0,1).")
            if not (0 <= init_soc <= 1):
                errors.append(f"Membro {mid}: battery_init_soc_perc fuori range [0,1].")

    # 5) TIP metadata (non bloccante adesso, ma prepara il terreno)
    for m in members:
        mid = int(m["id"])
        if bool(m.get("new_plant", True)):
            # commissioning_month e installed_capacity_kw sono utili per TIP/seniority
            cm = str(m.get("commissioning_month", "")).strip()
            cap_kw = float(m.get("installed_capacity_kw", 0.0) or 0.0)
            if cm == "":
                warnings.append(f"Membro {mid}: new_plant=True ma commissioning_month vuoto (warning).")
            if cap_kw <= 0 and has_production(mid):
                warnings.append(f"Membro {mid}: installed_capacity_kw <= 0 (warning).")

    return errors, warnings


# =============================================================================
# Persistence: period
# =============================================================================


def plot_timeseries(df: pd.DataFrame, y_cols: List[str], height: int = 320):
    """Crea un grafico Plotly per serie temporali.
    
    Accetta un DataFrame con `DatetimeIndex` (caso standard dei risultati) oppure con
    colonna `time`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dati da plottare.
    y_cols : List[str]
        Colonne numeriche da visualizzare.
    height : int
        Altezza del grafico in pixel.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figura Plotly.
    
    Raises
    ------
    ValueError
        Se manca un asse temporale (DatetimeIndex o colonna `time`).
    """
    if isinstance(df.index, pd.DatetimeIndex):
        d = df.reset_index()
        # la prima colonna dopo reset_index è il tempo (può chiamarsi 'index' o il nome dell'indice)
        time_col = d.columns[0]
        d = d.rename(columns={time_col: "time"})
    else:
        d = df.copy()
        if "time" not in d.columns:
            raise ValueError("plot_timeseries: manca colonna 'time' e l'indice non è DatetimeIndex.")

    fig = px.line(d, x="time", y=y_cols)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(rangeslider_visible=True)  # slider sotto per zoom su intervalli
    return fig

def load_period() -> Optional[PeriodConfig]:
    """Carica il periodo di simulazione (PeriodConfig) dallo scenario attivo.
    
    Legge i metadati periodo (tz, t0, t1) dallo scenario e costruisce un
    `PeriodConfig` tramite `build_period_config()`.
    
    Returns
    -------
    Optional[PeriodConfig]
        Periodo se presente e valido, altrimenti None.
    """
    scen = load_scenario(SCENARIO_JSON)
    if scen is None:
        scen = migrate_legacy_to_scenario(SCENARIO_JSON, LEGACY_MEMBERS_JSON, LEGACY_PERIOD_JSON)
    meta = get_period_meta(scen)
    if not meta or ("t0" not in meta) or ("t1" not in meta):
        return None
    try:
        tz = meta.get("tz", "UTC")
        t0 = pd.Timestamp(meta["t0"])
        t1 = pd.Timestamp(meta["t1"])
        return build_period_config(t0, t1, tz=tz)
    except Exception:
        return None


def save_period(period: PeriodConfig) -> None:
    """Persiste il periodo di simulazione nello scenario attivo.
    
    Side Effects
    ------------
    - Aggiorna `scenario.json` con {"tz","t0","t1"}.
    - Può aggiornare anche il path legacy periodo (delegato a `update_period()`).
    """
    meta = {"tz": period.tz, "t0": str(period.t0), "t1": str(period.t1)}
    update_period(SCENARIO_JSON, meta, legacy_period_path=LEGACY_PERIOD_JSON)


def infer_period_if_needed(members: List[dict]) -> Optional[PeriodConfig]:
    """Inferisce e persiste il periodo di simulazione se non già presente nello scenario.
    
    Strategia:
    - se `scenario.json` contiene già t0/t1: ritorna quel periodo;
    - altrimenti seleziona il primo membro (id crescente) che ha consumi disponibili;
    - calcola (t0, t1) tramite `infer_period_from_consumption()`;
    - costruisce `PeriodConfig` e lo salva nello scenario.
    
    Parameters
    ----------
    members : List[dict]
        Lista membri scenario.
    
    Returns
    -------
    Optional[PeriodConfig]
        Periodo già presente o inferito; None se nessun membro ha consumi.
    
    Side Effects
    ------------
    Aggiorna i metadati periodo in `scenario.json` (e legacy, se configurato).
    """
    current = load_period()
    if current is not None:
        return current

    # primo membro (id crescente) che ha consumi
    candidates = sorted([int(m["id"]) for m in members if has_consumption(int(m["id"]))])
    if not candidates:
        return None

    ref_id = candidates[0]
    t0, t1 = infer_period_from_consumption(consumption_path(ref_id), tz="UTC")
    period = build_period_config(t0, t1, tz="UTC")
    save_period(period)
    return period


# =============================================================================
# Streamlit state
# =============================================================================


# Carica e migra i membri SOLO al primo run della pagina (evita overwrite su rerun)
if "members" not in st.session_state:
    _raw_members = load_members()
    _migrated_members = migrate_members_schema(_raw_members)
    st.session_state["members"] = _migrated_members
    # Persisti la migrazione schema una sola volta (se necessario)
    if _raw_members != _migrated_members:
        save_members(_migrated_members)

st.session_state.setdefault("members_dirty", False)
st.session_state.setdefault("last_run", None)


# =============================================================================
# Run manager (lista, carica, rinomina, elimina)
# =============================================================================


def _read_time_indexed_csv(path: Path) -> pd.DataFrame:
    """Legge un CSV con timestamp in prima colonna (index) e normalizza in UTC.
    
    Il CSV è letto con `index_col=0`. L'indice viene convertito con
    `pd.to_datetime(..., utc=True, errors="raise")` e rinominato in "time".
    
    Parameters
    ----------
    path : pathlib.Path
        Path del CSV da leggere.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame indicizzato da DatetimeIndex UTC.
    
    Raises
    ------
    Exception
        Se il parsing del timestamp fallisce (errors="raise").
    """
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=True, errors="raise")
    df.index = pd.DatetimeIndex(idx)
    df.index.name = "time"
    return df


def load_energy_run_for_ui(run_dir: Path, label: Optional[str] = None) -> dict:
    """Carica dal filesystem i risultati di una run energetica per la UI.
    
    Attende nella directory `run_dir` i file:
    - cer_hourly.csv
    - members_hourly_long.csv
    - members_summary.csv
    - cer_summary.csv
    - period_meta.csv (opzionale ai fini UI)
    
    `members_hourly_long.csv` viene trasformato in un dizionario
    `{member_id -> DataFrame orario}`, rimuovendo la colonna `member_id`.
    
    Parameters
    ----------
    run_dir : pathlib.Path
        Directory run (tipicamente outputs/run_<run_id>).
    label : Optional[str]
        Label descrittiva della run (se disponibile).
    
    Returns
    -------
    dict
        Struttura dati per la UI con risultati CER e risultati per membro.
    
    Notes
    -----
    L'indice temporale viene normalizzato in UTC.
    """
    p_cer = run_dir / "cer_hourly.csv"
    p_mlong = run_dir / "members_hourly_long.csv"
    p_msum = run_dir / "members_summary.csv"
    p_csum = run_dir / "cer_summary.csv"
    p_meta = run_dir / "period_meta.csv"

    cer_hourly = _read_time_indexed_csv(p_cer)
    members_long = _read_time_indexed_csv(p_mlong)

    members_hourly: Dict[str, pd.DataFrame] = {}
    if "member_id" in members_long.columns:
        for mid, df in members_long.groupby("member_id"):
            tmp = df.drop(columns=["member_id"]).copy()
            tmp.index = pd.DatetimeIndex(pd.to_datetime(tmp.index, utc=True))
            members_hourly[str(mid)] = tmp.sort_index()

    members_summary = pd.read_csv(p_msum) if p_msum.exists() else pd.DataFrame()
    cer_summary = pd.read_csv(p_csum) if p_csum.exists() else pd.DataFrame()

    out_paths = {
        "cer_hourly": p_cer,
        "members_hourly_long": p_mlong,
        "members_summary": p_msum,
        "cer_summary": p_csum,
        "period_meta": p_meta,
    }

    # run_id dal folder name (run_<id>)
    run_id = run_dir.name.replace("run_", "", 1) if run_dir.name.startswith("run_") else run_dir.name

    return {
        "run_id": run_id,
        "label": label,
        "out_dir": str(run_dir),
        "paths": {k: str(v) for k, v in out_paths.items()},
        "cer_hourly": cer_hourly,
        "cer_summary": cer_summary,
        "members_hourly": members_hourly,
        "members_summary": members_summary,
    }



# =============================================================================
# UI (nuova): Scenario -> input modificabile; "Salva modifiche" versiona e ricalcola
# =============================================================================

st.subheader("Input simulazione")

# --- Scenario selector ---
scenarios = _list_scenarios()
active_name = _normalize_scenario_name(st.session_state.get("active_scenario", "default"))
if active_name not in scenarios:
    scenarios = [active_name] + [s for s in scenarios if s != active_name]

col_left, col_right = st.columns([1.1, 1.9])

with col_left:
    picked = st.selectbox("Scenario", options=scenarios, index=scenarios.index(active_name) if active_name in scenarios else 0)
    picked_norm = _normalize_scenario_name(picked)
    if picked_norm != active_name:
        st.session_state["active_scenario"] = picked_norm
        st.rerun()

    with st.expander("➕ Nuovo scenario"):
        new_sc = st.text_input("Nome scenario", key="new_scenario_name")
        if st.button("Crea scenario", use_container_width=True, key="btn_create_scenario"):
            nm = _normalize_scenario_name(new_sc)
            sp = _scenario_paths(nm)
            if not sp["scenario_json"].exists():
                _ = migrate_legacy_to_scenario(sp["scenario_json"], LEGACY_MEMBERS_JSON, LEGACY_PERIOD_JSON)
            st.session_state["active_scenario"] = nm
            st.success(f"Scenario creato: {nm}")
            st.rerun()

    # --- Run selector (tutte le versioni) ---
    run_dirs = sorted([d for d in OUTPUTS_DIR.glob('run_*') if d.is_dir()], reverse=True)
    run_ids = [d.name.replace('run_', '', 1) for d in run_dirs]
    if not run_ids:
        st.info("Nessuna run/versione nello scenario selezionato.")
        selected_run_id = None
        selected_run_dir = None
    else:
        active_rid = get_active_energy_run(ACTIVE_RUN_TXT)
        default_idx = 0
        if active_rid and active_rid in run_ids:
            default_idx = run_ids.index(active_rid)
        selected_run_id = st.selectbox("Seleziona run (versione)", options=run_ids, index=default_idx, key="sel_run_id")
        selected_run_dir = run_dirs[run_ids.index(selected_run_id)]

    # --- Actions ---
    btn_cols = st.columns(2)
    if btn_cols[0].button("Carica run", use_container_width=True, disabled=selected_run_dir is None):
        # 1) carica risultati
        last = load_energy_run_for_ui(selected_run_dir, label=None)
        st.session_state["last_run"] = last
        set_active_energy_run(ACTIVE_RUN_TXT, str(last["run_id"]))

        # 2) carica configurazione membri/periodo e la porta nello scenario (workspace)
        cfg_path = selected_run_dir / "run_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            except Exception:
                cfg = None
            if isinstance(cfg, dict):
                # periodo
                pmeta = cfg.get('period') or {}
                if isinstance(pmeta, dict) and pmeta.get('t0') and pmeta.get('t1'):
                    try:
                        tz = pmeta.get('tz', 'UTC')
                        t0 = str(pmeta['t0'])
                        t1 = str(pmeta['t1'])
                        update_period(SCENARIO_JSON, {"tz": tz, "t0": t0, "t1": t1}, legacy_period_path=LEGACY_PERIOD_JSON)
                    except Exception:
                        pass

                # membri
                mlist = cfg.get('members') or []
                members = []
                if isinstance(mlist, list):
                    for m in mlist:
                        if not isinstance(m, dict):
                            continue
                        mid = int(m.get('id'))
                        mm = default_member(mid)
                        mm['name'] = m.get('name', mm['name'])
                        mm['role'] = m.get('role', mm['role'])
                        mm['num'] = int(m.get('num', mm['num']) or 1)
                        mm['voltage_level'] = m.get('voltage_level', mm['voltage_level'])
                        mm['new_plant'] = bool(m.get('new_plant', mm['new_plant']))
                        mm['commissioning_month'] = m.get('commissioning_month', mm['commissioning_month'])
                        mm['installed_capacity_kw'] = float(m.get('installed_capacity_kw', mm['installed_capacity_kw']) or 0.0)
                        mm['prod_mode'] = m.get('production_mode', mm.get('prod_mode', 'totale'))
                        mm['selected_areas'] = list(m.get('selected_areas') or [])
                        mm['has_battery'] = bool(m.get('has_battery', False))
                        batt = m.get('battery') or {}
                        if mm['has_battery'] and isinstance(batt, dict):
                            mm['battery_capacity_kwh'] = float(batt.get('capacity_kwh', 0.0) or 0.0)
                            mm['battery_dod'] = float(batt.get('dod', 0.8) or 0.8)
                            mm['battery_roundtrip_eff'] = float(batt.get('roundtrip_eff', 0.9) or 0.9)
                            mm['battery_derating_factor'] = float(batt.get('derating_factor', 0.0) or 0.0)
                            mm['battery_init_soc_perc'] = float(batt.get('init_soc_perc', 0.2) or 0.2)
                        members.append(mm)

                members = migrate_members_schema(members)
                update_members(SCENARIO_JSON, members, legacy_members_path=LEGACY_MEMBERS_JSON)
                st.session_state["members"] = members
                st.session_state["members_dirty"] = False

        st.success(f"Run caricata: {selected_run_id}")
        st.rerun()

    if btn_cols[1].button("Salva modifiche", use_container_width=True, type="primary"):
        # Salva scenario (membri) + ricalcola e crea una nuova versione
        save_members(st.session_state["members"])

        period = infer_period_if_needed(st.session_state["members"])
        if period is None:
            st.error("Carica almeno un file consumi per definire il periodo di simulazione.")
            st.stop()

        errs, warns = validate_registry(st.session_state["members"], period)
        if warns:
            st.warning("Warning:\n- " + "\n- ".join(warns))
        if errs:
            st.error("Errori bloccanti:\n- " + "\n- ".join(errs))
            st.stop()

        run_id = pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')
        run_dir = OUTPUTS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build MemberSpec list
        members_specs = []
        member_num = {}
        for m in st.session_state["members"]:
            mid = int(m["id"])
            mid_key = str(mid)
            num = int(m.get("num", 1) or 1)
            member_num[mid_key] = max(1, num)

            prod_file = production_path(mid) if has_production(mid) else None
            ps = ProductionSpec(
                enabled=prod_file is not None,
                mode=m.get("prod_mode", "totale"),
                selected_areas=tuple(m.get("selected_areas", []) or []),
            )

            batt = None
            if bool(m.get("has_battery", False)):
                batt = BatterySpec(
                    capacity_kwh=float(m.get("battery_capacity_kwh", 0.0) or 0.0),
                    dod=float(m.get("battery_dod", 0.8)),
                    roundtrip_eff=float(m.get("battery_roundtrip_eff", 0.9)),
                    derating_factor=float(m.get("battery_derating_factor", 0.0)),
                    init_soc_perc=float(m.get("battery_init_soc_perc", 0.2)),
                )

            members_specs.append(
                MemberSpec(
                    member_id=mid_key,
                    name=m.get("name", f"Membro {mid}"),
                    consumption_csv=consumption_path(mid),
                    production_csv=prod_file,
                    production_spec=ps,
                    battery=batt,
                )
            )

        # Compute member flows
        members_hourly = {}
        member_rows = []
        for spec in members_specs:
            data = load_and_validate_member(spec, period)
            df_member = compute_member_energy_hourly(
                P_load_15min_kW=data["P_load_15min_kW"],
                P_prod_hourly_kW=data["P_prod_hourly_kW"],
                battery=spec.battery,
            )
            n = int(member_num.get(spec.member_id, 1))
            if n != 1:
                df_member = df_member.copy()
                for c in df_member.columns:
                    if c == "SOC_perc":
                        continue
                    df_member[c] = df_member[c] * float(n)

            members_hourly[spec.member_id] = df_member
            member_rows.append({"member_id": spec.member_id, "name": spec.name, "num": n, **summarize_member(df_member)})

        members_summary = pd.DataFrame(member_rows).sort_values('member_id')
        cer_hourly = compute_cer_hourly(members_hourly)
        cer_summary = pd.DataFrame([summarize_cer(cer_hourly, members_hourly)])

        out_paths = save_outputs(
            out_dir=run_dir,
            period=period,
            members_hourly=members_hourly,
            cer_hourly=cer_hourly,
            members_summary=members_summary,
            cer_summary=cer_summary,
        )

        run_cfg = {
            "period": {"tz": period.tz, "t0": str(period.t0), "t1": str(period.t1)},
            "members": [
                {
                    "id": int(m["id"]),
                    "name": m.get("name", f"Membro {m['id']}"),
                    "role": m.get("role", "prosumer"),
                    "num": int(m.get("num", 1)),
                    "voltage_level": m.get("voltage_level", "BT"),
                    "new_plant": bool(m.get("new_plant", True)),
                    "commissioning_month": m.get("commissioning_month", ""),
                    "installed_capacity_kw": float(m.get("installed_capacity_kw", 0.0) or 0.0),
                    "has_consumption_file": has_consumption(int(m["id"])),
                    "has_production_file": has_production(int(m["id"])),
                    "production_mode": m.get("prod_mode", "totale"),
                    "selected_areas": list(m.get("selected_areas", []) or []),
                    "has_battery": bool(m.get("has_battery", False)),
                    "battery": None if not bool(m.get("has_battery", False)) else {
                        "capacity_kwh": float(m.get("battery_capacity_kwh", 0.0) or 0.0),
                        "dod": float(m.get("battery_dod", 0.8)),
                        "roundtrip_eff": float(m.get("battery_roundtrip_eff", 0.9)),
                        "derating_factor": float(m.get("battery_derating_factor", 0.0)),
                        "init_soc_perc": float(m.get("battery_init_soc_perc", 0.2)),
                    },
                }
                for m in st.session_state["members"]
            ],
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding='utf-8')

        # label automatico (scenario + run_id)
        auto_label = f"{ACTIVE_SCENARIO} - {run_id}"
        write_energy_run_meta(run_dir, {"run_id": run_id, "label": auto_label, "created_at_utc": pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')})

        try:
            run_dir_rel = str(run_dir.relative_to(SESSION_DIR))
        except Exception:
            run_dir_rel = str(run_dir)

        kpi = {}
        try:
            if cer_summary is not None and len(cer_summary) > 0:
                kpi = {k: (float(v) if v is not None and str(v) != 'nan' else v) for k, v in cer_summary.iloc[0].to_dict().items()}
        except Exception:
            kpi = {}

        rec = {
            "run_id": run_id,
            "label": auto_label,
            "created_at_utc": pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "run_dir": run_dir_rel,
            "period": {"tz": period.tz, "t0": str(period.t0), "t1": str(period.t1)},
            "members_n": int(len(members_specs)),
            "kpi": kpi,
        }
        append_energy_run_record(RUNS_INDEX_JSONL, rec)
        set_active_energy_run(ACTIVE_RUN_TXT, run_id)

        st.session_state["last_run"] = {
            "run_id": run_id,
            "label": auto_label,
            "out_dir": str(run_dir),
            "paths": out_paths,
            "cer_hourly": cer_hourly,
            "cer_summary": cer_summary,
            "members_hourly": members_hourly,
            "members_summary": members_summary,
        }

        st.success("Modifiche salvate e run ricalcolata.")
        st.rerun()

with col_right:
    st.markdown("#### Informazioni run")
    last = st.session_state.get("last_run")
    rid = (last or {}).get("run_id") or get_active_energy_run(ACTIVE_RUN_TXT) or "—"
    st.text_input("Run ID", value=str(rid), disabled=True)

    # KPI sintetici Ep/Ec (se disponibili)
    ep = None
    ec = None
    mn = len(st.session_state.get('members') or [])
    if isinstance(last, dict) and isinstance(last.get('cer_summary'), pd.DataFrame) and not last['cer_summary'].empty:
        row = last['cer_summary'].iloc[0].to_dict()
        ep = row.get('E_prod_tot_kWh')
        ec = row.get('E_load_tot_kWh')

    mcols = st.columns(3)
    mcols[0].metric("Energia prodotta (kWh)", "—" if ep is None else f"{float(ep):,.0f}")
    mcols[1].metric("Energia consumata (kWh)", "—" if ec is None else f"{float(ec):,.0f}")
    mcols[2].metric("Numero membri", f"{mn}")


# =============================================================================
# Membri CER (lista modificabile) + add/delete
# =============================================================================

st.subheader("Membri CER")

members = st.session_state.get("members") or []

# Tabella modificabile
registry_df = members_to_registry_df(members)
edited_df = st.data_editor(
    registry_df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "role": st.column_config.SelectboxColumn("Ruolo", options=["consumer", "producer", "prosumer"]),
        "voltage_level": st.column_config.SelectboxColumn("Tensione", options=["BT", "MT"]),
        "prod_mode": st.column_config.SelectboxColumn("Produzione mode", options=["totale", "aree"]),
        "has_battery": st.column_config.CheckboxColumn("Batteria"),
        "new_plant": st.column_config.CheckboxColumn("Nuovo impianto (TIP)"),
    },
    key="members_registry_editor_v2",
)

if not edited_df.equals(registry_df):
    st.session_state["members"] = registry_df_to_members(edited_df, members)
    st.session_state["members_dirty"] = True
    members = st.session_state["members"]

# Stato input (read-only)
st.markdown("#### Stato input")
rows = []
for m in members:
    mid = int(m["id"])
    rows.append({
        "member_id": mid,
        "nome": m.get("name", f"Membro {mid}"),
        "consumi": "✅" if has_consumption(mid) else "—",
        "produzione": "✅" if has_production(mid) else "—",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

btn_cols = st.columns([1, 1, 2])
if btn_cols[0].button("Aggiungi membro", use_container_width=True):
    st.session_state["add_member_open"] = True

# eliminazione con selettore separato
ids = sorted([int(m["id"]) for m in members])
sel_del = btn_cols[1].selectbox("member_id", options=ids if ids else ["—"], key="sel_member_delete")
if btn_cols[1].button("Elimina membro", use_container_width=True, disabled=(not ids)):
    pick = int(sel_del)
    members = [m for m in members if int(m.get("id")) != pick]
    st.session_state["members"] = members
    st.session_state["members_dirty"] = True
    # elimina folder input del membro
    d = INPUTS_DIR / f"member_{pick}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    st.success(f"Eliminato membro {pick}")
    st.rerun()


with st.expander("Informazioni membro", expanded=bool(st.session_state.get("add_member_open", False))):
    mode = st.radio("Modalità", options=["Nuovo membro", "Aggiorna file membro esistente"], horizontal=True)

    if mode == "Nuovo membro":
        new_name = st.text_input("Nome membro", key="new_member_name")
        up_load = st.file_uploader("CSV consumi (15 min) - richiesto", type=["csv"], key="new_member_load")
        up_prod = st.file_uploader("CSV produzione (1h) - opzionale", type=["csv"], key="new_member_prod")

        if st.button("Crea membro", type="primary"):
            new_id = next_member_id(members)
            mm = default_member(new_id)
            mm["name"] = (new_name or "").strip() or f"Membro {new_id}"
            members.append(mm)
            st.session_state["members"] = members
            st.session_state["members_dirty"] = True

            if up_load is None:
                st.error("Per creare un membro serve il file consumi.")
                st.stop()

            save_uploaded_file(up_load, consumption_path(new_id))
            if up_prod is not None:
                save_uploaded_file(up_prod, production_path(new_id))

            save_members(members)
            _ = infer_period_if_needed(members)
            st.session_state["add_member_open"] = False
            st.success(f"Creato membro ID {new_id} e caricati i file.")
            st.rerun()

    else:
        if not ids:
            st.info("Nessun membro disponibile.")
        else:
            pick = st.selectbox("Seleziona member_id", options=ids, key="pick_member_upload")
            up_load = st.file_uploader("Sostituisci CSV consumi (15 min)", type=["csv"], key="upd_member_load")
            up_prod = st.file_uploader("Sostituisci CSV produzione (1h)", type=["csv"], key="upd_member_prod")

            if st.button("Salva file", type="primary"):
                if up_load is not None:
                    save_uploaded_file(up_load, consumption_path(int(pick)))
                if up_prod is not None:
                    save_uploaded_file(up_prod, production_path(int(pick)))
                _ = infer_period_if_needed(members)
                st.success("File aggiornati.")
                st.rerun()

st.divider()


# =============================================================================
# 4) Risultati (come prima)
# =============================================================================


last = st.session_state.get("last_run")
if last is not None:
    st.divider()
    st.subheader("Risultati")

    _lab = str(last.get("label") or "").strip()
    if _lab:
        st.write(f"Run: **{_lab}** (id `{last['run_id']}`) — output: `{last['out_dir']}`")
    else:
        st.write(f"Run: `{last['run_id']}` — output: `{last['out_dir']}`")

    st.markdown("### KPI CER")
    st.dataframe(last["cer_summary"], use_container_width=True)

    st.markdown("### KPI membri")
    st.dataframe(last["members_summary"], use_container_width=True)

    # Grafici CER
    cer_hourly: pd.DataFrame = last.get("cer_hourly")
    if isinstance(cer_hourly, pd.DataFrame) and not cer_hourly.empty:
        plot_mode = st.selectbox("Risoluzione grafici CER", options=["hourly", "daily", "monthly"], index=1)
        if plot_mode == "daily":
            cer_plot = cer_hourly.resample("D").sum()
        elif plot_mode == "monthly":
            cer_plot = cer_hourly.resample("MS").sum()
        else:
            cer_plot = cer_hourly

        st.markdown("### Energia condivisa")
        fig1 = plot_timeseries(cer_plot, y_cols=["E_cond_kWh"], height=320)
        st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("### Immissione vs Prelievo")
        fig2 = plot_timeseries(cer_plot, y_cols=["E_imm_CER_kWh", "E_prel_CER_kWh"], height=320)
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

    # Download
    st.markdown("### Download risultati")
    items = [
        ("CER orario", "cer_hourly"),
        ("Membri orari (long)", "members_hourly_long"),
        ("Summary membri", "members_summary"),
        ("Summary CER", "cer_summary"),
        ("Metadata periodo", "period_meta"),
    ]
    cols = st.columns(len(items))
    for (label, key), col in zip(items, cols):
        p = Path(last["paths"].get(key, ""))
        if p.exists():
            with col:
                st.download_button(
                    label=label,
                    data=p.read_bytes(),
                    file_name=p.name,
                    mime="text/csv",
                    use_container_width=True,
                )

    # Dettagli membro (come richiesto): selettore + 3 mini-sezioni (rete/autoconsumo/batteria)
    st.markdown("### Dettagli membro")
    _mh = last.get("members_hourly")
    if not isinstance(_mh, dict) or not _mh:
        st.info("Nessun dettaglio membro disponibile per questa run.")
    else:
        def _sort_mid(x: str):
            try:
                return (0, int(str(x)))
            except Exception:
                return (1, str(x))

        member_ids = sorted([str(k) for k in _mh.keys()], key=_sort_mid)
        sel_mid = st.selectbox("Seleziona membro", options=member_ids, key="sel_detail_member")
        dfm = _mh.get(sel_mid)

        if not isinstance(dfm, pd.DataFrame) or dfm.empty:
            st.warning("Dati orari del membro non disponibili.")
        else:
            tab_rete, tab_auto, tab_batt = st.tabs(["Rete", "Autoconsumo", "Batteria"])

            with tab_rete:
                cols = [c for c in ["E_prel_kWh", "E_imm_kWh"] if c in dfm.columns]
                if not cols:
                    st.info("Colonne rete non disponibili per questo membro.")
                else:
                    st.plotly_chart(
                        plot_timeseries(dfm, y_cols=cols, height=300),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

            with tab_auto:
                cols = [c for c in ["E_aut_kWh", "E_aut_PV_kWh", "E_aut_batt_kWh"] if c in dfm.columns]
                # fallback se non esistono i dettagli PV/batt
                if not cols and "E_aut_kWh" in dfm.columns:
                    cols = ["E_aut_kWh"]
                if not cols:
                    st.info("Colonne autoconsumo non disponibili per questo membro.")
                else:
                    st.plotly_chart(
                        plot_timeseries(dfm, y_cols=cols, height=300),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

            with tab_batt:
                cols_energy = [c for c in ["E_batt_charge_kWh", "E_batt_discharge_kWh", "E_batt_loss_kWh"] if c in dfm.columns]
                cols_soc = [c for c in ["SOC_perc"] if c in dfm.columns]

                if not cols_energy and not cols_soc:
                    st.info("Nessun dato batteria per questo membro (batteria assente o non modellata).")
                else:
                    if cols_energy:
                        st.plotly_chart(
                            plot_timeseries(dfm, y_cols=cols_energy, height=260),
                            use_container_width=True,
                            config=PLOTLY_CONFIG,
                        )
                    if cols_soc:
                        st.plotly_chart(
                            plot_timeseries(dfm, y_cols=cols_soc, height=260),
                            use_container_width=True,
                            config=PLOTLY_CONFIG,
                        )
