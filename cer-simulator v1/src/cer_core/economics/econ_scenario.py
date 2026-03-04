"""cer_core.economics.econ_scenario

Persistenza su filesystem degli *scenari economici* utilizzati dal simulatore di
Comunità Energetiche Rinnovabili (CER).

Questo modulo implementa le operazioni di:

* **salvataggio** di un insieme di assunzioni economiche
  (:class:`~cer_core.economics.economic_model.EconomicsAssumptions`) in una
  directory di scenario composta da file CSV e un manifest JSON;
* **caricamento** delle assunzioni da directory verso oggetti di dominio;
* **listing** degli scenari disponibili e gestione di un puntatore "active".

Struttura su disco
------------------

Ogni scenario è una directory sotto ``scenarios_root/<slug>/`` con:

* ``manifest.json``: metadati (versione schema, timestamp UTC, fingerprint del
  contenuto, elenco file);
* file tabellari CSV (es. ``users.csv``, ``tariffs_buy.csv``);
* profili temporali CSV opzionali (es. ``pzo_profile.csv``) salvati con indice
  denominato ``time``.

Input/Output e convenzioni
--------------------------

* **Formato CSV**: separatore predefinito pandas (virgola), intestazione in prima
  riga, senza colonna indice per le *tabelle*; per i *profili* l'indice temporale
  viene serializzato in una colonna ``time``.
* **Timestamp**: i profili sono letti con ``parse_dates=["time"]`` e risultano
  tipicamente *naive* (senza timezone) salvo che il CSV contenga informazioni di
  timezone. Il significato (UTC vs locale) è demandato al layer di business.
* **Fingerprint**: l'hash ``content_fingerprint_sha256`` è calcolato sui file di
  contenuto (CSV) escludendo il ``manifest.json`` per evitare dipendenze da
  metadati (timestamp/ordinamento JSON).

Side effects
------------

* Creazione directory e scrittura/riscrittura di file CSV/JSON.
* Se ``overwrite=False`` e lo slug esiste già, viene creata automaticamente una
  variante suffissata con timestamp UTC.

Nota di compatibilità
---------------------

I nomi file e la struttura delle directory sono parte del *contratto* con la UI
Streamlit (pagina "Valutazione economica"); modifiche a questi elementi
impatterebbero direttamente il caricamento/salvataggio degli scenari.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from cer_core.economics.economic_model import EconomicsAssumptions
from cer_core.bilanciamento.fingerprint import sha256_dir_files


SCHEMA_VERSION = 1


def econ_scenario_content_fingerprint_sha256(scenario_dir: Path) -> str:
    """Calcola un fingerprint (SHA-256) stabile del contenuto dello scenario.

    Il fingerprint è calcolato ricorsivamente sui file *di contenuto* della
    directory di scenario e **esclude** intenzionalmente ``manifest.json``.

    Parameters
    ----------
    scenario_dir:
        Directory che contiene i file di scenario.

    Returns
    -------
    str
        Hash SHA-256 in formato esadecimale.
    """
    return sha256_dir_files(scenario_dir, exclude_names={"manifest.json"})


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "scenario"
    s = name.lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_-")
    return s or "scenario"


@dataclass(frozen=True)
class EconScenarioInfo:
    """Metadati minimi di uno scenario economico salvato.

    Attributes
    ----------
    scenario_dir:
        Directory di scenario su filesystem.
    name:
        Nome leggibile (eventualmente derivato dal manifest).
    created_at_utc, updated_at_utc:
        Timestamp ISO-8601 in UTC (suffisso ``Z``) se presenti nel manifest.
    """

    scenario_dir: Path
    name: str
    created_at_utc: str
    updated_at_utc: str


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Scrive un file di testo in modo atomico.

    La scrittura avviene su un file temporaneo nella stessa directory e viene
    poi eseguita una ``os.replace`` (operazione atomica sui filesystem POSIX).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        # In caso di eccezione prima di os.replace
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _write_table_csv(path: Path, df: pd.DataFrame) -> None:
    """Serializza una tabella su CSV (senza indice) con scrittura atomica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_table_csv(path: Path) -> pd.DataFrame:
    """Legge una tabella CSV prodotta da :func:`_write_table_csv`."""
    return pd.read_csv(path)


def _write_profile_csv(path: Path, df: pd.DataFrame) -> None:
    """Serializza un profilo temporale su CSV con colonna indice ``time``.

    Note
    ----
    Il profilo viene salvato preservando l'indice in una colonna ``time``.
    L'informazione di timezone è preservata solo se presente nella
    rappresentazione testuale dei timestamp.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()

    # Normalizza il profilo: preferiamo salvare sempre il timestamp come indice
    # chiamato "time" per evitare CSV con colonne duplicate (es. df con colonna
    # "time" + index_label="time"). Questo caso rompe il load perché pandas
    # rinomina la seconda colonna in "time.1".
    try:
        if out.index.name != "time":
            if "time" in out.columns:
                out["time"] = pd.to_datetime(out["time"], utc=True)
                out = out.set_index("time")
            elif "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
                out = out.set_index("timestamp")
                out.index.name = "time"
            else:
                out.index = pd.to_datetime(out.index, utc=True)
                out.index.name = "time"
        else:
            out.index = pd.to_datetime(out.index, utc=True)
            out.index.name = "time"
    except Exception:
        # fallback: salva comunque, ma prova a mantenere l'indice con label "time"
        out.index.name = "time"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.close(fd)
        # Ensure time index is saved explicitly
        out.to_csv(tmp, index=True, index_label="time")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_profile_csv(path: Path) -> pd.DataFrame:
    """Legge un profilo temporale CSV con colonna ``time`` come indice.

    Robustezza / backward-compat
    ----------------------------
    Versioni precedenti potevano salvare profili con una colonna dati "time"
    e contemporaneamente serializzare anche l'indice con ``index_label='time'``.
    In CSV questo genera intestazioni duplicate e pandas le disambigua ("time.1").
    Qui tentiamo di recuperare automaticamente il timestamp corretto.
    """

    raw = pd.read_csv(path)

    # Se esiste una colonna disambiguata (time.1) ed il "time" principale è un
    # semplice contatore, preferiamo time.1.
    time_col = None
    if "time.1" in raw.columns and "time" in raw.columns:
        try:
            s0 = raw["time"]
            is_counter = pd.to_numeric(s0, errors="coerce").notna().all()
            if is_counter:
                t1 = pd.to_datetime(raw["time.1"], errors="coerce", utc=True)
                if t1.notna().mean() > 0.9:
                    raw["time"] = t1
        except Exception:
            pass

    # normal path: parse 'time'
    if "time" in raw.columns:
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce", utc=True)
        time_col = "time"
    elif "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
        time_col = "timestamp"
    else:
        # fallback: primo campo datetime-like
        for c in raw.columns:
            t = pd.to_datetime(raw[c], errors="coerce", utc=True)
            if t.notna().mean() > 0.9:
                raw[c] = t
                time_col = c
                break

    if time_col is None:
        # ultimo fallback: indice implicito
        raw.index = pd.to_datetime(raw.index, errors="coerce", utc=True)
        raw.index.name = "time"
        return raw

    df = raw.set_index(time_col)
    df.index.name = "time"

    # cleanup: se esiste una colonna che replica l'indice temporale, rimuovila
    for c in ("time.1", "timestamp.1"):
        if c in df.columns:
            try:
                t = pd.to_datetime(df[c], errors="coerce", utc=True)
                if t.notna().mean() > 0.9 and (t.values == df.index.values).all():
                    df = df.drop(columns=[c])
            except Exception:
                pass
    return df


# -----------------------------------------------------------------------------
# Scenario save/load
# -----------------------------------------------------------------------------


_TABLE_FILES = {
    "policy_cer": "policy_cer.csv",
    "users": "users.csv",
    "tariffs_buy": "tariffs_buy.csv",
    "tariffs_sell": "tariffs_sell.csv",
    "assets_pv": "assets_pv.csv",
    "assets_wind": "assets_wind.csv",
    "assets_bess": "assets_bess.csv",
    "tax_by_class": "tax_by_class.csv",
    "tax_overrides": "tax_overrides.csv",
    "dcf_params": "dcf_params.csv",
}

_PROFILE_FILES = {
    "pzo_profile": "pzo_profile.csv",
    "tip_profile": "tip_profile.csv",
    "tiad_profile": "tiad_profile.csv",
    "buy_profiles": "buy_profiles.csv",
    "sell_profiles": "sell_profiles.csv",
}


def save_econ_scenario(
    scenarios_root: Path,
    name: str,
    assumptions: EconomicsAssumptions,
    *,
    overwrite: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Salva un pacchetto di assunzioni economiche su disco.

    La funzione materializza un oggetto :class:`~cer_core.economics.economic_model.EconomicsAssumptions`
    in una directory di scenario, secondo il layout atteso dalla UI.

    Parameters
    ----------
    scenarios_root:
        Directory radice che contiene gli scenari economici.
    name:
        Nome umano dello scenario; viene slugificato per costruire il path.
    assumptions:
        Struttura dati con tabelle e profili del modello economico.
    overwrite:
        Se ``True`` (default) sovrascrive lo slug esistente; se ``False`` crea
        automaticamente una variante con suffisso timestamp UTC.
    extra_meta:
        Dizionario opzionale serializzato sotto la chiave ``meta`` del manifest.

    Returns
    -------
    pathlib.Path
        Percorso della directory scenario creata/aggiornata.

    Side Effects
    ------------
    * Creazione directory.
    * Scrittura di molteplici CSV e del file ``manifest.json`` (scritture atomiche).
    """
    scenarios_root.mkdir(parents=True, exist_ok=True)
    slug = _slugify(name)
    scen_dir = scenarios_root / slug

    if scen_dir.exists() and not overwrite:
        # crea una variante con timestamp
        slug = f"{slug}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        scen_dir = scenarios_root / slug

    scen_dir.mkdir(parents=True, exist_ok=True)

    # tables
    _write_table_csv(scen_dir / _TABLE_FILES["policy_cer"], assumptions.policy_cer)
    _write_table_csv(scen_dir / _TABLE_FILES["users"], assumptions.users)
    _write_table_csv(scen_dir / _TABLE_FILES["tariffs_buy"], assumptions.tariffs_buy)
    _write_table_csv(scen_dir / _TABLE_FILES["tariffs_sell"], assumptions.tariffs_sell)
    _write_table_csv(scen_dir / _TABLE_FILES["assets_pv"], assumptions.assets_pv)
    # wind (opzionale): se non presente, elimina eventuale file legacy
    p_wind = scen_dir / _TABLE_FILES["assets_wind"]
    if getattr(assumptions, "assets_wind", None) is not None:
        _write_table_csv(p_wind, assumptions.assets_wind)
    else:
        if p_wind.exists():
            try:
                p_wind.unlink()
            except OSError:
                pass
    _write_table_csv(scen_dir / _TABLE_FILES["assets_bess"], assumptions.assets_bess)
    _write_table_csv(scen_dir / _TABLE_FILES["tax_by_class"], assumptions.tax_by_class)
    # tax overrides (opzionale): se assente, elimina eventuale file legacy
    p_tax_over = scen_dir / _TABLE_FILES["tax_overrides"]
    if assumptions.tax_overrides is not None:
        _write_table_csv(p_tax_over, assumptions.tax_overrides)
    else:
        if p_tax_over.exists():
            try:
                p_tax_over.unlink()
            except OSError:
                pass
    _write_table_csv(scen_dir / _TABLE_FILES["dcf_params"], assumptions.dcf_params)

    # optional profiles: se assenti, elimina eventuali file pre-esistenti per coerenza in overwrite
    def _write_or_delete_profile(key: str, df_opt: Optional[pd.DataFrame]) -> None:
        p = scen_dir / _PROFILE_FILES[key]
        if df_opt is None or (isinstance(df_opt, pd.DataFrame) and df_opt.empty):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
            return
        _write_profile_csv(p, df_opt)

    _write_or_delete_profile("pzo_profile", assumptions.pzo_profile)
    _write_or_delete_profile("tip_profile", assumptions.tip_profile)
    _write_or_delete_profile("tiad_profile", assumptions.tiad_profile)
    _write_or_delete_profile("buy_profiles", assumptions.buy_profiles)
    _write_or_delete_profile("sell_profiles", assumptions.sell_profiles)

    # manifest
    manifest_path = scen_dir / "manifest.json"
    now = _now_utc_iso()
    content_fp = ""
    try:
        content_fp = econ_scenario_content_fingerprint_sha256(scen_dir)
    except Exception:
        content_fp = ""
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": str(name).strip() or slug,
        "slug": slug,
        "updated_at_utc": now,
        "created_at_utc": now,
        "content_fingerprint_sha256": content_fp,
        "files": {
            "tables": list(_TABLE_FILES.values()),
            "profiles": [fn for key, fn in _PROFILE_FILES.items() if (scen_dir / fn).exists()],
        },
    }

    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and old.get("created_at_utc"):
                manifest["created_at_utc"] = str(old.get("created_at_utc"))
        except Exception:
            pass

    if extra_meta:
        manifest["meta"] = extra_meta

    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return scen_dir


def load_econ_scenario(scenario_dir: Path) -> Tuple[EconomicsAssumptions, Dict[str, Any]]:
    """Carica un pacchetto di assunzioni economiche da disco.

    Parameters
    ----------
    scenario_dir:
        Directory dello scenario economico.

    Returns
    -------
    (EconomicsAssumptions, dict)
        La struttura dati delle assunzioni e il manifest/meta (se presente).

    Raises
    ------
    FileNotFoundError
        Se la directory non esiste oppure mancano file obbligatori.
    """
    if not scenario_dir.exists() or not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario economico non trovato: {scenario_dir}")

    manifest_path = scenario_dir / "manifest.json"
    meta: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            meta = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}

    # Annotazione non bloccante su eventuali mismatch di versione schema
    try:
        v = meta.get("schema_version")
        if isinstance(v, int) and v != SCHEMA_VERSION:
            meta.setdefault("_warnings", []).append(
                f"schema_version mismatch: expected {SCHEMA_VERSION}, found {v}"
            )
    except Exception:
        pass

    def must(path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Scenario incompleto: manca {label} ({path.name})")
        return _read_table_csv(path)

    policy = must(scenario_dir / _TABLE_FILES["policy_cer"], "policy_cer")
    users = must(scenario_dir / _TABLE_FILES["users"], "users")
    buy = must(scenario_dir / _TABLE_FILES["tariffs_buy"], "tariffs_buy")
    sell = must(scenario_dir / _TABLE_FILES["tariffs_sell"], "tariffs_sell")
    pv = must(scenario_dir / _TABLE_FILES["assets_pv"], "assets_pv")
    wind = None
    p_wind = scenario_dir / _TABLE_FILES.get("assets_wind", "assets_wind.csv")
    if p_wind.exists():
        wind = _read_table_csv(p_wind)
    bess = must(scenario_dir / _TABLE_FILES["assets_bess"], "assets_bess")
    tax_by_class = must(scenario_dir / _TABLE_FILES["tax_by_class"], "tax_by_class")
    dcf = must(scenario_dir / _TABLE_FILES["dcf_params"], "dcf_params")

    tax_overrides = None
    p_to = scenario_dir / _TABLE_FILES["tax_overrides"]
    if p_to.exists():
        tax_overrides = _read_table_csv(p_to)

    # optional profiles
    def opt_profile(key: str) -> Optional[pd.DataFrame]:
        p = scenario_dir / _PROFILE_FILES[key]
        return _read_profile_csv(p) if p.exists() else None

    pzo = opt_profile("pzo_profile")
    tip = opt_profile("tip_profile")
    tiad = opt_profile("tiad_profile")
    buy_prof = opt_profile("buy_profiles")
    sell_prof = opt_profile("sell_profiles")

    assumptions = EconomicsAssumptions(
        policy_cer=policy,
        users=users,
        tariffs_buy=buy,
        tariffs_sell=sell,
        assets_pv=pv,
        assets_wind=wind,
        assets_bess=bess,
        tax_by_class=tax_by_class,
        tax_overrides=tax_overrides,
        dcf_params=dcf,
        pzo_profile=pzo,
        tip_profile=tip,
        tiad_profile=tiad,
        buy_profiles=buy_prof,
        sell_profiles=sell_prof,
    )
    return assumptions, meta


def list_econ_scenarios(scenarios_root: Path) -> List[EconScenarioInfo]:
    """Elenca scenari economici salvati (ordinati per ``updated_at_utc`` desc)."""
    if not scenarios_root.exists():
        return []

    out: List[EconScenarioInfo] = []
    for d in scenarios_root.iterdir():
        if not d.is_dir():
            continue
        manifest = d / "manifest.json"
        name = d.name
        created = ""
        updated = ""
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8")) or {}
                name = str(m.get("name") or d.name)
                created = str(m.get("created_at_utc") or "")
                updated = str(m.get("updated_at_utc") or "")
            except Exception:
                pass
        out.append(EconScenarioInfo(scenario_dir=d, name=name, created_at_utc=created, updated_at_utc=updated))

    out.sort(key=lambda x: x.updated_at_utc or x.created_at_utc or "", reverse=True)
    return out


def set_active_econ_scenario(active_path: Path, scenario_slug: str) -> None:
    """Imposta lo slug dello scenario economico attivo.

    Il valore è salvato in chiaro nel file indicato da ``active_path``.
    """
    active_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(active_path, str(scenario_slug).strip(), encoding="utf-8")


def get_active_econ_scenario(active_path: Path) -> Optional[str]:
    """Ritorna lo slug dello scenario economico attivo, se presente."""
    if not active_path.exists():
        return None
    v = active_path.read_text(encoding="utf-8").strip()
    return v or None


def get_econ_scenario_content_fingerprint_sha256(scenario_dir: Path) -> Optional[str]:
    """Ritorna l'hash di contenuto dello scenario economico.

    - Preferisce il valore già salvato nel manifest (content_fingerprint_sha256)
    - In fallback calcola l'hash ricorsivo sui file CSV (escludendo manifest)
    """
    try:
        manifest_path = scenario_dir / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
            v = m.get("content_fingerprint_sha256")
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass

    try:
        return econ_scenario_content_fingerprint_sha256(scenario_dir)
    except Exception:
        return None
