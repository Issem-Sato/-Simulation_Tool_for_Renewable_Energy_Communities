from __future__ import annotations

"""
cer_core.bilanciamento.run_registry
==================================

Registry e metadati per la gestione dei **run** del simulatore CER.

Il modulo fornisce primitive di persistenza su filesystem per:
- **run energetici** (bilanciamento): indice append-only in formato JSON Lines
  (`runs_index.jsonl`), puntatore alla run attiva (`active_run.txt`) e metadata
  per run (`run_meta.json` dentro la cartella della run);
- **run economici**: indice JSON Lines analogo (utilizzato dalla parte economica).

Obiettivo progettuale
---------------------
- Append su JSONL per robustezza (una riga corrotta non invalida l'intero file).
- Funzioni piccole e idempotenti, pensate per essere chiamate dalla UI Streamlit.
- Side-effect limitati e riproducibili: nessun dato viene ricostruito in memoria
  se non a partire dai file su disco.

Formati e convenzioni
---------------------
- Registry run energetiche: **JSONL** (una run per riga, JSON object).
  Campi minimi usati dalla UI: ``run_id``, ``created_at_utc``, ``run_dir``.
  Altri campi (``period``, ``members_n``, ``kpi``, ``label``) possono essere presenti.
- ``run_dir`` può essere relativo a ``session_dir`` oppure assoluto.
- Label run: normalizzata con trim + collasso spazi; confronti case-insensitive.

Assunzioni e invarianti
-----------------------
- Il modulo non impone uno schema rigido del record JSONL: accetta e preserva
  campi extra (forward compatibility).
- Le operazioni di riscrittura (registry, active, meta) sono eseguite in modo
  **atomico** (scrittura su file temporaneo + replace) per ridurre il rischio di
  corruzione in caso di crash/interruzione.

Side effects
------------
- Scrittura/append su file JSONL.
- Creazione/riscrittura di ``active_*.txt``.
- Scrittura di ``run_meta.json`` nella cartella del run.
- Eliminazione di cartelle run (``shutil.rmtree``) nelle funzioni ``delete_*``.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# File name convention inside each energy run folder (bilanciamento/outputs/run_*/...)
RUN_META_FILENAME = "run_meta.json"



def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to *path* atomically.

    The content is first written to a temporary file in the same directory and
    then replaced via ``Path.replace``. This reduces the risk of truncated files
    if the process crashes during writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


# =============================================================================
# JSONL helpers (robust, tolleranti agli errori)
# =============================================================================


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a record to a JSON Lines file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _rewrite_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Rewrite a JSON Lines file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file; skip corrupted lines."""
    if not path.exists():
        return []

    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except Exception:
                # tolleranza: una riga corrotta non deve bloccare la UI
                logger.debug("Skipping corrupted JSONL line in %s", path)
                continue
    return out


def _set_active(path: Path, run_id: str) -> None:
    """Persist the active run id atomically."""
    _atomic_write_text(path, str(run_id).strip())


def _get_active(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    v = path.read_text(encoding="utf-8").strip()
    return v or None


# =============================================================================
# Energy run registry
# =============================================================================


@dataclass(frozen=True)
class EnergyRunRecord:
    """Record minimale di un run energetico.

    Salvato come JSON Lines (una riga per run) per append semplice e robusto.
    """

    run_id: str
    created_at_utc: str
    run_dir: str  # path relativo (consigliato) o assoluto

    period: Dict[str, Any]
    members_n: int
    kpi: Dict[str, Any]


def append_energy_run_record(index_path: Path, record: Dict[str, Any]) -> None:
    """Appende un record (dict) al registry JSONL delle run energetiche."""
    _append_jsonl(index_path, record)


def read_energy_run_registry(index_path: Path) -> List[Dict[str, Any]]:
    """Legge il registry JSONL delle run energetiche.

Linee corrotte vengono ignorate per tolleranza."""
    return _read_jsonl(index_path)


def set_active_energy_run(active_path: Path, run_id: str) -> None:
    """Imposta il run energetico attivo (scrittura atomica)."""
    _set_active(active_path, run_id)


def get_active_energy_run(active_path: Path) -> Optional[str]:
    """Ritorna il run energetico attivo, oppure None se non impostato."""
    return _get_active(active_path)


# =============================================================================
# Energy run metadata + management (label required)
# =============================================================================


def normalize_run_label(label: str) -> str:
    """Normalizza un nome run per confronti (trim + collassa spazi)."""
    return " ".join(str(label or "").strip().split())


def energy_run_meta_path(run_dir: Path) -> Path:
    """Return the path to the run metadata file within *run_dir*."""
    return run_dir / RUN_META_FILENAME


def write_energy_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    """Scrive il metadata file del run (run_meta.json).

    La scrittura è atomica per ridurre il rischio di file tronchi.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    p = energy_run_meta_path(run_dir)
    _atomic_write_text(p, json.dumps(meta, ensure_ascii=False, indent=2))


def read_energy_run_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Legge ``run_meta.json``.

    Returns
    -------
    Optional[Dict[str, Any]]
        Metadata del run, oppure None se il file è assente o non è JSON valido.
    """
    p = energy_run_meta_path(run_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Invalid run_meta.json in %s", run_dir)
        return None


def get_energy_run_label(run_dir: Path) -> Optional[str]:
    meta = read_energy_run_meta(run_dir)
    if not meta:
        return None
    v = meta.get("label")
    v = normalize_run_label(v)
    return v or None


def _to_abs_run_dir(session_dir: Path, run_dir_value: str) -> Path:
    p = Path(str(run_dir_value))
    if not p.is_absolute():
        p = session_dir / p
    return p


def list_energy_runs_indexed(session_dir: Path, index_path: Path) -> List[Dict[str, Any]]:
    """Ritorna la lista dei run energetici esistenti, arricchita con label e path assoluto.

    - Fonte primaria: registry JSONL (runs_index.jsonl)
    - Label: record['label'] -> run_meta.json -> fallback run_dir.name
    - Deduplica per label (case-insensitive) tenendo il run più recente.
    """
    recs = read_energy_run_registry(index_path)
    if not recs:
        return []

    recs_sorted = sorted(recs, key=lambda r: str(r.get("created_at_utc", "")), reverse=True)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for r in recs_sorted:
        run_id = str(r.get("run_id", "")).strip()
        run_dir_value = str(r.get("run_dir", "")).strip()
        if not run_id or not run_dir_value:
            continue

        run_dir_abs = _to_abs_run_dir(session_dir, run_dir_value)
        if not run_dir_abs.exists() or not run_dir_abs.is_dir():
            continue

        label = normalize_run_label(str(r.get("label", "")))
        if not label:
            label = get_energy_run_label(run_dir_abs) or ""
        if not label:
            label = run_dir_abs.name

        label_key = label.casefold()
        if label_key in seen:
            continue
        seen.add(label_key)

        rr = dict(r)
        rr["label"] = label
        rr["run_dir_abs"] = run_dir_abs
        out.append(rr)

    return out


def find_energy_run_by_label(session_dir: Path, index_path: Path, label: str) -> Optional[Dict[str, Any]]:
    """Trova il run (più recente) con una certa label (case-insensitive)."""
    target = normalize_run_label(label).casefold()
    if not target:
        return None
    for r in list_energy_runs_indexed(session_dir, index_path):
        if str(r.get("label", "")).casefold() == target:
            return r
    return None


def delete_energy_run(session_dir: Path, index_path: Path, run_id: str) -> bool:
    """Elimina una run energetica (cartella + riga/e dal registry).

    Ritorna True se ha trovato il run_id nel registry (e ha tentato l'eliminazione).
    """
    rid = str(run_id).strip()
    if not rid:
        return False

    recs = read_energy_run_registry(index_path)
    matches = [r for r in recs if str(r.get("run_id", "")).strip() == rid]
    if not matches:
        return False

    # prova a cancellare la directory (usa il primo match valido)
    run_dir_abs: Optional[Path] = None
    for r in matches:
        run_dir_value = str(r.get("run_dir", "")).strip()
        if not run_dir_value:
            continue
        p = _to_abs_run_dir(session_dir, run_dir_value)
        if p.exists() and p.is_dir():
            run_dir_abs = p
            break

    if run_dir_abs is not None:
        try:
            shutil.rmtree(run_dir_abs)
        except Exception:
            # non blocchiamo: comunque ripuliamo il registry
            logger.debug("Failed to delete run directory %s", run_dir_abs)
            pass

    new_recs = [r for r in recs if str(r.get("run_id", "")).strip() != rid]
    _rewrite_jsonl(index_path, new_recs)
    return True


def delete_energy_run_by_label(session_dir: Path, index_path: Path, label: str) -> Optional[str]:
    """Elimina *tutte* le run con questa label (case-insensitive).

    Ritorna il run_id più recente eliminato (utile per log/UI), oppure None.
    """
    target = normalize_run_label(label).casefold()
    if not target:
        return None

    recs = read_energy_run_registry(index_path)
    if not recs:
        return None

    # ordina: più recenti prima (per determinare il "più recente" da restituire)
    recs_sorted = sorted(recs, key=lambda r: str(r.get("created_at_utc", "")), reverse=True)
    to_delete: List[str] = []
    run_dirs: Dict[str, Path] = {}

    for r in recs_sorted:
        rid = str(r.get("run_id", "")).strip()
        v = str(r.get("run_dir", "")).strip()
        if not rid or not v:
            continue
        p = _to_abs_run_dir(session_dir, v)
        if not p.exists() or not p.is_dir():
            continue
        lab = normalize_run_label(str(r.get("label", "")))
        if not lab:
            lab = get_energy_run_label(p) or ""
        if not lab:
            lab = p.name
        if lab.casefold() != target:
            continue

        if rid not in run_dirs:
            to_delete.append(rid)
            run_dirs[rid] = p

    if not to_delete:
        return None

    # delete dirs
    for rid, p in run_dirs.items():
        try:
            shutil.rmtree(p)
        except Exception:
            logger.debug("Failed to delete run directory %s", p)
            pass

    # rewrite registry once
    keep = [r for r in recs if str(r.get("run_id", "")).strip() not in set(to_delete)]
    _rewrite_jsonl(index_path, keep)

    return to_delete[0]


def update_energy_run_label(session_dir: Path, index_path: Path, run_id: str, new_label: str) -> None:
    """Aggiorna la label di un run (meta + registry)."""
    rid = str(run_id).strip()
    lab = normalize_run_label(new_label)
    if not rid or not lab:
        raise ValueError("run_id e new_label sono obbligatori")

    # aggiorna meta
    recs = read_energy_run_registry(index_path)
    run_dir_abs: Optional[Path] = None
    for r in recs:
        if str(r.get("run_id", "")).strip() == rid:
            v = str(r.get("run_dir", "")).strip()
            if v:
                run_dir_abs = _to_abs_run_dir(session_dir, v)
            break
    if run_dir_abs is not None and run_dir_abs.exists():
        meta = read_energy_run_meta(run_dir_abs) or {}
        meta["run_id"] = meta.get("run_id") or rid
        meta["label"] = lab
        write_energy_run_meta(run_dir_abs, meta)

    # aggiorna tutte le occorrenze nel registry (raro, ma safe)
    changed = False
    for r in recs:
        if str(r.get("run_id", "")).strip() == rid:
            r["label"] = lab
            changed = True
    if changed:
        _rewrite_jsonl(index_path, recs)


# =============================================================================
# Economic run registry
# =============================================================================


@dataclass(frozen=True)
class EconomicRunRecord:
    """Record minimale di un run economico."""

    run_id: str
    created_at_utc: str
    out_dir: str

    energy_run_id: str
    energy_run_dir: str
    econ_scenario_slug: Optional[str]

    # Audit trail (opzionali): fingerprint degli input
    energy_scenario_sha256: Optional[str]
    energy_run_config_sha256: Optional[str]
    econ_scenario_content_sha256: Optional[str]

    kpi_total: Dict[str, Any]


def append_economic_run_record(index_path: Path, record: Dict[str, Any]) -> None:
    """Appende un record (dict) al registry JSONL delle run economiche."""
    _append_jsonl(index_path, record)


def read_economic_run_registry(index_path: Path) -> List[Dict[str, Any]]:
    """Legge il registry JSONL delle run economiche.

Linee corrotte vengono ignorate per tolleranza."""
    return _read_jsonl(index_path)


def set_active_economic_run(active_path: Path, run_id: str) -> None:
    """Imposta il run economico attivo (scrittura atomica)."""
    _set_active(active_path, run_id)


def get_active_economic_run(active_path: Path) -> Optional[str]:
    """Ritorna il run economico attivo, oppure None se non impostato."""
    return _get_active(active_path)
