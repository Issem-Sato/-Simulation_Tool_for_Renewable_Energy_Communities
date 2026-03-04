from __future__ import annotations

"""cer_core.bilanciamento.scenario
===================================

Gestione della persistenza dello **scenario energetico** (input) per il modulo di
bilanciamento CER.

Lo scenario rappresenta il *contratto* tra UI (`cer_app`) e logica di dominio
(`cer_core.bilanciamento`): descrive l'insieme dei membri e i metadati temporali
(periodo) necessari a eseguire una run energetica. Il formato è serializzato su
filesystem come JSON (`scenario.json`) nella directory dello scenario.

Schema (minimo) di `scenario.json`
----------------------------------
Il modulo mantiene uno schema volutamente leggero, con le chiavi minime:

- ``schema_version`` (int): versione dello schema (default: 1)
- ``created_at_utc`` (str, ISO 8601 con suffisso ``Z``): timestamp creazione
- ``updated_at_utc`` (str, ISO 8601 con suffisso ``Z``): timestamp ultima modifica
- ``period`` (dict): metadati periodo (tipicamente ``tz``, ``t0``, ``t1``)
- ``members`` (list[dict]): elenco membri (payload UI; validazione di dettaglio
  avviene nei moduli di bilanciamento, non qui)

Compatibilità legacy
--------------------
Per retro-compatibilità il progetto può includere due file legacy:

- ``members.json`` (lista membri)
- ``period.json`` (metadati periodo)

`migrate_legacy_to_scenario()` consente di creare `scenario.json` da questi file
senza cancellarli.

Fingerprint del contenuto logico
--------------------------------
Il modulo espone un fingerprint SHA-256 deterministico del contenuto *logico* dello
scenario, utile per audit trail e deduplicazione. Per evitare instabilità dovuta ai
timestamp, il fingerprint **esclude** ``created_at_utc`` e ``updated_at_utc`` e
considera solo ``schema_version``, ``period`` e ``members``.

Policy: il fingerprint è **indipendente dall'ordine** dei membri; i membri

vengono ordinati (se possibile) per ``id`` (o ``member_id``) prima dell'hash.

Side-effects su filesystem
--------------------------
- Lettura e scrittura di `scenario.json`
- Scrittura opzionale dei file legacy (write-through) in `update_members`/`update_period`

Le scritture su `scenario.json` sono **atomiche** (write temp + replace) per
ridurre il rischio di corruzione del file in caso di crash/interruzione.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cer_core.bilanciamento.fingerprint import sha256_json_canonical

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _now_utc_iso() -> str:
    """Ritorna un timestamp ISO 8601 in UTC con suffisso ``Z`` e senza microsecondi."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Scrive testo su filesystem in modo atomico (best-effort).

    Strategia:
    - scrive su file temporaneo ``<name>.tmp`` nella stessa directory;
    - rimpiazza il file target con ``Path.replace()``, che su filesystem POSIX è
      un'operazione atomica rispetto ai lettori.

    Parameters
    ----------
    path:
        File target.
    text:
        Contenuto da scrivere.
    encoding:
        Encoding del testo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def _normalize_scenario_inplace(scenario: Dict[str, Any]) -> bool:
    """Normalizza in-place lo scenario garantendo le chiavi e i tipi minimi.

    Normalizzazione applicata:
    - ``schema_version``: int (default: ``SCHEMA_VERSION``)
    - ``members``: list (default: ``[]``)
    - ``period``: dict (default: ``{}``)

    Returns
    -------
    bool
        True se sono state applicate modifiche al dizionario.
    """
    changed = False

    # schema_version
    sv = scenario.get("schema_version", SCHEMA_VERSION)
    try:
        sv_int = int(sv) if sv is not None else SCHEMA_VERSION
    except Exception:
        sv_int = SCHEMA_VERSION
    if scenario.get("schema_version") != sv_int:
        scenario["schema_version"] = sv_int
        changed = True

    # members
    members = scenario.get("members", [])
    if not isinstance(members, list):
        scenario["members"] = []
        changed = True

    # period
    period = scenario.get("period", {})
    if not isinstance(period, dict):
        scenario["period"] = {}
        changed = True

    return changed


def _ensure_timestamps_inplace(scenario: Dict[str, Any]) -> bool:
    """Garantisce la presenza dei timestamp di creazione/aggiornamento.

    Returns
    -------
    bool
        True se sono stati aggiunti/modificati campi.
    """
    changed = False
    now = _now_utc_iso()
    if "updated_at_utc" not in scenario:
        scenario["updated_at_utc"] = now
        changed = True
    if "created_at_utc" not in scenario:
        scenario["created_at_utc"] = scenario.get("updated_at_utc", now)
        changed = True
    return changed


def load_scenario(scenario_path: Path) -> Optional[Dict[str, Any]]:
    """Carica `scenario.json` da filesystem.

    Parameters
    ----------
    scenario_path:
        Path del file `scenario.json`.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dizionario scenario se il file esiste ed è parseabile; None se il file non
        esiste o se la lettura/parsing fallisce.

    Notes
    -----
    - In caso di JSON invalido o errori I/O, la funzione logga l'eccezione e
      ritorna None (comportamento fail-soft per UI).
    - Viene applicata una normalizzazione light (chiavi e tipi minimi), senza
      riscrivere automaticamente il file.
    """
    if not scenario_path.exists():
        return None
    try:
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        if not isinstance(scenario, dict):
            logger.warning("scenario.json is not a JSON object: %s", scenario_path)
            return None
        _normalize_scenario_inplace(scenario)
        return scenario
    except Exception:
        logger.exception("Failed to load scenario: %s", scenario_path)
        return None


def save_scenario(scenario_path: Path, scenario: Dict[str, Any]) -> None:
    """Salva `scenario.json` su filesystem (scrittura atomica).

    La funzione:
    - normalizza lo scenario (schema_version/members/period);
    - aggiorna ``updated_at_utc`` e inizializza ``created_at_utc`` se mancante;
    - serializza in JSON UTF-8 (indent=2) e scrive in modo atomico.

    Parameters
    ----------
    scenario_path:
        Path del file `scenario.json`.
    scenario:
        Dizionario scenario da persistere.

    Side Effects
    ------------
    - Scrittura/overwrite del file target.
    """
    _normalize_scenario_inplace(scenario)
    scenario["updated_at_utc"] = _now_utc_iso()
    if "created_at_utc" not in scenario:
        scenario["created_at_utc"] = scenario["updated_at_utc"]
    payload = json.dumps(scenario, ensure_ascii=False, indent=2)
    _atomic_write_text(scenario_path, payload, encoding="utf-8")


def migrate_legacy_to_scenario(
    scenario_path: Path,
    legacy_members_path: Path,
    legacy_period_path: Path,
) -> Dict[str, Any]:
    """Crea `scenario.json` a partire dai file legacy se necessario.

    Comportamento:
    - Se `scenario.json` esiste:
        - carica e normalizza;
        - salva **solo** se la normalizzazione ha richiesto modifiche o se mancano
          i timestamp di tracciamento.
    - Se `scenario.json` non esiste:
        - tenta di leggere ``members.json`` e ``period.json``;
        - crea un nuovo scenario minimale e lo salva.

    Backward compatible: non elimina i file legacy.

    Returns
    -------
    Dict[str, Any]
        Scenario risultante (in memoria), normalizzato.
    """
    existing = load_scenario(scenario_path)
    if existing is not None:
        changed = _normalize_scenario_inplace(existing)
        changed |= _ensure_timestamps_inplace(existing)
        if changed:
            save_scenario(scenario_path, existing)
        return existing

    members: List[Dict[str, Any]] = []
    period: Dict[str, Any] = {}

    if legacy_members_path.exists():
        try:
            loaded = json.loads(legacy_members_path.read_text(encoding="utf-8"))
            members = loaded if isinstance(loaded, list) else []
        except Exception:
            logger.exception("Failed to load legacy members: %s", legacy_members_path)
            members = []

    if legacy_period_path.exists():
        try:
            loaded = json.loads(legacy_period_path.read_text(encoding="utf-8"))
            period = loaded if isinstance(loaded, dict) else {}
        except Exception:
            logger.exception("Failed to load legacy period: %s", legacy_period_path)
            period = {}

    scenario: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now_utc_iso(),
        "updated_at_utc": _now_utc_iso(),
        "period": period,
        "members": members,
    }
    save_scenario(scenario_path, scenario)
    return scenario


def get_members(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ritorna la lista membri dallo scenario (fallback a lista vuota)."""
    v = scenario.get("members", [])
    return list(v) if isinstance(v, list) else []


def get_period_meta(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Ritorna il dizionario periodo dallo scenario (fallback a dict vuoto)."""
    v = scenario.get("period", {})
    return dict(v) if isinstance(v, dict) else {}


def update_members(
    scenario_path: Path,
    members: List[Dict[str, Any]],
    legacy_members_path: Optional[Path] = None,
) -> None:
    """Aggiorna i membri nello scenario e persiste su disco.

    Parameters
    ----------
    scenario_path:
        Path di `scenario.json`.
    members:
        Lista di membri (lista di dict) da salvare.
    legacy_members_path:
        Se fornito, abilita write-through su `members.json` legacy (retro-compatibilità).

    Side Effects
    ------------
    - Scrive `scenario.json` (atomico).
    - Opzionalmente scrive `members.json` legacy.
    """
    scenario = load_scenario(scenario_path) or {"schema_version": SCHEMA_VERSION, "period": {}, "members": []}
    _normalize_scenario_inplace(scenario)
    scenario["members"] = members
    save_scenario(scenario_path, scenario)

    if legacy_members_path is not None:
        try:
            payload = json.dumps(members, ensure_ascii=False, indent=2)
            _atomic_write_text(legacy_members_path, payload, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write legacy members: %s", legacy_members_path)


def update_period(
    scenario_path: Path,
    period_meta: Dict[str, Any],
    legacy_period_path: Optional[Path] = None,
) -> None:
    """Aggiorna il periodo nello scenario e persiste su disco.

    Parameters
    ----------
    scenario_path:
        Path di `scenario.json`.
    period_meta:
        Metadati periodo (dict) da salvare.
    legacy_period_path:
        Se fornito, abilita write-through su `period.json` legacy (retro-compatibilità).

    Side Effects
    ------------
    - Scrive `scenario.json` (atomico).
    - Opzionalmente scrive `period.json` legacy.
    """
    scenario = load_scenario(scenario_path) or {"schema_version": SCHEMA_VERSION, "period": {}, "members": []}
    _normalize_scenario_inplace(scenario)
    scenario["period"] = period_meta
    save_scenario(scenario_path, scenario)

    if legacy_period_path is not None:
        try:
            payload = json.dumps(period_meta, ensure_ascii=False, indent=2)
            _atomic_write_text(legacy_period_path, payload, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write legacy period: %s", legacy_period_path)


# -----------------------------------------------------------------------------
# Fingerprint (audit trail)
# -----------------------------------------------------------------------------


def _member_sort_key_for_fingerprint(m: Dict[str, Any]) -> Tuple[int, Any]:
    """Chiave di ordinamento robusta per i membri nel fingerprint.

    Ordina prioritariamente per campo `id` (o `member_id`) se castabile. In caso
    contrario ricade su una rappresentazione stringa stabile.

    Returns
    -------
    tuple
        (0, int_id) se disponibile, altrimenti (1, str_fallback)
    """
    raw = m.get("id", m.get("member_id", None))
    if raw is None:
        return (2, json.dumps(m, ensure_ascii=False, sort_keys=True))
    try:
        return (0, int(raw))
    except Exception:
        return (1, str(raw))


def scenario_content_fingerprint_sha256(scenario: Dict[str, Any]) -> str:
    """Calcola un fingerprint SHA-256 deterministico del contenuto logico.

    L'hash considera esclusivamente:
    - ``schema_version``
    - ``period``
    - ``members``

    I membri sono ordinati (se possibile) per `id`/`member_id` per rendere il
    fingerprint stabile rispetto a reorder accidentali (order-insensitive).

    Parameters
    ----------
    scenario:
        Scenario in memoria.

    Returns
    -------
    str
        Digest SHA-256 esadecimale del JSON canonicalizzato.
    """
    members = get_members(scenario)
    try:
        members_sorted = sorted(
            (m for m in members if isinstance(m, dict)),
            key=_member_sort_key_for_fingerprint,
        )
    except Exception:
        members_sorted = members  # fallback: order-sensitive

    payload = {
        "schema_version": int(scenario.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
        "period": get_period_meta(scenario),
        "members": members_sorted,
    }
    return sha256_json_canonical(payload)


def scenario_file_content_fingerprint_sha256(scenario_path: Path) -> Optional[str]:
    """Calcola il fingerprint del contenuto logico di `scenario.json` su disco.

    Returns
    -------
    Optional[str]
        Fingerprint se il file esiste ed è valido; None altrimenti.
    """
    sc = load_scenario(scenario_path)
    if sc is None:
        return None
    try:
        return scenario_content_fingerprint_sha256(sc)
    except Exception:
        logger.exception("Failed to fingerprint scenario: %s", scenario_path)
        return None
