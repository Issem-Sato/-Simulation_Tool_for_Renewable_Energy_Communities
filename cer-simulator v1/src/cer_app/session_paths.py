from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# =============================================================================
# Session root pointers (shared across Streamlit pages)
# =============================================================================

SESSIONS_ROOT = Path("data/sessions")
CURRENT_FILE = Path("data/.current_session")


def normalize_session_name(name: str) -> str:
    """Normalizza un nome sessione inserito dall'utente.

    Regole:
    - conserva solo caratteri "sicuri" (lettere, numeri, underscore, trattino)
    - converte spazi/whitespace in underscore
    - evita path traversal ("..", "/", "\\")
    """
    import re

    n = (name or "").strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^A-Za-z0-9_\-]", "", n)
    # hardening
    n = n.replace("..", "_")
    n = n.replace("/", "_").replace("\\", "_")
    return n or "default"


def list_sessions() -> list[str]:
    """Elenca le sessioni disponibili (cartelle sotto data/sessions)."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for p in sorted(SESSIONS_ROOT.glob("*")):
        if p.is_dir() and not p.name.startswith("."):
            out.append(p.name)
    if "default" not in out:
        out.insert(0, "default")
    return out


def get_current_session_name() -> str:
    """Ritorna il nome della sessione corrente (come salvato in data/.current_session)."""
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CURRENT_FILE.exists():
        name = CURRENT_FILE.read_text(encoding="utf-8").strip()
        if name:
            return name
    return "default"


def set_current_session(name: str) -> Path:
    """Imposta la sessione corrente e crea la cartella se manca."""
    n = normalize_session_name(name)
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    d = SESSIONS_ROOT / n
    d.mkdir(parents=True, exist_ok=True)
    (d / "cache").mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.write_text(n, encoding="utf-8")
    return d


def get_current_session_dir() -> Path:
    """Return the current session directory.

    - Nome sessione salvato in ``data/.current_session``
    - Root sessioni: ``data/sessions/<session>/``
    - Timebase master del progetto: UTC (la gestione del tz è nei dati, non nel path)
    """
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CURRENT_FILE.exists():
        name = CURRENT_FILE.read_text(encoding="utf-8").strip()
        if name:
            return set_current_session(name)

    # default
    return set_current_session("default")


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path

    # Common
    cache_dir: Path

    # Scenario (source of truth)
    scenario_json: Path

    # Meteo
    session_meta_json: Path
    meteo_hourly_csv: Path
    meteo_daily_csv: Path

    # Bilanciamento
    bil_dir: Path
    bil_inputs_dir: Path
    bil_outputs_dir: Path
    bil_members_json: Path
    bil_period_json: Path
    bil_runs_index_jsonl: Path
    bil_active_run_txt: Path

    # Economics
    econ_dir: Path
    econ_outputs_dir: Path
    econ_scenarios_dir: Path
    econ_active_scenario_txt: Path

    econ_runs_index_jsonl: Path
    econ_active_run_txt: Path


def get_paths(session_dir: Optional[Path] = None) -> SessionPaths:
    """Costruisce i path standard della sessione (e crea le cartelle minime).

    Nota: questo modulo NON decide il contenuto dei file; definisce solo convenzioni
    di percorso, usate in modo consistente dalle pagine Streamlit.
    """
    d = session_dir or get_current_session_dir()
    d.mkdir(parents=True, exist_ok=True)

    cache_dir = d / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Scenario
    scenario_json = d / "scenario.json"

    # Meteo
    session_meta_json = d / "session_meta.json"
    meteo_hourly_csv = d / "meteo_hourly.csv"
    meteo_daily_csv = d / "meteo_daily.csv"

    # Bilanciamento
    bil_dir = d / "bilanciamento"
    bil_inputs_dir = bil_dir / "inputs"
    bil_outputs_dir = bil_dir / "outputs"
    bil_dir.mkdir(parents=True, exist_ok=True)
    bil_inputs_dir.mkdir(parents=True, exist_ok=True)
    bil_outputs_dir.mkdir(parents=True, exist_ok=True)

    bil_members_json = bil_dir / "members.json"
    bil_period_json = bil_dir / "period.json"
    bil_runs_index_jsonl = bil_dir / "runs_index.jsonl"
    bil_active_run_txt = bil_dir / "active_run.txt"

    # Economics
    econ_dir = d / "economics"
    econ_outputs_dir = econ_dir / "outputs"
    econ_scenarios_dir = econ_dir / "scenarios"
    econ_active_scenario_txt = econ_dir / "active_scenario.txt"

    econ_runs_index_jsonl = econ_dir / "runs_index.jsonl"
    econ_active_run_txt = econ_dir / "active_run.txt"

    econ_dir.mkdir(parents=True, exist_ok=True)
    econ_outputs_dir.mkdir(parents=True, exist_ok=True)
    econ_scenarios_dir.mkdir(parents=True, exist_ok=True)

    return SessionPaths(
        session_dir=d,
        cache_dir=cache_dir,
        scenario_json=scenario_json,
        session_meta_json=session_meta_json,
        meteo_hourly_csv=meteo_hourly_csv,
        meteo_daily_csv=meteo_daily_csv,
        bil_dir=bil_dir,
        bil_inputs_dir=bil_inputs_dir,
        bil_outputs_dir=bil_outputs_dir,
        bil_members_json=bil_members_json,
        bil_period_json=bil_period_json,
        bil_runs_index_jsonl=bil_runs_index_jsonl,
        bil_active_run_txt=bil_active_run_txt,
        econ_dir=econ_dir,
        econ_outputs_dir=econ_outputs_dir,
        econ_scenarios_dir=econ_scenarios_dir,
        econ_active_scenario_txt=econ_active_scenario_txt,
        econ_runs_index_jsonl=econ_runs_index_jsonl,
        econ_active_run_txt=econ_active_run_txt,
    )
