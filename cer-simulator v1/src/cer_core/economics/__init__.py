"""cer_core.economics

Motore di valutazione economico-finanziaria per la CER.

Il package è pensato per essere orchestrato da una pagina Streamlit (es.
``cer_app/pages/4_Valutazione_Economica.py``) che prepara i dataset di input
(tariffe, asset, fiscalità, policy di riparto) e richiama le funzioni qui
esposte.
"""

from .economic_model import (  # noqa: F401
    EnergyRunData,
    EconomicsAssumptions,
    EconomicsResult,
    list_energy_runs,
    load_energy_run,
    evaluate_economics,
    save_economic_outputs,
)
