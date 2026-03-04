# CER Simulator

Streamlit-based simulator for Renewable Energy Communities (CER) and collective self-consumption.
It supports scenario analysis of energy flows (production/consumption/sharing) and economic-financial evaluation, including BESS-related sensitivity analyses.

## Thesis
The full explanation is in file thesis.pdf

**Title:** CER Simulator  
**Candidate:** Filip Juren  
**Supervisor:** Luigi De Paoli  
**Academic Year:** 2024/25

## Main features
- Modular architecture: Streamlit orchestration layer (`src/cer_app`) + core computation modules (`src/cer_core`)
- Session/scenario/run persistence for reproducibility and traceability
- Renewable production modeling (PV / wind) and data acquisition workflows
- Consumption profile generation and aggregation
- Energy balancing and shared energy computation (hourly)
- Economic evaluation with tariffs/incentives assumptions and scenario comparisons
